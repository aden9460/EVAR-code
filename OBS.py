import math
import time
import os
import torch
import torch.nn as nn
import transformers

import matplotlib.pyplot as plt

DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class OBS(object):
    def __init__(self, layer, layer_idx, args):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

        self.args = args
        self.no_compensate = args.no_compensate

        # Gradient accumulators for Taylor method
        self.grad_first = None   # First-order gradient (from final backward pass)
        self.grad_second = None  # Second-order gradient accumulation (grad² from multiple backward passes)
        self.taylor_samples = 0  # Number of processed samples

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out                

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()  # [hsize, seqlen]
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def struct_prune(
        self, sparsity, headsize=1, percdamp=0.0, layer_idx=None, 
    ):
        assert self.columns % headsize == 0

        tick = time.time()
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if percdamp > 0:
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(H.size(0), device=self.dev)
            H[diag, diag] += damp

        column_mask = torch.zeros(self.columns, dtype=torch.bool, device=self.dev) # 1 for remove
        pruned_columns = column_mask.count_nonzero()
        target_columns = round(self.columns // headsize * sparsity) * headsize

        if headsize > 1:
            pass
        else:
            blocksize = (target_columns - 512) // 2

        while pruned_columns < target_columns:     
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
            if headsize > 1:
                Hinv_diag = torch.stack([Hinv[i:i+headsize, i:i+headsize] for i in range(0, self.columns, headsize)])
                Hinv_diag = torch.diagonal(torch.linalg.cholesky(Hinv_diag), dim1=-2, dim2=-1).reshape(-1)
                Hinv_diag = Hinv_diag ** 2
            else:
                Hinv_diag = Hinv.diag()

            error = torch.sum(W ** 2 / Hinv_diag.unsqueeze(0), dim=0)
            error[column_mask] = torch.inf
            if headsize > 1:
                head_sort_idx = error.view(-1, headsize).sum(1).argsort()
                column_sort_idx = torch.hstack([torch.arange(x * headsize, x * headsize + headsize) for x in head_sort_idx])
                cnt = headsize
            else:
                column_sort_idx = error.argsort()
                cnt = min(target_columns - pruned_columns, max(blocksize, 64), 1024)

            W = W[:, column_sort_idx]
            Hinv = Hinv[column_sort_idx, :][:, column_sort_idx]
            Hinv = torch.linalg.cholesky(Hinv, upper=True)[:cnt]
            
            W1 = W[:, :cnt].clone()
            Hinv1 = Hinv[:, :cnt]
            Err1 = torch.zeros_like(W1)

            for i in range(cnt):
                Err1[:, i:i+1] = W1[:, i:i+1] / Hinv1[i, i]
                if not self.no_compensate:
                    W1[:, i:] -= Err1[:, i:i+1].matmul(Hinv1[i:i+1, i:])  # local update

            W[:, :cnt] = 0
            if not self.no_compensate:
                end = self.columns - pruned_columns
                W[:, cnt:end] -= Err1.matmul(Hinv[:, cnt:end])  # global update

            column_sort_idx_inv = torch.argsort(column_sort_idx)
            W = W[:, column_sort_idx_inv]

            pruned_idx = column_sort_idx[:cnt]
            H[pruned_idx, :] = H[:, pruned_idx] = 0
            H[pruned_idx, pruned_idx] = 1
            column_mask[pruned_idx] = 1
            pruned_columns += cnt

            if headsize > 1:
                pass
            else:
                blocksize = (blocksize - 512) // 2

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        # print('time %.2f' % (time.time() - tick), flush=True)
        print('pruned columns %d/%d' % ((self.layer.weight.sum(0) == 0).sum().item(), self.layer.weight.size(1)), flush=True)

        if DEBUG:
            out_gap = torch.mean((self.layer(self.inp1) - self.out1) ** 2).item()
            out = torch.mean(self.out1 ** 2).item()
            print('output_gap:', out_gap, flush=True)
            print('output:', out, flush=True)
            print('output_gap / output:', out_gap / out, flush=True)

        # Return the indices of pruned columns for Torch-Pruning integration
        pruned_indices = column_mask.nonzero(as_tuple=True)[0]
        return pruned_indices

    def accumulate_hessian_diag(self):
        """
        Accumulate Hessian diagonal approximation (second-order gradient = grad²)

        Call timing: After each loss.backward(), before zero_grad()
        """
        if self.layer.weight.grad is None:
            print(f"    WARNING: gradient is None for layer {self.layer.__class__.__name__}, skipping accumulation")
            return

        grad = self.layer.weight.grad.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            grad = grad.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            grad = grad.t()

        # Validate gradients
        grad_mean = grad.mean().item()
        grad_std = grad.std().item()
        grad_max = grad.abs().max().item()

        if grad_max < 1e-8:
            print(f"    WARNING: very small gradients (max={grad_max:.2e}), this may affect pruning quality")

        # Gradient squared
        grad_squared = grad ** 2

        # Accumulate
        if self.grad_second is None:
            self.grad_second = grad_squared
        else:
            self.grad_second += grad_squared

        self.taylor_samples += 1

        # Print statistics every 5 samples
        if self.taylor_samples % 5 == 0:
            print(f"    Taylor sample {self.taylor_samples}: grad mean={grad_mean:.2e}, std={grad_std:.2e}, max={grad_max:.2e}")

    def finalize_hessian_diag(self):
        """
        Normalize accumulated Hessian diagonal

        Call timing: After all samples are processed
        """
        if self.grad_second is not None and self.taylor_samples > 0:
            self.grad_second /= self.taylor_samples

    def capture_first_order_grad(self):
        """
        Capture first-order gradient (for param_first and param_mix)

        Call timing: After final loss.backward()
        """
        if self.layer.weight.grad is None:
            print(f"    WARNING: gradient is None for layer {self.layer.__class__.__name__}, cannot capture first-order gradient")
            return

        grad = self.layer.weight.grad.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            grad = grad.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            grad = grad.t()

        # Validate gradients
        grad_mean = grad.mean().item()
        grad_std = grad.std().item()
        grad_max = grad.abs().max().item()

        print(f"    First-order grad: mean={grad_mean:.2e}, std={grad_std:.2e}, max={grad_max:.2e}")

        self.grad_first = grad

    def taylor_prune_llm(self, sparsity, headsize=64, percdamp=0.01,
                         layer_idx=0, taylor_type='param_mix'):
        """
        Structured pruning based on LLM-Pruner Taylor importance

        Supports three Taylor variants:
        - param_first: I = |w · ∂L/∂w| (first-order)
        - param_second: I = |w · H_ii · w| (pure second-order)
        - param_mix: I = |w · ∂L/∂w - 0.5 · w · H_ii · w| (mixed, recommended)

        Args:
            sparsity: Pruning ratio (0-1)
            headsize: Head size (VAR uses fixed 64)
            percdamp: Reserved parameter (not used for Taylor)
            layer_idx: Layer index (for logging)
            taylor_type: 'param_first', 'param_second', 'param_mix'

        Returns:
            prune_indices: Column indices to prune [Tensor]
        """
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        # Calculate salience (importance)
        if taylor_type == 'param_first':
            # First-order: S = w · ∂L/∂w
            if self.grad_first is None:
                raise ValueError("First order gradient not captured. Call capture_first_order_grad() first.")
            salience = W * self.grad_first

        elif taylor_type == 'param_second':
            # Pure second-order: S = w · H_ii · w
            if self.grad_second is None:
                raise ValueError("Second order gradient not accumulated. Call accumulate_hessian_diag() during training.")
            salience = W * self.grad_second * W

        elif taylor_type == 'param_mix':
            # Mixed: S = w · ∂L/∂w - 0.5 · w · H_ii · w
            if self.grad_first is None or self.grad_second is None:
                raise ValueError("Both first and second order gradients required for param_mix.")
            salience = W * self.grad_first - 0.5 * W * self.grad_second * W

        else:
            raise ValueError(f"Unknown taylor_type: {taylor_type}. Must be 'param_first', 'param_second', or 'param_mix'.")

        # Aggregate to output channels (sum across input dimension)
        importance = salience.abs().sum(dim=1)  # [out_channels]

        # Group by head (VAR specific)
        if headsize > 1:
            num_heads = importance.shape[0] // headsize
            assert importance.shape[0] % headsize == 0, f"out_channels={importance.shape[0]} must be divisible by headsize={headsize}"

            head_importance = importance.view(num_heads, headsize).sum(dim=1)  # [num_heads]

            # Select heads to prune (lowest importance) - Fixed: use round instead of int
            num_prune_heads = round(num_heads * sparsity)
            if num_prune_heads == 0:
                print(f"    Taylor {taylor_type}: Layer {layer_idx}, sparsity too low, no heads pruned")
                return torch.tensor([], dtype=torch.long, device=W.device)

            if num_prune_heads >= num_heads:
                print(f"    Taylor {taylor_type}: Layer {layer_idx}, sparsity too high, pruning {num_heads-1}/{num_heads} heads")
                num_prune_heads = num_heads - 1  # Keep at least 1 head

            prune_head_indices = head_importance.argsort()[:num_prune_heads]

            # Convert to channel indices
            prune_indices = []
            for head_idx in prune_head_indices:
                prune_indices.extend(range(head_idx * headsize, (head_idx + 1) * headsize))
            prune_indices = torch.tensor(prune_indices, dtype=torch.long, device=W.device)

            # Detailed logging
            actual_sparsity = num_prune_heads / num_heads
            print(f"    Taylor {taylor_type}: Layer {layer_idx}")
            print(f"      Requested sparsity: {sparsity:.3f} ({sparsity*num_heads:.1f} heads)")
            print(f"      Actual sparsity: {actual_sparsity:.3f} ({num_prune_heads}/{num_heads} heads)")
            print(f"      Head importance range: [{head_importance.min().item():.6f}, {head_importance.max().item():.6f}]")
            print(f"      Pruned heads: {prune_head_indices.tolist()}")
            print(f"      Kept heads: {sorted(set(range(num_heads)) - set(prune_head_indices.tolist()))}")

        else:
            # headsize=1, prune by column directly
            num_prune = round(self.columns * sparsity)  # Fixed: use round instead of int
            if num_prune == 0:
                return torch.tensor([], dtype=torch.long, device=W.device)

            prune_indices = importance.argsort()[:num_prune]
            print(f"    Taylor {taylor_type}: Layer {layer_idx}, pruned {num_prune}/{self.columns} columns")

        # Execute pruning (zero out)
        W[:, prune_indices] = 0
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        print(f"    Pruned columns {(self.layer.weight.sum(0) == 0).sum().item()}/{self.layer.weight.size(1)}")

        return prune_indices

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        # Release Taylor-related memory
        self.grad_first = None
        self.grad_second = None
        torch.cuda.empty_cache()


    def magnitude_prune(self, sparsity, percdamp, headsize, layer_idx):
        """
        Magnitude-based pruning: prune channels/heads with smallest weight magnitude.

        Args:
            sparsity: target pruning ratio
            percdamp: not used (kept for API consistency)
            headsize: if > 1, prune by heads; if == 1, prune by columns
            layer_idx: not used (kept for API consistency)

        Returns:
            prune_col_idx: torch.LongTensor of pruned column indices
        """
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if headsize > 1:
            num_heads = W.shape[1] // headsize
            assert W.shape[1] % headsize == 0, "Column count must be divisible by headsize"
            # Calculate number of heads to prune
            target_heads = round(num_heads * sparsity)
            # Calculate head scores (sum of absolute values per head)
            head_scores = W.abs().reshape(W.shape[0], num_heads, headsize).sum(dim=(0, 2))  # [num_heads]
            prune_head_idx = torch.argsort(head_scores)[:target_heads]  # heads to prune
            # Get column indices for all pruned heads
            prune_col_idx = []
            for h in prune_head_idx:
                prune_col_idx.extend(range(h * headsize, (h + 1) * headsize))
            prune_col_idx = torch.tensor(prune_col_idx, device=W.device)
        else:
            # headsize=1, prune by columns directly
            num_prune = round(W.shape[1] * sparsity)
            col_scores = W.abs().sum(dim=0)
            prune_col_idx = torch.argsort(col_scores)[:num_prune]

        # Prune (zero out)
        W[:, prune_col_idx] = 0
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        print('pruned columns %d/%d' % ((self.layer.weight.sum(0) == 0).sum().item(), self.layer.weight.size(1)), flush=True)
        return prune_col_idx

    def taylor_prune(self, sparsity, percdamp, headsize, layer_idx):
        """
        Taylor-based pruning: prune channels/heads with smallest |W * grad| score.

        Args:
            sparsity: target pruning ratio
            percdamp: not used (kept for API consistency)
            headsize: if > 1, prune by heads; if == 1, prune by columns
            layer_idx: not used (kept for API consistency)

        Returns:
            prune_col_idx: torch.LongTensor of pruned column indices
        """
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        # Need gradients
        if self.layer.weight.grad is None:
            raise RuntimeError("Taylor pruning requires gradients from backward pass")
        grad = self.layer.weight.grad.clone()
        if isinstance(self.layer, nn.Conv2d):
            grad = grad.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            grad = grad.t()
        grad = grad.float()

        if headsize > 1:
            num_heads = W.shape[1] // headsize
            assert W.shape[1] % headsize == 0, "Column count must be divisible by headsize"
            target_heads = round(num_heads * sparsity)
            # Calculate Taylor scores per head
            taylor_scores = (W * grad).abs().reshape(W.shape[0], num_heads, headsize).sum(dim=(0, 2))  # [num_heads]
            prune_head_idx = torch.argsort(taylor_scores)[:target_heads]
            prune_col_idx = []
            for h in prune_head_idx:
                prune_col_idx.extend(range(h * headsize, (h + 1) * headsize))
            prune_col_idx = torch.tensor(prune_col_idx, device=W.device)
        else:
            num_prune = round(W.shape[1] * sparsity)
            taylor_scores = (W * grad).abs().sum(dim=0)
            prune_col_idx = torch.argsort(taylor_scores)[:num_prune]

        # Prune (zero out)
        W[:, prune_col_idx] = 0
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        print('pruned columns %d/%d' % ((self.layer.weight.sum(0) == 0).sum().item(), self.layer.weight.size(1)), flush=True)
        return prune_col_idx
