"""Row alignment for the grouped MoE kernel: a deterministic torch counting sort.

Replaces vLLM's `_C` `moe_align_block_size`. Rows are sorted by local expert (stable, so the
order inside an expert is the row order), each expert's rows are padded to `block_m` with the
sentinel R, and every block of `block_m` positions belongs to exactly one expert. No atomics,
no host sync: all outputs have a size that depends only on (R, num_local, block_m).
"""
import torch


def padded_rows(r: int, num_local: int, block_m: int) -> int:
    """Upper bound of the padded length, rounded up to a whole number of blocks."""
    n = r + num_local * (block_m - 1)
    return (n + block_m - 1) // block_m * block_m


def moe_align(local_expert: torch.Tensor, num_local: int, block_m: int):
    """local_expert int32 [R] (-1 = not in the current group) -> (sorted_rows int32, block_expert int32, num_padded int32 [1]).

    sorted_rows [P] holds row ids, sentinel R in the padding; block_expert [P // block_m] holds the
    local expert of each block (-1 past num_padded); num_padded is the used length (a multiple of block_m).
    """
    if local_expert.dtype != torch.int32 or local_expert.dim() != 1:
        raise ValueError("local_expert must be int32 [R]")
    if num_local < 1 or block_m < 1:
        raise ValueError("num_local and block_m must be positive")
    r = local_expert.numel()
    dev = local_expert.device
    p = padded_rows(r, num_local, block_m)
    key = torch.where(local_expert >= 0, local_expert, torch.full_like(local_expert, num_local)).to(torch.int64)
    order = torch.sort(key, stable=True).indices            # row ids by (expert, row)
    counts = torch.bincount(key, minlength=num_local + 1)[:num_local]
    padded = (counts + block_m - 1) // block_m * block_m
    pad_end = torch.cumsum(padded, 0)
    pad_start = pad_end - padded
    start = torch.cumsum(counts, 0) - counts
    ks = key[order]
    valid = ks < num_local
    ks_c = ks.clamp(max=num_local - 1)
    dest = pad_start[ks_c] + torch.arange(r, device=dev) - start[ks_c]
    sorted_rows = torch.full((p,), r, dtype=torch.int32, device=dev)
    sorted_rows[dest[valid]] = order[valid].to(torch.int32)
    nb = p // block_m
    pos = torch.arange(nb, device=dev, dtype=torch.int64) * block_m
    idx = torch.searchsorted(pad_end, pos, right=True)
    block_expert = torch.where(idx < num_local, idx, torch.full_like(idx, -1)).to(torch.int32)
    num_padded = pad_end[-1:].to(torch.int32)
    return sorted_rows, block_expert, num_padded
