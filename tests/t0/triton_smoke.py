"""Smallest Triton kernel on the Spark: fp32 vector add must be exact (H0 criterion).

Run as a script (`tools/spark/py.sh tests/t0/triton_smoke.py`) or through
test_triton_smoke.py. Needs torch with CUDA and triton; not importable on the Mac.
"""


def main():
    import torch
    import triton
    import triton.language as tl
    import triton.testing

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    n = 1 << 20
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(n, device="cuda", dtype=torch.float32, generator=g)
    b = torch.randn(n, device="cuda", dtype=torch.float32, generator=g)
    out = torch.empty_like(a)

    def launch():
        add_kernel[(triton.cdiv(n, 1024),)](a, b, out, n, BLOCK=1024)

    launch()
    torch.cuda.synchronize()
    assert torch.equal(out, a + b), "triton add differs from torch add"
    ms = triton.testing.do_bench(launch)
    assert ms > 0, "do_bench returned a non-positive time"
    print(f"triton smoke OK {torch.__version__} {triton.__version__}")


if __name__ == "__main__":
    main()
