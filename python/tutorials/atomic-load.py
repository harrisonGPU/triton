import torch
import triton
import triton.language as tl


@triton.jit
def atomic_rmw_load_kernel(ptr, out, N: tl.constexpr):
    offsets = tl.arange(0, N)
    # Simulate an atomic load by adding 0, using acquire semantics.
    val = tl.atomic_add(ptr + offsets, 0, sem="acquire", scope="gpu")
    tl.store(out + offsets, val)


@triton.jit
def atomic_load_kernel(ptr, out, N: tl.constexpr):
    offsets = tl.arange(0, N)
    val = tl.atomic_load(ptr + offsets, sem="acquire", scope="gpu")
    tl.store(out + offsets, val)


def main():
    N = 8
    device = "cuda"

    torch.manual_seed(0)
    x = torch.randint(0, 1000, (N,), dtype=torch.int32, device=device)

    out_rmw = torch.zeros_like(x)
    out_load = torch.zeros_like(x)

    atomic_rmw_load_kernel[(1,)](x, out_rmw, N=N)
    atomic_load_kernel[(1,)](x, out_load, N=N)

    print("input        :", x.tolist())
    print("atomic_add(0):", out_rmw.tolist())
    print("atomic_load  :", out_load.tolist())

    match = torch.equal(out_rmw, out_load)
    print("results match:", match)
    assert match, "atomic_add(ptr, 0) and atomic_load produced different results!"


if __name__ == "__main__":
    main()
