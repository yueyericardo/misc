import argparse
import time
import torch
from torch.profiler import record_function, ProfilerActivity


torch._C._jit_set_nvfuser_single_node_mode(True)
torch._C._debug_set_autodiff_subgraph_inlining(False)


# -----------------------------------------------------------------------
# benchmark utils


def timeit(
    func, *args, steps=200, warmup=10, show_profile=False, label=None, label_padding=35
):
    if label is None:
        assert func.__name__, "please provide a label for this benchmark"
        label = func.__name__

    # warmup
    torch.cuda.nvtx.range_push(f"{label}_warmup")
    for _ in range(warmup):
        func(*args)
    torch.cuda.nvtx.range_pop()  # pop label_warmup

    # start timer
    torch.cuda.synchronize()
    start = time.time()

    torch.cuda.nvtx.range_push(f"{label}")
    if show_profile:
        print("\n" + "=" * 70 + " " + label + " " + "=" * 70)
        with torch.profiler.profile(
            activities=[ProfilerActivity.CUDA], use_cuda=True
        ) as prof:
            with record_function("run_total"):
                for i in range(steps):
                    torch.cuda.nvtx.range_push(f"{i}th_iteration")
                    func(*args)
                    torch.cuda.nvtx.range_pop()
        print(
            prof.key_averages().table(
                sort_by="self_cuda_time_total", max_src_column_width=200, row_limit=15
            )
        )
    else:
        for i in range(steps):
            torch.cuda.nvtx.range_push(f"{i}th_iteration")
            func(*args)
            torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()  # pop label

    # stop timer
    torch.cuda.synchronize()
    time_ms = ((time.time() - start) / steps) * 1000

    print(f"{label.ljust(label_padding)}: {time_ms:.3f} ms/step")


def profile(func, *args, steps=30, warmup=10, label=None, label_padding=35):
    """
    Simply a convenient wrapper of the timeit function with profile=True.
    """
    return timeit(
        func,
        *args,
        steps=steps,
        warmup=warmup,
        show_profile=True,
        label=label,
        label_padding=label_padding,
    )


# -----------------------------------------------------------------------
# addmm_silu


def silu(x):
    return torch.nn.functional.silu(x) * x.sin()


silu_scripted = torch.jit.script(silu)
from custom_silu import MySiLU
# from silu_fused_wrong import FusedSiLU

silu_myfused = lambda x: MySiLU.apply(x) * x.sin()

silu.__name__ = "silu"
silu_scripted.__name__ = "silu_scripted"
silu_myfused.__name__ = "silu_myfused"


# -----------------------------------------------------------------------
# some inputs
device = "cuda"
batch_size = 10000
x = torch.rand([batch_size, 512], device=device, requires_grad=True)
weight0 = torch.rand([512, 512], device=device, requires_grad=True)
bias0 = torch.rand([1, 512], device=device, requires_grad=True)
weight1 = torch.rand([512, 512], device=device, requires_grad=True)
bias1 = torch.rand([1, 512], device=device, requires_grad=True)
I_N = torch.ones_like(x)

# print(addmm_silu_scripted.graph_for(x))


def run(func, *args):
    torch.cuda.nvtx.range_push("forward")
    y = func(*args)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("1st order")
    (y__x,) = torch.autograd.grad(y, [x], I_N, create_graph=True, allow_unused=True)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("2nd order")
    (y__x__x,) = torch.autograd.grad(
        y__x, [x], I_N, create_graph=True, allow_unused=True
    )
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("3rd order")
    (y__x__x__x,) = torch.autograd.grad(
        y__x__x, [x], I_N, create_graph=True, allow_unused=True
    )
    torch.cuda.nvtx.range_pop()

    # torch.cuda.nvtx.range_push("backward")
    # (y__x__x.sum() + y__x.sum()).backward()
    # torch.cuda.nvtx.range_pop()
    return y, y__x, y__x__x, y__x__x__x


def validate_result():
    # test functorch result
    res_ref = run(silu, x)
    res_scripted = run(silu_scripted, x)
    res_myfused = run(silu_myfused, x)
    num = len(res_ref)
    for i in range(num):
        # print(f"max_error_{i}: scripted: {(res_ref[i] - res_scripted[i]).max():.2e}")
        print(f"max_error_{i}: myfused: {(res_ref[i] - res_myfused[i]).max():.2e}")
        assert torch.allclose(res_ref[i], res_scripted[i], atol=5e-5)
        assert torch.allclose(res_ref[i], res_myfused[i], atol=5e-5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--profile", default=False, action="store_true")
    parser.add_argument("--half", default=False, action="store_true")
    args = parser.parse_args()

    validate_result()

    with torch.cuda.amp.autocast(enabled=args.half):
        if args.half:
            x = x.half()
            I_N = I_N.half()
        if args.profile:
            profile(run, silu, x, label="silu")
            profile(run, silu_scripted, x, label="silu_scripted")
            # profile(run, silu_fusion, x, label="silu_fusion")
            profile(run, silu_myfused, x, label="silu_myfused")
        else:
            timeit(run, silu, x, label="silu")
            timeit(run, silu_scripted, x, label="silu_scripted")
            # timeit(run, silu_fusion, x, label="silu_fusion")
            timeit(run, silu_myfused, x, label="silu_myfused")
