import time
from functorch import vmap, jacrev, jacfwd, vjp, jvp
import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = False

_ = torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
D1 = 2  # x, y
D2 = 3  # u, v, p
B = 5000
x = torch.randn(B, D1).to(device)
I_N = torch.eye(D2).to(device)

model = nn.Sequential(
    nn.Linear(D1, 512), nn.SiLU(),
    nn.Linear(512, 512), nn.SiLU(),
    nn.Linear(512, 512), nn.SiLU(),
    nn.Linear(512, 512), nn.SiLU(),
    nn.Linear(512, 512), nn.SiLU(),
    nn.Linear(512, 512), nn.SiLU(),
    nn.Linear(512, D2),
).to(device)


def predict(x):
    torch.cuda.nvtx.range_push("forward")
    out = model(x)
    torch.cuda.nvtx.range_pop()
    return out, out  # return two outputs is needed for jacrev auxiliary object


def reference_jacobian():
    x_ = x.clone().requires_grad_()
    ones = torch.ones(B, device=x.device)
    pred, _ = predict(x_)
    jacobian_rows = [None] * D2
    for i in range(D2):
        torch.cuda.nvtx.range_push("autograd")
        jacobian_rows[i] = torch.autograd.grad(pred[:, i], x_, ones, create_graph=True)[0]
        torch.cuda.nvtx.range_pop()
    jacobian = torch.stack(jacobian_rows)  # [D2, B, D1]
    # l = jacobian.sum()
    # l.backward()
    return jacobian.transpose(0, 1), pred


def functorch_jacobian():
    x_ = x.clone().requires_grad_()
    jacobian, pred = vmap(jacrev(predict, argnums=0, has_aux=True), in_dims=0)(x_)  # [B, D2, D1]
    # l = jacobian.sum()
    # l.backward()
    return jacobian, pred


def vjp_jacobian():
    x_ = x.clone().requires_grad_()
    def predict(x):
        out = model(x)
        return out
    def get_vjp(x, v):
        (pred, vjpfunc) = vjp(predict, x)
        return vjpfunc(v)[0], pred
    jacobian, pred = vmap(vmap(get_vjp, in_dims=(None, 0)), in_dims=(0, None))(x_, I_N)
    pred = pred[:, 0, :]
    # l = jacobian.sum()
    # l.backward()
    return jacobian, pred


def reference_hessian():
    x_ = x.clone().requires_grad_()
    ones = torch.ones(B, device=x.device)
    pred, _ = predict(x_)
    jacobian_rows = [None] * D2
    hessian_rows = [None] * (D2 * D1)
    for i in range(D2):
        torch.cuda.nvtx.range_push("autograd jacobian")
        jacobian_rows[i] = torch.autograd.grad(pred[:, i], x_, ones, create_graph=True)[0]
        torch.cuda.nvtx.range_pop()

    # for i in range(D2 - 1):
    for i in range(D2):
        for j in range(D1):
            torch.cuda.nvtx.range_push("autograd hesian")
            hessian_rows[i * D1 + j] = torch.autograd.grad(jacobian_rows[i][:, j], x_, ones, create_graph=True)[0]
            torch.cuda.nvtx.range_pop()

    jacobian = torch.stack(jacobian_rows)  # [D2, B, D1]
    # hessian = torch.stack(hessian_rows[:D1 * (D2 - 1)])  # [D2 * D1, B, D1]
    hessian = torch.stack(hessian_rows)  # [D2 * D1, B, D1]
    # l = hessian.sum()
    # l.backward()
    return hessian.transpose(0, 1), pred


def functorch_hessian():
    x_ = x.clone().requires_grad_()
    hessian, pred = vmap(jacfwd(jacrev(predict, argnums=0, has_aux=True), argnums=0, has_aux=True), in_dims=0)(x_)  # [B, D2, D1, D1]
    # l = hessian.sum()
    # l.backward()
    return hessian, pred


def vjp_hessian():
    x_ = x.clone().requires_grad_()
    I_N1 = torch.eye(D2).to(device)
    I_N2 = torch.eye(D1).to(device)
    def predict(x):
        out = model(x)
        return out
    def get_jac_hessian(x, v1, v2):
        def get_jac(x):
            (pred, vjpfunc) = vjp(predict, x)
            return vjpfunc(v1)[0], pred
        (jacobian, hessianfunc, pred) = vjp(get_jac, x, has_aux=True)
        hess = hessianfunc(v2)[0]
        return hess, jacobian, pred
    hessian, jacobian, pred= vmap(vmap(vmap(get_jac_hessian, in_dims=(None, None, 0)), in_dims=(None, 0, None)), in_dims=(0, None, None))(x, I_N1, I_N2)
    pred = pred[:, 0, 0, :]
    jacobian = jacobian[:, :, 0, :]
    # l = hessian.sum()
    # l.backward()
    return hessian, jacobian, pred


def validate_result():
    # test functorch result
    ref_jac, ref_pred = reference_jacobian()
    ft_jac, ft_pred = functorch_jacobian()
    vjp_jac, vjp_pred = vjp_jacobian()
    print(f"max jacobian   error: functorch: {(ref_jac - ft_jac).max():.2e}, vjp {(ref_jac - vjp_jac).max():.2e}")
    print(f"max prediction error: functorch: {(ref_pred - ft_pred).max():.2e}, vjp {(ref_pred - vjp_pred).max():.2e}")
    assert torch.allclose(ft_jac, ref_jac, atol=2e-6)
    assert torch.allclose(ft_pred, ref_pred, atol=2e-6)
    assert torch.allclose(vjp_jac, ref_jac, atol=2e-6)
    assert torch.allclose(vjp_pred, ref_pred, atol=2e-6)

    ref_hes, ref_pred = reference_hessian()
    ft_hes, ft_pred = functorch_hessian()
    vjp_hes, vjp_jac, vjp_pred = vjp_hessian()
    ref_hes = ref_hes.view_as(ft_hes)
    print(f"max hessian    error: functorch: {(ref_hes - ft_hes).max():.2e}, vjp {(ft_hes - vjp_hes).max():.2e}")
    assert torch.allclose(ft_hes, ref_hes, atol=2e-6)
    assert torch.allclose(ft_pred, ref_pred, atol=2e-6)
    assert torch.allclose(vjp_hes, ref_hes, atol=2e-6)
    assert torch.allclose(vjp_pred, ref_pred, atol=2e-6)
    assert torch.allclose(vjp_jac, ref_jac, atol=2e-6)


validate_result()


# warm up
for i in range(10):
    reference_jacobian()
    functorch_jacobian()
    reference_hessian()
    functorch_hessian()


def benchmark(func):
    N = 20
    start = time.time()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for i in range(N):
        torch.cuda.nvtx.range_push(func.__name__)
        _ = func()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    time_ms = ((time.time() - start) / N) * 1000
    max_mem_gb = torch.cuda.max_memory_allocated() / float(1 << 30)
    print(f"{func.__name__}: {time_ms:.3f} ms, {max_mem_gb:.4f} GB")

# benchmark jacobian
benchmark(reference_jacobian)
benchmark(functorch_jacobian)
benchmark(vjp_jacobian)

# benchmark hessian
benchmark(reference_hessian)
benchmark(functorch_hessian)
benchmark(vjp_hessian)


# ========== V100-SXM2-16GB ==========

# without backward (1000)
# max jacobian error: 3.818422555923462e-08
# max prediction error: 0.0
# max hessian error: 1.885928213596344e-08
# max prediction error: 0.0
# reference_jacobian: 3.371 ms
# functorch_jacobian: 2.631 ms
# reference_hessian: 12.921 ms
# functorch_hessian: 8.470 ms

# with backward (1000)
# max jacobian error: 3.818422555923462e-08
# max prediction error: 0.0
# max hessian error: 1.885928213596344e-08
# max prediction error: 0.0
# reference_jacobian: 6.786 ms
# functorch_jacobian: 4.891 ms
# reference_hessian: 27.087 ms
# functorch_hessian: 16.028 ms

# without backward (3000)
# max jacobian error: 5.029141902923584e-08
# max prediction error: 0.0
# max hessian error: 2.0489096641540527e-08
# max prediction error: 0.0
# reference_jacobian: 6.475 ms
# functorch_jacobian: 3.066 ms
# reference_hessian: 28.806 ms
# functorch_hessian: 14.185 ms

# with backward (3000)
# max jacobian error: 5.029141902923584e-08
# max prediction error: 0.0
# max hessian error: 2.0489096641540527e-08
# max prediction error: 0.0
# reference_jacobian: 14.804 ms
# functorch_jacobian: 10.111 ms
# reference_hessian: 68.296 ms
# functorch_hessian: 42.193 ms
