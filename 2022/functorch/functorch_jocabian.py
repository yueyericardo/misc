import time
from functorch import vmap, jacrev, jacfwd, vjp
import torch
import torch.nn as nn
import torchsnooper
import snoop
torchsnooper.register_snoop(verbose=True)
_ = torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
D1 = 2  # x, y
D2 = 3  # u, v, p
B = 10000
x = torch.randn(B, D1).to(device)
I_N = torch.eye(D2).to(device)

model = nn.Sequential(
    nn.Linear(D1, 512), nn.Tanh(),
    nn.Linear(512, 512), nn.Tanh(),
    nn.Linear(512, 512), nn.Tanh(),
    nn.Linear(512, 512), nn.Tanh(),
    nn.Linear(512, 512), nn.Tanh(),
    nn.Linear(512, 512), nn.Tanh(),
    nn.Linear(512, D2),
).to(device)


# def predict(x):
#     torch.cuda.nvtx.range_push("forward")
#     out = model(x)
#     torch.cuda.nvtx.range_pop()
#     return out, out  # return two outputs is needed for jacrev auxiliary object


def predict(x):
    torch.cuda.nvtx.range_push("forward")
    out = model(x)
    torch.cuda.nvtx.range_pop()
    return out

def reference_jacobian():
    x_ = x.clone().requires_grad_()
    ones = torch.ones(B, device=x.device)
    pred = predict(x_)
    jacobian_rows = [None] * D2
    for i in range(D2):
        # if i == 1:
            # continue
        torch.cuda.nvtx.range_push("autograd")
        jacobian_rows[i] = torch.autograd.grad(pred[:, i], x_, ones, create_graph=True)[0]
        torch.cuda.nvtx.range_pop()
    jacobian_rows = [j for j in jacobian_rows if j is not None]
    jacobian = torch.stack(jacobian_rows)  # [D2, B, D1]
    return jacobian, pred


# def functorch_jacobian():
#     x_ = x.requires_grad_()
#     jacobian, pred = vmap(jacrev(predict, argnums=0, has_aux=True), in_dims=0)(x_)  # [B, D2, D1]
#     return jacobian, pred

def functorch_jacobian():
    def get_vjp(x, v):
        (pred, vjpfunc) = vjp(predict, x)
        vjp()
        return vjpfunc(v)[0], pred

    x_ = x.clone().requires_grad_()
    jacobian, pred = vmap(vmap(get_vjp, in_dims=(None, 0)), in_dims=(0, None))(x_, I_N)
    # TODO forward has redundent works
    pred = pred[:, 0]
    return jacobian, pred


def functorch_hessian():
    def get_vjp(x, v):
        (pred, vjpfunc) = vjp(predict, x)
        return vjpfunc(v)[0], pred

    x_ = x.clone().requires_grad_()
    jacobian, pred = vmap(vmap(get_vjp, in_dims=(None, 0)), in_dims=(0, None))(x_, I_N)
    # TODO
    pred = pred[:, 0]
    return jacobian, pred


# test functorch result
ref_jac, ref_pred = reference_jacobian()
ft_jac, ft_pred = functorch_jacobian()
print(f"max jacobian error: {(ref_jac.transpose(0, 1) - ft_jac).max()}")
print(f"max prediction error: {(ref_pred - ft_pred).max()}")
assert torch.allclose(ft_jac, ref_jac.transpose(0, 1), atol=5e-5)
assert torch.allclose(ft_pred, ref_pred, atol=2e-7)

# benchmark reference
N = 100
start = time.time()
torch.cuda.synchronize()
for i in range(N):
    torch.cuda.nvtx.range_push(f"reference_jacobian")
    ref_jac, ref_pred = reference_jacobian()
    torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
time_ms = ((time.time() - start) / N) * 1000
print(f'reference_jacobian: {time_ms:.3f} ms')


# benchmark functorch
N = 100
start = time.time()
torch.cuda.synchronize()
for i in range(N):
    torch.cuda.nvtx.range_push(f"functorch_jacobian")
    ft_jac, ft_pred = functorch_jacobian()
    torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
time_ms = ((time.time() - start) / N) * 1000
print(f'functorch_jacobian: {time_ms:.3f} ms')

# Sample output from V100-SXM2-16GB
# max jacobian error: 5.029141902923584e-08
# max prediction error: 0.0
# reference_jacobian: 6.153 ms
# functorch_jacobian: 3.086 ms
