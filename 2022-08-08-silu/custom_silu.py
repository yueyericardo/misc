import functools
import torch
from torch.autograd import Function
from torch._C._nvfuser import Fusion, FusionDefinition, DataType

def silu1(fd, x):
    one = fd.define_constant(1.0)
    # y = sigmoid(x)
    y = fd.ops.sigmoid(x)
    # z = sigmoid(x)
    return fd.ops.mul(y, fd.ops.add(one, fd.ops.mul(x, fd.ops.sub(one, y))))

def silu2(fd, x):
    one = fd.define_constant(1.0)

    # y = sigmoid(x)
    y = fd.ops.sigmoid(x)
    # dy = y * (1 - y)
    dy = fd.ops.mul(y, fd.ops.sub(one, y))
    # z = 1 + x * (1 - y)
    z = fd.ops.add(one, fd.ops.mul(x, fd.ops.sub(one, y)))
    # term1 = dy * z
    term1 = fd.ops.mul(dy, z)

    # term2 = y * ((1 - y) - x * dy)
    term2 = fd.ops.mul(y, fd.ops.sub(fd.ops.sub(one, y), fd.ops.mul(x, dy)))

    return fd.ops.add(term1, term2)

def silu3(fd, x):
    one = fd.define_constant(1.0)
    two = fd.define_constant(2.0)

    # y = sigmoid(x)
    y = fd.ops.sigmoid(x)
    # dy = y * (1 - y)
    dy = fd.ops.mul(y, fd.ops.sub(one, y))
    # ddy = (1 - 2y) * dy
    ddy = fd.ops.mul(fd.ops.sub(one, fd.ops.mul(two, y)), dy)
    # term1 = ddy * (2 + x - 2xy)
    term1 = fd.ops.mul(ddy, fd.ops.sub(fd.ops.add(two, x), fd.ops.mul(two, fd.ops.mul(x, y))))

    # term2 = dy * (1 - 2 (y + x * dy))
    term2 = fd.ops.mul(dy, fd.ops.sub(one, fd.ops.mul(two, fd.ops.add(y, fd.ops.mul(x, dy)))))

    return fd.ops.add(term1, term2)

def silu4(x):
    y = torch.sigmoid(x)
    dy = y * (1 - y)
    ddy = (1 - 2 * y) * dy
    dddy = (1- 2 * y) * ddy - 2 * dy * dy
    z = 1 - 2 * (y + x * dy)
    term1 = dddy * (2 + x - 2 * x * y)
    term2 = 2 * ddy * z
    term3 = dy * (-2) * (2 * dy + x * ddy)
    return term1 + term2 + term3

@functools.cache
def silu_backward_for(dtype, dim):
    if dtype == torch.half:
        dtype = DataType.Half
    elif dtype == torch.float:
        dtype = DataType.Float
    elif dtype == torch.double:
        dtype = DataType.Double
    else:
        raise TypeError("Unsupported dtype")

    fusion = Fusion()

    with FusionDefinition(fusion) as fd :
        x = fd.define_tensor(dim, dtype)
        grad_input = silu1(fd, x)
        grad_input = fd.ops.cast(grad_input, dtype)
        fd.add_output(grad_input)

    return fusion

@functools.cache
def silu_double_backward_for(dtype, dim):
    if dtype == torch.half:
        dtype = DataType.Half
    elif dtype == torch.float:
        dtype = DataType.Float
    elif dtype == torch.double:
        dtype = DataType.Double
    else:
        raise TypeError("Unsupported dtype")

    fusion = Fusion()

    with FusionDefinition(fusion) as fd :
        x = fd.define_tensor(dim, dtype)
        grad_input = silu2(fd, x)
        grad_input = fd.ops.cast(grad_input, dtype)
        fd.add_output(grad_input)

    return fusion

@functools.cache
def silu_triple_backward_for(dtype, dim):
    if dtype == torch.half:
        dtype = DataType.Half
    elif dtype == torch.float:
        dtype = DataType.Float
    elif dtype == torch.double:
        dtype = DataType.Double
    else:
        raise TypeError("Unsupported dtype")

    fusion = Fusion()

    with FusionDefinition(fusion) as fd :
        x = fd.define_tensor(dim, dtype)
        grad_input = silu3(fd, x)
        grad_input = fd.ops.cast(grad_input, dtype)
        fd.add_output(grad_input)

    return fusion


class MySiLU(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.nn.functional.silu(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return MySiLU_1.apply(x) * grad_output


class MySiLU_1(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        silu_backward = silu_backward_for(x.dtype, x.dim())
        return silu_backward.execute([x])[0]

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return MySiLU_2.apply(x) * grad_output


class MySiLU_2(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        silu_double_backward = silu_double_backward_for(x.dtype, x.dim())
        return silu_double_backward.execute([x])[0]

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return MySiLU_3.apply(x) * grad_output


class MySiLU_3(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        silu_triple_backward = silu_triple_backward_for(x.dtype, x.dim())
        return silu_triple_backward.execute([x])[0]

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return silu4(x) * grad_output


if __name__ == "__main__":
    input = torch.randn(20, 20, dtype=torch.double, requires_grad=True, device="cuda")
    test = torch.autograd.gradcheck(MySiLU.apply, input, eps=1e-6, atol=1e-4)
    print(test)

    test = torch.autograd.gradcheck(MySiLU_1.apply, input, eps=1e-6, atol=1e-4)
    print(test)

    test = torch.autograd.gradcheck(MySiLU_2.apply, input, eps=1e-6, atol=1e-4)
    print(test)

    test = torch.autograd.gradcheck(MySiLU_3.apply, input, eps=1e-6, atol=1e-4)
    print(test)
