import torch
import mbtopk

import torchsnooper
import snoop
torchsnooper.register_snoop(verbose=True)

# @snoop
def test(x, k):
    dim = 1
    largest = True
    sorted = False

    values, indices = torch.topk(x, k, dim, largest, sorted)
    values_ref = values.clone()
    indices_ref = indices.clone()
    values = values.zero_()
    indices = indices.zero_()

    print('-' * 50)
    torch.ops.mbtopk.multiBlockTopK(x, k, dim, largest, sorted, values, indices)
    print(values_ref)
    print(indices_ref)
    print(values)
    print(indices)
    assert(torch.allclose(values, values_ref))
    assert(torch.allclose(indices, indices_ref))


x = torch.randn((2, 5), dtype=torch.float32, device="cuda")
test(x, k=3)
