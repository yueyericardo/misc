import torch
import mbtopk
torch.manual_seed(1)
import torchsnooper
import snoop
torchsnooper.register_snoop(verbose=True)

#@snoop
def test(x, k):
    dim = 1
    largest = True
    sorted = False

    values, indices = torch.topk(x, k, dim, largest, sorted)
    values_ref = values.clone()
    indices_ref = indices.clone()
    values = values.zero_()
    indices = indices.zero_()
    print(x)
    a = torch.sort(x, descending=True, dim=1)[0][:, :5]
    print('sorted')
    print(a)
    print('-' * 50)
    print(values_ref)
    print(indices_ref)
    # torch.ops.mbtopk.multiBlockTopK(x, k, dim, largest, sorted, values, indices)
    torch.ops.mbtopk.multiBlockTopK(x, k, dim, largest, sorted, values, indices)
    print('-' * 50)
    print(values)
    print(indices)
    assert(torch.allclose(values.sort()[0], values_ref.sort()[0]))
    assert(torch.allclose(indices.sort()[0], indices_ref.sort()[0]))


x = torch.randn((6, 6), dtype=torch.float32, device="cuda")
test(x, k=3)
x = torch.randn((6, 8192), dtype=torch.float32, device="cuda")
test(x, k=3)
x = torch.randn((6, 30000), dtype=torch.float32, device="cuda")
test(x, k=3)
