import torch
import mbtopk
torch.manual_seed(1)
import torchsnooper
import snoop
torchsnooper.register_snoop(verbose=True)

#@snoop
def test(x, k):
    dim = 1
    largest = False
    sorted = False

    # x = x.abs()
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


x = torch.randn((6, 30000), dtype=torch.float32, device="cuda")
# x = torch.tensor([[4.0158, 3.8557, 3.7402, 3.6092, 3.5917],
#         [5.1249, 3.9560, 3.9426, 3.7941, 3.7187],
#         [4.4902, 4.2852, 3.9487, 3.6333, 3.5653],
#         [4.2879, 3.9424, 3.9098, 3.7816, 3.7129],
#         [3.9525, 3.7470, 3.7286, 3.7038, 3.6325],
#         [4.0618, 3.8468, 3.7639, 3.7621, 3.7464]], dtype=torch.float32, device="cuda")
test(x, k=3)
