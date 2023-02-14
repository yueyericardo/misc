import torch
import mbtopk
torch.manual_seed(1)
import torchsnooper
import snoop
torchsnooper.register_snoop(verbose=True)


def test(x, k):
    dim = 1
    sorted = False
    values_ref, indices_ref = torch.topk(x, k, dim, largest, sorted)
    x = x.cuda()
    values, indices = torch.topk(x, k, dim, largest, sorted)
    assert(torch.allclose(values.sort(dim=dim)[0].cpu(), values_ref.sort(dim=dim)[0]))
    assert(torch.allclose(indices.sort(dim=dim)[0].cpu(), indices_ref.sort(dim=dim)[0]))


for largest in [True, False]:
    x = torch.randn((6, 6), dtype=torch.float32)
    test(x, k=3)
    x = torch.randn((6, 8192), dtype=torch.float32)
    test(x, k=3)
    x = torch.randn((6, 30000), dtype=torch.float32)
    test(x, k=3)
    x = torch.randn((200, 500000), dtype=torch.float32)
    test(x, k=2000)
    x = torch.randn((6, 6, 6), dtype=torch.float32)
    test(x, k=3)
