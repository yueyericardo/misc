import torch
import time
import torch
import mbtopk
torch.manual_seed(1)

k=2000
dim = 1
largest = True
sorted = False

def BS12_sorted_orig():
    x = torch.randn((60,201600), dtype=torch.float32, device="cuda", requires_grad=True)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(1000):
        y,z = x.topk(k=k, dim=dim, sorted=sorted, largest=largest)
    torch.cuda.synchronize()
    time_for_topk = (time.time() - start_time) / 1000
    print(time_for_topk)

def BS12_sorted_mb():
    x = torch.randn((60,201600), dtype=torch.float32, device="cuda", requires_grad=True)
    values = torch.empty((60, 2000), dtype=torch.float32, device="cuda")
    indices = torch.empty((60, 2000), dtype=torch.long, device="cuda")
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(1000):
        torch.ops.mbtopk.multiBlockTopK(x, k, dim, largest, sorted, values, indices)
    torch.cuda.synchronize()
    time_for_topk = (time.time() - start_time) / 1000
    print(time_for_topk)


BS12_sorted_orig()
BS12_sorted_mb()
