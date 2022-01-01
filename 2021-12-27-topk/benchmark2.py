import torch
import time
import torch
import mbtopk
torch.manual_seed(1)

k=2000
dim = 1
largest = True
sorted = False
N = 16
M = 10000
def BS12_sorted_orig():
    x = torch.randn((N,M), dtype=torch.float32, device="cuda", requires_grad=True)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(1000):
        y,z = x.topk(k=k, dim=dim, sorted=sorted, largest=largest)
    torch.cuda.synchronize()
    time_for_topk = (time.time() - start_time) / 1000
    print(time_for_topk)

def BS12_sorted_mb():
    x = torch.randn((N,M), dtype=torch.float32, device="cuda", requires_grad=True)
    values = torch.empty((N, k), dtype=torch.float32, device="cuda")
    indices = torch.empty((N, k), dtype=torch.long, device="cuda")
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(1000):
        torch.ops.mbtopk.multiBlockTopK(x, k, dim, largest, sorted, values, indices)
    torch.cuda.synchronize()
    time_for_topk = (time.time() - start_time) / 1000
    print(time_for_topk)


for n in range(5, 60, 5):
    for m in [2**11, 2**14, 2**18, 2**20]:
        print(f"----------------- N = {n}, M = {m} -----------------")
        N = n
        M = m
        BS12_sorted_orig()
        BS12_sorted_mb()
