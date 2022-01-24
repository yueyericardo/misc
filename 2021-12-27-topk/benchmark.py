import torch
import time
import torch
import mbtopk
import pandas as pd

dim = 1
largest = True
sorted = False
torch.manual_seed(1)


def benchmark(D1, D2, k):
    x = torch.randn((D1, D2), dtype=torch.float32, device="cuda", requires_grad=True)

    # orig
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(1000):
        values_ref, indices_ref = x.topk(k=k, dim=dim, sorted=sorted, largest=largest)
    torch.cuda.synchronize()
    time_orig = (time.time() - start_time) / 1000
    print(f'{time_orig:.10f}')

    # mb
    values = torch.empty((D1, k), dtype=torch.float32, device="cuda")
    indices = torch.empty((D1, k), dtype=torch.long, device="cuda")
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(1000):
        torch.ops.mbtopk.multiBlockTopK(x, k, dim, largest, sorted, values, indices)
    torch.cuda.synchronize()
    time_mb = (time.time() - start_time) / 1000
    print(f'{time_mb:.10f}')

    # test
    assert(torch.allclose(values.sort()[0], values_ref.sort()[0]))
    assert(torch.allclose(indices.sort()[0], indices_ref.sort()[0]))

    speedup = time_orig / time_mb
    print(f'{speedup:.10f}')

    return (time_orig, time_mb, speedup)


data = []

k = 2000
for d1 in list(range(10, 140, 30))+[600, 660]:
    D2 = [2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17] if d1<600 else [2**11, 2**12]
    for d2 in D2:
        print(f"----------------- D1 = {d1}, D2 = {d2} -----------------")
        time_orig, time_mb, speedup = benchmark(d1, d2, k)
        data.append([d1, d2, k, time_orig, time_mb, speedup])

df = pd.DataFrame(data=data, columns=['D1', 'D2', 'k', 'time_orig', 'time_mb', 'speedup'])
# print(df)
# df.to_csv('benchmark.csv')


D1s = df.D1.unique()
import matplotlib.pyplot as plt
import datetime
import subprocess
import os
plt.figure(figsize=(6, 4), dpi=150)
for D1 in D1s:
    tdf = df[df.D1 == D1]
    plt.plot(tdf.D2, tdf.speedup, label=f'D1={D1}', marker='x')
plt.xlabel('D2')
plt.ylabel('speedup')
plt.ylim(0, 4)
plt.xlim(0, 25000)
plt.legend()
dir = 'benchmark'
if not os.path.exists(dir):
    os.makedirs(dir)
timenow = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"
imgfilename = f"{dir}/{timenow}.png"
plt.savefig(imgfilename, dpi=200, bbox_inches='tight')
cmd = f"if command -v imgcat > /dev/null; then imgcat {imgfilename} --height 30; fi"
subprocess.run(cmd, shell=True, check=True, universal_newlines=True)
print(f"\nBenchmark plot saved at {os.path.realpath(imgfilename)}")
plt.show()
