import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rc
import datetime
import subprocess
from distutils.spawn import find_executable

rc('mathtext', fontset='cm')
usetex = bool(find_executable('latex')) and bool(find_executable('dvipng'))
# usetex = False
rc('axes.spines', **{'right': False, 'top': False})
rc('axes', grid=True)
plt.rcParams['grid.linewidth'] = 0.15
if usetex:
    rc('xtick', labelsize=14)
    rc('ytick', labelsize=14)
    rc('text', usetex=True)
    rc('font', **{'size': 16, 'family': 'serif', 'sans-serif': [
       'Lucid', 'Arial', 'Helvetica'], 'serif': ['Times', 'Times New Roman']})
else:
    rc('font', **{'size': 14})

a = [1, 5, 8, 10]
b = [0.000362774133682251, 0.0015004491806030274,
     0.003101539611816406, 0.0052121269702911375]

b = np.array(b) * 100000
offset = (np.max(b) - np.min(b)) / 20
plt.figure(figsize=(11, 6), dpi=200)

plt.plot(a, b, '--bo', label='pyaev forward')
for i, txt in enumerate(b):
    plt.annotate(f'{txt:.2f}', (a[i], b[i]+offset), ha='center', va='center')
# plt.xlim(-1, 9)
# plt.ylim(-0.5, 4)
plt.legend(frameon=True, loc='upper left')
plt.xlabel(r'Size ($m$)')
plt.ylabel(r'Yaxis ($ms$)')
plt.title(f'Title $\sigma$')
plt.show()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output',
                    default=None,
                    help='output image filename')
args = parser.parse_args()
dir = 'plotimg'
if not os.path.exists(dir):
    os.makedirs(dir)
if args.output is None:
    args.output = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"
imgfilepath = f"{dir}/{args.output}.png"
plt.savefig(imgfilepath, dpi=200, bbox_inches='tight')
cmd = f"if command -v imgcat > /dev/null; then imgcat {imgfilepath} --height 30; fi"
subprocess.run(cmd, shell=True, check=True, universal_newlines=True)
print(f"\nBenchmark plot saved at {os.path.realpath(imgfilepath)}\n")
