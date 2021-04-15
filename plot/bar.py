import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rc
import datetime
import subprocess
from distutils.spawn import find_executable


def autolabel(rects, text='cuaev'):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{}\n{:.2f}'.format(text, height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     fontsize=13,
                     ha='center', va='bottom')


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

b = [0.000362774133682251, 0.0015004491806030274,
     0.003101539611816406, 0.0052121269702911375]
labels = ['step 1 ', 'step 2', 'step 3', 'step 4']
a = np.arange(len(labels))

b = np.array(b) * 100000
plt.figure(figsize=(11, 6), dpi=200)
plt.xticks(a, labels)
barwidth = 0.5
abar = plt.bar(a, b, width=barwidth)
# plt.xlim(-1, 9)
# plt.ylim(-0.5, 4)
# plt.legend(frameon=True, loc='upper left')
autolabel(abar, text='')
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
