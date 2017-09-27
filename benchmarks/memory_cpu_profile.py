"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Create memory and CPU profile plots.
"""
import numpy as np

from mlens.utils.dummy import LayerGenerator
from mlens.utils.utils import CMLog

from time import sleep, perf_counter

from sklearn.datasets import make_friedman1

import matplotlib.pyplot as plt
from seaborn import set_palette



def run():
    """Run profiling."""
    lc = LayerGenerator().get_sequential('stack', False, False)

    cm = CMLog(verbose=False)
    cm.monitor()

    sleep(5)

    t1 = int(np.floor(perf_counter() - cm._t0) * 10)
    sleep(0.1)
    x, z = make_friedman1(int(5 * 1e6))

    sleep(5)

    t2 = int(np.floor(perf_counter() - cm._t0) * 10)
    sleep(0.1)
    lc.fit(x, z)
    t3 = int(np.floor(perf_counter() - cm._t0) * 10)

    sleep(5)

    while not hasattr(cm, 'cpu'):
        cm.collect()
        sleep(1)

    return cm, t1, t2, t3


def plot_rss(cm, t1, t2, t3):
    """Plot the memory profile."""
    f = plt.figure(figsize=(8, 6))
    plt.plot(range(cm.cpu.shape[0]), cm.rss / 1000000)
    plt.axvline(t1 - 3, color='darkcyan', linestyle='--', linewidth=1.0,
                label='load data')
    plt.axvline(t2, color='blue', linestyle='--', linewidth=1.0,
                label='fit start')
    plt.axvline(t3, color='blue', linestyle='-.', linewidth=1.0,
                label='fit end')
    plt.xticks([i for i in [0, 50, 100, 150, 200, 250]],
               [i for i in [0, 5, 10, 15, 20, 25]])
#    plt.ylim(120, 240)
    plt.title("ML-Ensemble memory profile (working set)")
    plt.ylabel("Working set memory (MB)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()

    if PRINT:
        try:
            f.savefig("dev/img/memory_profile.png", dpi=600)
        except:
            f.savefig("memory_profile.png", dpi=600)


def plot_cpu(cm, t1, t2, t3):
    """Plot the CPU profile."""
    f = plt.figure()
    plt.plot(range(cm.cpu.shape[0]), cm.cpu)
    plt.axvline(t1 - 3, color='darkcyan', linestyle='--', linewidth=1.0,
                label='load data')
    plt.axvline(t2, color='blue', linestyle='--', linewidth=1.0,
                label='fit start')
    plt.axvline(t3, color='blue', linestyle='-.', linewidth=1.0,
                label='fit end')
    plt.xticks([i for i in [0, 50, 100, 150, 200, 250]],
               [i for i in [0, 5, 10, 15, 20, 25]])
    plt.title("ML-Ensemble CPU profile")
    plt.ylabel("CPU utilization (%)")
    plt.xlabel("Time (s)")
    plt.legend()

    if PRINT:
        try:
            f.savefig("dev/cpu_profile.png", dpi=600)
        except:
            f.savefig("cpu_profile.png", dpi=600)

if __name__ == '__main__':

    PRINT = False

    set_palette('husl', 100)
    CM, T1, T2, T3 = run()
    plot_rss(CM, T1, T2, T3)
    plot_cpu(CM, T1, T2, T3)

