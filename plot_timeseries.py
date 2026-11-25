"""Figure 1 of the report: edge count and 4-star count over time, with the periods spent in the
dense regime shaded (scenario A, N = 15; 200 steps for readability, the report shows 800).

    python plot_timeseries.py
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dyngraph import seed_numba, simulate  # noqa: E402
from experiment import save, setup  # noqa: E402


def shade(ax, modes):
    """Shade the time intervals in which the mode is 2 (the dense graph)."""
    on = np.flatnonzero(np.diff(np.r_[0, (modes == 2).astype(int), 0]))
    for a, b in zip(on[::2], on[1::2]):
        ax.axvspan(a - 0.5, b - 0.5, color='#4080C0', alpha=0.25, lw=0)


if __name__ == '__main__':
    true, X1, Y1, X2, Y2 = setup('A')
    seed_numba(1)
    d = simulate(15, 200, X1, Y1, X2, Y2, true['p0'], true['q0'], (3, 4))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
    for ax, y, lab in ((ax1, d['edges'], r'Edge count $A_N(k)$'), (ax2, d['S4'], r'$S_4$ count')):
        shade(ax, d['modes'])
        ax.step(np.arange(len(y)), y, where='mid', color='#222222', lw=0.9, label=lab)
        ax.fill_between([], [], color='#4080C0', alpha=0.25, label='Regime 2 (dense)')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)
    ax1.set_ylabel('Number of edges')
    ax2.set_ylabel(r'Number of $S_4$')
    ax2.set_xlabel('Time step $k$')
    fig.suptitle('Network density and motif amplification during regime shifts', fontsize=13)
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    save(fig, 'figures/timeseries')
