"""Monte Carlo study of the report: which pair of star counts gives the most precise
method-of-moments estimates, depending on the density of the two hidden regimes?

R trajectories of T steps are simulated on N vertices; on each, the six parameters
(p0, q0, alpha, q1, beta, q2) are estimated with each pair of stars in COMBOS, on the same data.
The table reports the standard deviation of the estimates across trajectories.

Scenarios (report, Table 2): on-times W(1.5, alpha) and W(1.5, beta), regime sojourns G(0.3) and
G(0.6); the off-time parameters set the density.
    A  asymmetric      q1 = 0.4, q2 = 0.8   (graph 1 sparse, graph 2 dense)
    B  sparse          q1 = 0.1, q2 = 0.15
    C  dense           q1 = 0.9, q2 = 0.95

    python experiment.py --scenario A                 # N = 15, T = 100000, R = 200
    python experiment.py --scenario A --quick         # T = 5000, R = 20, about a minute
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dyngraph import estimate, moments, seed_numba, simulate  # noqa: E402

PARAMS = ['p0', 'q0', 'alpha', 'q1', 'beta', 'q2']
COMBOS = [(3, 4), (3, 5), (4, 5), (3, 6), (4, 6)]
LAM = 1.5
SCENARIOS = {'A': (0.4, 0.8), 'B': (0.1, 0.15), 'C': (0.9, 0.95)}
LABELS = {'p0': r'$\hat{p}_0$', 'q0': r'$\hat{q}_0$', 'alpha': r'$\hat{\alpha}$',
          'q1': r'$\hat{q}_1$', 'beta': r'$\hat{\beta}$', 'q2': r'$\hat{q}_2$'}


def setup(scenario):
    q1, q2 = SCENARIOS[scenario]
    true = {'p0': 0.3, 'q0': 0.6, 'alpha': 0.5, 'q1': q1, 'beta': 0.3, 'q2': q2}
    X1 = {'type': 'weibull', 'lam': LAM, 'alpha': true['alpha']}
    Y1 = {'type': 'geometric', 'p': q1}
    X2 = {'type': 'weibull', 'lam': LAM, 'alpha': true['beta']}
    Y2 = {'type': 'geometric', 'p': q2}
    return true, X1, Y1, X2, Y2


def run(N, T, R, scenario, seed=42):
    true, X1, Y1, X2, Y2 = setup(scenario)
    np.random.seed(seed)
    seed_numba(seed)
    all_stars = (3, 4, 5, 6)
    print(f"Scenario {scenario}: N={N}, T={T}, R={R}, seed={seed}")
    print(f"Combos: {['S%d+S%d' % c for c in COMBOS]}\n")

    # phase 1: simulate R trajectories and compute all moments
    all_mom = []
    for r in range(R):
        t0 = time.time()
        data = simulate(N, T, X1, Y1, X2, Y2, true['p0'], true['q0'], all_stars)
        all_mom.append(moments(data, all_stars))
        if (r + 1) % max(1, R // 5) == 0:
            print(f"  Sim {r+1}/{R}  ({time.time()-t0:.1f}s)")

    # phase 2: estimate with each pair of stars on the same data
    results, n_ok = {}, {}
    for la, lb in COMBOS:
        label = f"S{la}+S{lb}"
        ests = []
        for m in all_mom:
            e = estimate(N, m, la, lb, LAM, LAM)
            if e and all(np.isfinite(e.get(k, np.nan)) for k in PARAMS):
                ests.append(e)
        print(f"  {label}: {len(ests)}/{R} ok")
        n_ok[label] = len(ests)
        if not ests:
            results[label] = None
            continue
        res = {}
        for p in PARAMS:
            v = np.array([e[p] for e in ests])
            res[p] = {'mean': float(np.mean(v)), 'std': float(np.std(v)),
                      'bias': float(np.mean(v) - true[p]), 'values': v.tolist()}
        results[label] = res

    # phase 3: table of standard deviations
    labels = [f"S{a}+S{b}" for a, b in COMBOS]
    print(f"\n{'Param':<8} {'True':<7} " + " ".join(f"{lab:<10}" for lab in labels) + "  Best")
    print("─" * 75)
    for p in PARAMS:
        row = f"{p:<8} {true[p]:<7.4f} "
        stds = []
        for lab in labels:
            if results[lab] and p in results[lab]:
                s = results[lab][p]['std']
                row += f"{s:<10.4f} "
                stds.append(s)
            else:
                row += f"{'N/A':<10} "
                stds.append(np.inf)
        row += f"  {labels[int(np.argmin(stds))]}"
        print(row)

    print("\nMean counts:")
    for s in all_stars:
        print(f"  S{s}: {np.mean([m[f'S{s}_0'] for m in all_mom]):.1f}")
    print(f"  Edges: {np.mean([m['A0'] for m in all_mom]):.1f}")
    return results, true, n_ok


def save(fig, name):
    fig.savefig(f"{name}.pdf", bbox_inches='tight')
    fig.savefig(f"{name}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {name}.pdf/.png")


def plot_comparison(results, true, name):
    """Standard deviation of each estimate under each pair of stars (best pair outlined)."""
    labels = [lab for lab in results if results[lab]]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    for idx, p in enumerate(PARAMS):
        ax = axes[idx // 3, idx % 3]
        stds = [results[lab][p]['std'] if p in results[lab] else 0 for lab in labels]
        bars = ax.bar(range(len(labels)), stds, color=colors, edgecolor='gray', linewidth=0.5)
        best = int(np.argmin(stds))
        bars[best].set_edgecolor('black')
        bars[best].set_linewidth(2)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, fontsize=9)
        ax.set_ylabel('Std', fontsize=10)
        ax.set_title(f'{LABELS[p]}  (true = {true[p]})', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Standard Deviation by Subgraph Combination', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save(fig, name)


def plot_histograms(results, true, name, combo="S3+S4"):
    """Distribution of the estimates across trajectories for one pair of stars."""
    if combo not in results or results[combo] is None:
        return
    r = results[combo]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    colors = ['#D05050', '#4080C0'] * 3
    for idx, p in enumerate(PARAMS):
        ax = axes[idx // 3, idx % 3]
        vals = np.array(r[p]['values'])
        ax.hist(vals, bins=min(35, max(15, len(vals) // 20)),
                color=colors[idx], alpha=0.85, edgecolor='white', linewidth=0.5)
        ax.axvline(true[p], color='black', linestyle='--', linewidth=1.5, label=f'True = {true[p]}')
        ax.set_title(f'{LABELS[p]} = {r[p]["mean"]:.4f} ({r[p]["std"]:.4f})', fontsize=11)
        ax.legend(fontsize=8)
    fig.suptitle(f'Parameter Estimates ({combo})', fontsize=13, fontweight='bold')
    fig.tight_layout()
    save(fig, name)


def plot_decay(name):
    """rho^(l-1), the weight of a regime in the mean l-star count, for a sparse and a dense graph."""
    ells = np.arange(3, 8)
    r1, r2 = 0.2, 0.6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.semilogy(ells, r1 ** (ells - 1), 'o-', color='#D05050',
                 label=r'$\varrho_1^{\ell-1}$ (sparse graph)', linewidth=2)
    ax1.semilogy(ells, r2 ** (ells - 1), 's-', color='#4080C0',
                 label=r'$\varrho_2^{\ell-1}$ (dense graph)', linewidth=2)
    ax1.set_xlabel(r'Star size $\ell$', fontsize=12)
    ax1.set_ylabel(r'$\varrho_i^{\ell-1}$', fontsize=12)
    ax1.set_title('Contribution to snapshot moment', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.set_xticks(ells)
    ax1.grid(alpha=0.3)
    ax2.plot(ells, (r2 / r1) ** (ells - 1), 'D-', color='#2E8B57', linewidth=2)
    ax2.set_xlabel(r'Star size $\ell$', fontsize=12)
    ax2.set_ylabel(r'$(\varrho_2/\varrho_1)^{\ell-1}$', fontsize=12)
    ax2.set_title('Ratio of contributions (dense / sparse)', fontsize=12)
    ax2.set_xticks(ells)
    ax2.grid(alpha=0.3)
    fig.suptitle('Decay of information content for higher-order stars', fontsize=13, fontweight='bold')
    fig.tight_layout()
    save(fig, name)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--scenario', choices=sorted(SCENARIOS), default='A')
    ap.add_argument('--N', type=int, default=15)
    ap.add_argument('--T', type=int, default=100000)
    ap.add_argument('--R', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--quick', action='store_true', help='T = 5000, R = 20')
    args = ap.parse_args()
    if args.quick:
        args.T, args.R = 5000, 20
        print("*** QUICK MODE ***\n")

    results, true, n_ok = run(args.N, args.T, args.R, args.scenario, args.seed)

    tag = f"{args.scenario}_N{args.N}_T{args.T}_R{args.R}"
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    out = {lab: {k: {kk: vv for kk, vv in v.items() if kk != 'values'} for k, v in r.items()}
           for lab, r in results.items() if r}
    with open(f'results/results_{tag}.json', 'w') as f:
        # 'converged': trajectories on which the estimator found a root; the standard
        # deviations of a pair of stars are computed on those only
        json.dump({'scenario': args.scenario, 'true': true, 'N': args.N, 'T': args.T, 'R': args.R,
                   'seed': args.seed, 'converged': n_ok, 'std_by_combo': out}, f, indent=2)
    plot_comparison(results, true, f"figures/comparison_{tag}")
    plot_histograms(results, true, f"figures/histograms_{tag}")
    plot_decay("figures/decay")
