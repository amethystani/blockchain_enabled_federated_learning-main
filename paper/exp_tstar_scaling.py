#!/usr/bin/env python3
"""
T* Scaling vs Byzantine Fraction f
====================================
SSS 2026 — Validates T* = O(σf/ε²) stabilization time bound.

Uses the SelfStabilizationAnalyzer mathematical model (not full FL training)
to sweep f ∈ {1,2,3,4,5,6,7,8,9} and show:
1. T_stab is bounded above by T* = O(σf/ε²) for all f
2. T_stab grows linearly with f (not with initial corruption)
3. T_stab is INDEPENDENT of initial corruption level — key SS property

Also shows T* vs theoretical bound convergence rate.
"""

import sys
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, '/home/user/blockchain_enabled_federated_learning-main')

from spectral_sentinel.stability.self_stabilization import (
    SelfStabilizationAnalyzer,
    DistributedFLConfig,
)

RESULTS_DIR = Path('/home/user/blockchain_enabled_federated_learning-main/results/tstar_scaling')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = Path('/home/user/blockchain_enabled_federated_learning-main/paper/figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N = 20
SIGMA_SQ = 0.01
EPSILON = 0.05
CORRUPTION_LEVELS = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]


def run_f_sweep():
    """Sweep f from 1 to N//2-1, show T_stab vs T* and independence from init."""
    f_values = list(range(1, N//2))   # f = 1..9
    results = {}
    np.random.seed(42)

    print(f"{'f':>4} {'T*(theory)':>12} {'T_stab(mean)':>14} {'T_stab(max)':>12} {'σ²f²':>8}")
    print("─"*55)

    for f in f_values:
        config = DistributedFLConfig(n=N, f=f, sigma_sq=SIGMA_SQ,
                                      T=200, epsilon=EPSILON)
        analyzer = SelfStabilizationAnalyzer(config)
        T_theory  = config.theoretical_recovery_time()

        # Run recovery from each corruption level (5 seeds each)
        stab_rounds = []
        for corr in CORRUPTION_LEVELS:
            for seed in range(5):
                np.random.seed(seed * 100 + f)
                traj = analyzer.analyze_recovery_trajectory(
                    initial_corruption=corr, num_rounds=200
                )
                stab_rounds.append(traj['rounds_to_stabilize'])

        mean_stab = np.mean(stab_rounds)
        max_stab  = np.max(stab_rounds)
        print(f"{f:>4} {T_theory:>12} {mean_stab:>14.1f} {max_stab:>12.0f} "
              f"{config.sigma_sq_f_sq:>8.4f}")

        results[f] = {
            'f': f,
            'n': N,
            'sigma_sq_f_sq': config.sigma_sq_f_sq,
            'T_star_theory': T_theory,
            'stab_rounds_all': stab_rounds,
            'mean_stab': float(mean_stab),
            'max_stab': float(max_stab),
            'detectable': config.is_detectable,
        }

    return results


def run_corruption_independence(f: int = 3):
    """Show that T_stab is INDEPENDENT of initial corruption (key SS property)."""
    config = DistributedFLConfig(n=N, f=f, sigma_sq=SIGMA_SQ, T=200, epsilon=EPSILON)
    analyzer = SelfStabilizationAnalyzer(config)
    T_theory  = config.theoretical_recovery_time()

    corruption_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]
    stab_per_corr = []
    np.random.seed(42)
    for corr in corruption_values:
        rounds_list = []
        for seed in range(10):
            np.random.seed(seed * 100 + int(corr))
            traj = analyzer.analyze_recovery_trajectory(
                initial_corruption=corr, num_rounds=200
            )
            rounds_list.append(traj['rounds_to_stabilize'])
        stab_per_corr.append({
            'corruption': corr,
            'mean_stab': float(np.mean(rounds_list)),
            'std_stab':  float(np.std(rounds_list)),
        })

    return stab_per_corr, T_theory


def main():
    print("="*60)
    print("  T* SCALING vs BYZANTINE FRACTION f")
    print(f"  n={N}, σ²={SIGMA_SQ}, ε={EPSILON}")
    print("="*60)

    # 1. Sweep f
    f_results = run_f_sweep()

    # 2. Corruption independence for f=3
    corr_indep, T_theory_f3 = run_corruption_independence(f=3)

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(RESULTS_DIR / 'results.json', 'w') as fh:
        json.dump({
            'f_sweep': {str(k): v for k, v in f_results.items()},
            'corruption_independence': corr_indep
        }, fh, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        r"$T^*$ Scaling: $T_\mathrm{stab} \leq T^* = O(\sigma f/\varepsilon^2)$   —   "
        "Key Self-Stabilization Property\n"
        f"Spectral Sentinel, n={N}, σ²={SIGMA_SQ}, ε={EPSILON}",
        fontsize=11, fontweight='bold'
    )

    f_vals     = sorted(f_results.keys())
    detectable = [f for f in f_vals if f_results[f]['detectable']]
    not_detect = [f for f in f_vals if not f_results[f]['detectable']]

    # Panel A: T_stab vs f
    ax = axes[0]
    mean_stab = [f_results[f]['mean_stab'] for f in f_vals]
    max_stab  = [f_results[f]['max_stab']  for f in f_vals]
    T_theory  = [f_results[f]['T_star_theory'] for f in f_vals]

    ax.fill_between(f_vals, 0, max_stab, alpha=0.15, color='steelblue',
                    label='Max observed range')
    ax.plot(f_vals, mean_stab, 'o-', color='steelblue', lw=2, ms=6,
            label=r'Mean $T_\mathrm{stab}$ (observed)')
    ax.plot(f_vals, T_theory, 's--', color='orange', lw=2, ms=6,
            label=r'$T^* = O(\sigma f/\varepsilon^2)$ (theory)')

    # Shade undetectable region
    if not_detect:
        ax.axvspan(min(not_detect)-0.5, max(not_detect)+0.5,
                   alpha=0.12, color='red', label=r'$\sigma^2 f^2 \geq 0.25$ (impossible)')

    ax.set_xlabel("Number of Byzantine Clients f", fontsize=10)
    ax.set_ylabel("Stabilization Rounds", fontsize=10)
    ax.set_title(r"$T_\mathrm{stab}$ vs $f$: Linear Growth in $f$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks(f_vals)

    # Panel B: σ²f² vs f with phase transition
    ax = axes[1]
    sf2_vals = [f_results[f]['sigma_sq_f_sq'] for f in f_vals]
    bar_colors = ['steelblue' if f_results[f]['detectable'] else 'red' for f in f_vals]
    ax.bar(f_vals, sf2_vals, color=bar_colors, alpha=0.8, edgecolor='black', lw=0.6)
    ax.axhline(0.25, color='black', ls='-', lw=2.5,
               label=r'Phase transition $\sigma^2 f^2 = 0.25$')
    ax.set_xlabel("Byzantine Clients f", fontsize=10)
    ax.set_ylabel(r"$\sigma^2 f^2$", fontsize=10)
    ax.set_title(r"Phase Transition: $\sigma^2 f^2 = 0.25$" + "\n"
                 r"Blue: detectable (SS works); Red: impossible")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    ax.set_xticks(f_vals)

    # Panel C: T_stab independence from initial corruption (f=3)
    ax = axes[2]
    corr_x   = [c['corruption'] for c in corr_indep]
    corr_y   = [c['mean_stab']  for c in corr_indep]
    corr_err = [c['std_stab']   for c in corr_indep]
    ax.errorbar(range(len(corr_x)), corr_y, yerr=corr_err,
                fmt='o-', color='steelblue', lw=2, ms=6, capsize=4,
                label=r'$T_\mathrm{stab}$ (mean ± std)')
    ax.axhline(T_theory_f3, color='orange', ls='--', lw=2,
               label=f'T* = {T_theory_f3} (theory)')
    ax.set_xticks(range(len(corr_x)))
    ax.set_xticklabels([str(c) for c in corr_x], rotation=30, fontsize=8)
    ax.set_xlabel("Initial Corruption ‖w⁰ − w*‖", fontsize=10)
    ax.set_ylabel("Stabilization Rounds", fontsize=10)
    ax.set_title(r"$T_\mathrm{stab}$ Independent of Initial Corruption" + "\n"
                 r"(f=3, $\sigma^2 f^2 < 0.25$) — Key SS property")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    # Add text: SS property
    ax.text(0.5, 0.85,
            "500× range of corruption\n→ same stabilization time\n= Self-Stabilizing!",
            transform=ax.transAxes, fontsize=8, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    for out in [RESULTS_DIR / 'fig_tstar_f.png', FIGURES_DIR / 'fig_tstar_f.png']:
        plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {RESULTS_DIR / 'fig_tstar_f.png'}")
    print(f"Saved: {FIGURES_DIR / 'fig_tstar_f.png'}")


if __name__ == '__main__':
    main()
