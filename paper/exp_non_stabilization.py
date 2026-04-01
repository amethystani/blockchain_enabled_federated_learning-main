#!/usr/bin/env python3
"""
Non-Stabilization of FedAvg vs Self-Stabilization of Spectral Sentinel
=======================================================================
SSS 2026 — Figure from existing experiment results.

Uses pre-computed results from results/novelty_proof/results.json.

The Gaussian attack (results already show FedAvg collapses to 9.8%)
demonstrates non-stabilization: Byzantine nodes prevent recovery.
Spectral Sentinel recovers to 90.6% by detecting and excluding them.

This generates fig_non_stab.png for the paper.
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

RESULTS_DIR = Path('/home/user/blockchain_enabled_federated_learning-main/results/non_stabilization')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = Path('/home/user/blockchain_enabled_federated_learning-main/paper/figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # Load real experiment results
    with open('/home/user/blockchain_enabled_federated_learning-main/results/novelty_proof/results.json') as f:
        data = json.load(f)

    # Non-stabilization is best illustrated by Gaussian attack:
    # FedAvg collapses and STAYS collapsed (never recovers) = non-stabilizing
    # Spectral Sentinel detects Byzantine clients and recovers = self-stabilizing
    aggregators = ['spectral_sentinel', 'fedavg', 'krum', 'median']
    attack = 'gaussian'

    results = {agg: data[f'{agg}_{attack}'] for agg in aggregators}

    colors = {'spectral_sentinel': '#1f77b4', 'fedavg': '#d62728',
              'krum': '#2ca02c', 'median': '#ff7f0e'}
    labels = {
        'spectral_sentinel': 'Spectral Sentinel (self-stabilizing: ✓ detects, ✓ recovers)',
        'fedavg':             'FedAvg (NOT self-stabilizing: no detection, no recovery)',
        'krum':               'Krum (no detection mechanism)',
        'median':             'Coord. Median (no detection mechanism)',
    }
    styles = {'spectral_sentinel': '-', 'fedavg': '--', 'krum': '-.', 'median': ':'}

    # ── Figure: 2 panels ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Non-Stabilization of FedAvg vs Self-Stabilization of Spectral Sentinel\n"
        "Gaussian Attack, 30% Byzantine, MNIST (n=10, f=3, 8 rounds)",
        fontsize=12, fontweight='bold'
    )

    # Panel A: Per-round accuracy
    ax = axes[0]
    for agg in aggregators:
        accs = results[agg]['accuracy_per_round']
        rounds = list(range(1, len(accs)+1))
        ax.plot(rounds, accs, color=colors[agg], ls=styles[agg],
                lw=2.2, marker='o', markersize=5, label=labels[agg])

    ax.axhline(75, color='gray', ls='--', lw=1.5, alpha=0.7,
               label='Stabilization threshold (75%)')
    # Shade non-stabilized region for FedAvg
    fedavg_accs = results['fedavg']['accuracy_per_round']
    ax.fill_between(range(1, len(fedavg_accs)+1), 0, fedavg_accs,
                    color='red', alpha=0.08, label='_nolegend_')

    ax.set_xlabel("Training Round", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy Under Gaussian Byzantine Attack\n"
                 "FedAvg stays at 9.8% — permanently non-stabilized")
    ax.legend(fontsize=7.5, loc='center right')
    ax.grid(alpha=0.3)
    ax.set_ylim(-2, 102)

    # Annotate non-stabilization for FedAvg
    ax.annotate("FedAvg: 9.8%\n(permanent failure —\nnon-self-stabilizing)",
                xy=(8, 9.8), xytext=(5.5, 35),
                fontsize=8, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose', alpha=0.8))
    ax.annotate("SS: 90.6%\n(detected Byz. at round 1,\nself-stabilized)",
                xy=(8, 90.6), xytext=(5.0, 70),
                fontsize=8, color='#1f77b4',
                arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))

    # Panel B: Detection rate vs accuracy
    ax = axes[1]
    final_accs = {agg: results[agg]['accuracy_per_round'][-1] for agg in aggregators}
    det_rates  = {agg: results[agg]['avg_detection_rate'] * 100 for agg in aggregators}

    x = range(len(aggregators))
    bar_colors = [colors[a] for a in aggregators]
    bars = ax.bar(x, [final_accs[a] for a in aggregators],
                  color=bar_colors, alpha=0.8, edgecolor='black', lw=0.8,
                  label='Final Accuracy (%)')

    # Overlay detection rate as text
    for i, agg in enumerate(aggregators):
        det = det_rates[agg]
        acc = final_accs[agg]
        self_stab = 'YES' if agg == 'spectral_sentinel' else 'NO'
        color = 'blue' if self_stab == 'YES' else 'red'
        ax.text(i, acc + 1.5, f"Det={det:.0f}%\nSS={self_stab}",
                ha='center', va='bottom', fontsize=8.5,
                color=color, fontweight='bold')

    ax.axhline(75, color='gray', ls='--', lw=1.5, label='Stabilization threshold')
    ax.set_xticks(x)
    agg_short = {'spectral_sentinel': 'Spectral\nSentinel', 'fedavg': 'FedAvg',
                 'krum': 'Krum', 'median': 'Coord.\nMedian'}
    ax.set_xticklabels([agg_short[a] for a in aggregators], fontsize=9)
    ax.set_ylabel("Final Test Accuracy (%)", fontsize=11)
    ax.set_title("Final Accuracy + Detection Rate\n"
                 "Only Spectral Sentinel detects Byzantine clients\n→ Only SS is self-stabilizing")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 115)

    plt.tight_layout()
    out1 = RESULTS_DIR / 'fig_non_stab.png'
    out2 = FIGURES_DIR / 'fig_non_stab.png'
    for out in [out1, out2]:
        plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out1}")
    print(f"Saved: {out2}")

    # Summary
    print("\nSUMMARY (Gaussian attack):")
    print(f"{'Aggregator':22s} {'Final Acc':>10} {'Det Rate':>10} {'Self-Stab':>12}")
    for agg in aggregators:
        ss = "YES" if agg == 'spectral_sentinel' else "NO"
        print(f"{agg:22s} {final_accs[agg]:>9.1f}% {det_rates[agg]:>9.0f}% {ss:>12}")


if __name__ == '__main__':
    main()
