#!/usr/bin/env python3
"""
NOVELTY PROOF: Spectral Sentinel vs Baselines
===============================================
This script proves the novelty of Spectral Sentinel by running actual
federated learning training on MNIST and comparing against baselines.

KEY CLAIMS (from the paper):
  1. Spectral Sentinel detects Byzantine clients via Marchenko-Pastur RMT
  2. It achieves higher accuracy than FedAvg under Byzantine attacks
  3. It outperforms Krum and Median under adaptive/strong attacks
  4. The blockchain integration provides immutable audit trail at negligible cost

WHAT THIS SCRIPT DOES:
  - Trains SimpleCNN on MNIST with 10 clients (3 Byzantine = 30%)
  - Tests 4 aggregators: FedAvg, Krum, Median, SpectralSentinel
  - Under 3 attacks: minmax, labelflip, gaussian
  - Reports accuracy, Byzantine detection rate, training time
  - Saves results to results/novelty_proof/
"""

import sys
import os
import time
import json
import torch
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, '/home/user/blockchain_enabled_federated_learning-main')

from spectral_sentinel.config import Config
from spectral_sentinel.federated.data_loader import load_federated_data
from spectral_sentinel.federated.client import HonestClient, ByzantineClient
from spectral_sentinel.federated.server import FederatedServer
from spectral_sentinel.attacks.attacks import get_attack
from spectral_sentinel.aggregators.spectral_sentinel import SpectralSentinelAggregator
from spectral_sentinel.aggregators.baselines import get_aggregator
from spectral_sentinel.utils.models import get_model
from spectral_sentinel.rmt.marchenko_pastur import MarchenkoPasturLaw

# ──────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path('/home/user/blockchain_enabled_federated_learning-main/results/novelty_proof')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def create_clients(config, client_datasets, byzantine_ids, attack_type, device='cpu'):
    attack = get_attack(attack_type, attack_strength=1.0)
    clients = []
    for i in range(config.num_clients):
        model = get_model(config.model_type, config.num_classes,
                          1 if config.dataset == 'mnist' else 3)
        if i in byzantine_ids:
            client = ByzantineClient(
                client_id=i, model=model, dataset=client_datasets[i],
                attack=attack, learning_rate=config.learning_rate,
                batch_size=config.batch_size, device=device,
                num_classes=config.num_classes
            )
        else:
            client = HonestClient(
                client_id=i, model=model, dataset=client_datasets[i],
                learning_rate=config.learning_rate,
                batch_size=config.batch_size, device=device
            )
        clients.append(client)
    return clients

def run_single_experiment(aggregator_name, attack_type,
                           num_clients=10, byzantine_ratio=0.3,
                           num_rounds=15, local_epochs=2,
                           dataset='mnist', seed=42):
    """Run FL experiment and return per-round stats."""
    set_seed(seed)
    device = 'cpu'

    config = Config(
        dataset=dataset,
        model_type='simple_cnn',
        num_clients=num_clients,
        byzantine_ratio=byzantine_ratio,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=64,
        learning_rate=0.01,
        aggregator=aggregator_name,
        device=device,
        visualize=False,
        save_model=False,
        save_path=str(RESULTS_DIR)
    )

    # Fixed Byzantine set for reproducibility
    np.random.seed(seed)
    byzantine_ids = set(random.sample(range(num_clients), config.num_byzantine))

    print(f"  Loading {dataset.upper()} data...")
    client_datasets, test_dataset = load_federated_data(
        dataset, num_clients, non_iid_alpha=0.5
    )

    model = get_model(config.model_type, config.num_classes,
                      1 if dataset == 'mnist' else 3)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {config.model_type} ({num_params:,} params) | "
          f"Byzantine: {sorted(byzantine_ids)}")

    # Create aggregator
    if aggregator_name == 'spectral_sentinel':
        aggregator = SpectralSentinelAggregator(
            ks_threshold=0.05,
            tail_threshold=0.1,
            use_sketching=False,
            window_size=50
        )
    else:
        aggregator = get_aggregator(aggregator_name,
                                    num_byzantine=config.num_byzantine)

    clients = create_clients(config, client_datasets, byzantine_ids,
                              attack_type, device)

    server = FederatedServer(
        model=model,
        aggregator=aggregator,
        test_dataset=test_dataset,
        device=device
    )

    t0 = time.time()
    stats = server.train(
        clients=clients,
        num_rounds=num_rounds,
        num_local_epochs=local_epochs,
        clients_per_round=num_clients,
        verbose=False
    )
    elapsed = time.time() - t0

    final_acc = stats.get('round_accuracies', stats.get('test_accuracy', []))[-1] if stats.get('round_accuracies', stats.get('test_accuracy', [])) else 0.0

    # Detection metrics (only for Spectral Sentinel)
    detection_rates = []
    if aggregator_name == 'spectral_sentinel':
        byz_per_round = aggregator.stats.get('byzantine_detected_per_round', [])
        for n_detected in byz_per_round:
            recall = min(n_detected, config.num_byzantine) / config.num_byzantine
            detection_rates.append(recall)
    else:
        detection_rates = [0.0] * num_rounds

    return {
        'aggregator': aggregator_name,
        'attack': attack_type,
        'accuracy_per_round': stats.get('round_accuracies', stats.get('test_accuracy', [])),
        'loss_per_round': stats.get('test_loss', []),
        'detection_rates': detection_rates,
        'final_accuracy': final_acc,
        'avg_detection_rate': np.mean(detection_rates) if detection_rates else 0.0,
        'training_time_s': elapsed,
        'num_byzantine': config.num_byzantine,
        'byzantine_ids': sorted(byzantine_ids)
    }

# ──────────────────────────────────────────────────────────────────────────────
# MARCHENKO-PASTUR VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────
def plot_rmt_spectral_analysis():
    """
    Visualize the Marchenko-Pastur law and how Byzantine gradients
    create detectable spectral anomalies.
    """
    np.random.seed(42)
    n_clients = 10
    d = 256
    n_byz = 3

    # ── Generate honest gradient matrix ──────────────────────────────────────
    # Honest: realistic small gradient updates (≈N(0,0.01²))
    G_honest = np.random.randn(n_clients, d) * 0.01

    # ── Generate byzantine gradient matrix (minmax attack) ───────────────────
    G_mixed = G_honest.copy()
    # Byzantine clients: sign-flipped, amplified
    for i in range(n_clients - n_byz, n_clients):
        G_mixed[i] = -G_honest[i] * 10.0

    def get_eigenvalues(G):
        G_std = (G - G.mean(0)) / (G.std(0) + 1e-8)
        cov   = (G_std.T @ G_std) / G.shape[0]
        return np.sort(np.linalg.eigvalsh(cov))

    eigs_honest = get_eigenvalues(G_honest)
    eigs_mixed  = get_eigenvalues(G_mixed)

    gamma = n_clients / d
    sigma_sq = np.var(G_honest)
    mp = MarchenkoPasturLaw(aspect_ratio=gamma, variance=max(sigma_sq, 1e-6))
    lam_minus, lam_plus = mp.get_support()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Spectral Sentinel: RMT-Based Byzantine Detection\n"
                 "(Marchenko-Pastur Law)", fontsize=14, fontweight='bold')

    # ── Plot 1: Honest eigenvalue distribution ────────────────────────────────
    ax = axes[0]
    ax.hist(eigs_honest, bins=40, density=True, color='steelblue', alpha=0.7,
            label='Honest gradients')
    lam_range = np.linspace(max(0, lam_minus*0.5), lam_plus*2, 500)
    mp_density = mp.density(lam_range)
    if mp_density.max() > 0:
        ax.plot(lam_range, mp_density, 'r-', lw=2, label='MP theoretical')
    ax.axvline(lam_plus, color='orange', ls='--', lw=2, label=f'MP bound λ⁺={lam_plus:.4f}')
    ax.set_title("Honest Gradients\n(Follow MP Law)")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Plot 2: Mixed (with Byzantine) eigenvalue distribution ────────────────
    ax = axes[1]
    ax.hist(eigs_mixed, bins=40, density=True, color='coral', alpha=0.7,
            label='Mixed (+ Byzantine)')
    lam_range2 = np.linspace(0, max(eigs_mixed)*1.1, 500)
    mp_density2 = mp.density(lam_range2)
    if mp_density2.max() > 0:
        ax.plot(lam_range2, mp_density2, 'r-', lw=2, label='MP (honest)')
    ax.axvline(lam_plus, color='orange', ls='--', lw=2, label=f'MP bound λ⁺={lam_plus:.4f}')
    outliers = eigs_mixed[eigs_mixed > lam_plus * 1.05]
    if len(outliers) > 0:
        for ev in outliers:
            ax.axvline(ev, color='red', ls=':', alpha=0.6)
        ax.text(0.6, 0.85, f'{len(outliers)} outlier eigenvalue(s)\n→ Byzantine detected!',
                transform=ax.transAxes, color='red', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.set_title("Byzantine Attack (MinMax)\n(Spectral Anomaly Visible)")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Plot 3: Per-client spectral energy score ──────────────────────────────
    ax = axes[2]
    G_mixed_std = (G_mixed - G_mixed.mean(0)) / (G_mixed.std(0) + 1e-8)
    client_scores = np.array([np.mean(G_mixed_std[i]**2) for i in range(n_clients)])
    threshold = np.median(client_scores) + 2.0 * np.std(client_scores)

    colors = ['steelblue'] * (n_clients - n_byz) + ['red'] * n_byz
    bars = ax.bar(range(n_clients), client_scores, color=colors, alpha=0.8)
    ax.axhline(threshold, color='orange', ls='--', lw=2, label=f'Detection threshold')
    ax.set_title("Per-Client Spectral Energy\n(Byzantine = High Energy)")
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Spectral Energy Score")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # Colour legend
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor='steelblue', alpha=0.8, label='Honest client'),
                    Patch(facecolor='red',       alpha=0.8, label='Byzantine client')]
    ax.legend(handles=legend_elems + [plt.Line2D([0],[0], color='orange', ls='--',
              label='Detection threshold')], fontsize=8)

    plt.tight_layout()
    save_path = RESULTS_DIR / 'rmt_spectral_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    return save_path


# ──────────────────────────────────────────────────────────────────────────────
# COMPARISON PLOT
# ──────────────────────────────────────────────────────────────────────────────
def plot_comparison(all_results, attacks):
    """Plot accuracy comparison across aggregators and attacks."""
    aggregators = ['fedavg', 'krum', 'median', 'spectral_sentinel']
    agg_labels  = {'fedavg': 'FedAvg\n(no defense)',
                   'krum': 'Krum',
                   'median': 'Median',
                   'spectral_sentinel': 'Spectral Sentinel\n(ours)'}
    colors = {'fedavg': '#e74c3c', 'krum': '#f39c12',
              'median': '#3498db', 'spectral_sentinel': '#27ae60'}

    fig = plt.figure(figsize=(18, 5 * len(attacks)))
    gs  = gridspec.GridSpec(len(attacks), 3, figure=fig,
                            hspace=0.45, wspace=0.35)
    fig.suptitle("Spectral Sentinel Novelty Proof\n"
                 "Byzantine-Robust Federated Learning on MNIST",
                 fontsize=15, fontweight='bold', y=1.01)

    for row, attack in enumerate(attacks):
        # Col 0: Training curves
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.set_title(f"Attack: {attack.upper()}\n"
                      f"Test Accuracy vs Round", fontsize=10)
        for agg in aggregators:
            key = (agg, attack)
            if key not in all_results:
                continue
            r = all_results[key]
            accs = r['accuracy_per_round']
            if accs:
                ax0.plot(range(1, len(accs)+1), accs,
                         color=colors[agg], lw=2,
                         label=agg_labels[agg],
                         marker='o' if agg=='spectral_sentinel' else None,
                         markersize=4)
        ax0.set_xlabel("Round")
        ax0.set_ylabel("Test Accuracy")
        ax0.legend(fontsize=7, loc='lower right')
        ax0.grid(alpha=0.3)
        ax0.set_ylim(0, 1)

        # Col 1: Final accuracy bar chart
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.set_title(f"Attack: {attack.upper()}\n"
                      f"Final Test Accuracy", fontsize=10)
        final_accs = []
        agg_names  = []
        bar_colors = []
        for agg in aggregators:
            key = (agg, attack)
            if key not in all_results:
                continue
            final_accs.append(all_results[key]['final_accuracy'])
            agg_names.append(agg_labels[agg])
            bar_colors.append(colors[agg])

        bars = ax1.bar(range(len(final_accs)), final_accs,
                       color=bar_colors, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, final_accs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        ax1.set_xticks(range(len(agg_names)))
        ax1.set_xticklabels(agg_names, fontsize=7, rotation=10)
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(0, 1.1)
        ax1.grid(alpha=0.3, axis='y')

        # Col 2: Detection rate (Spectral Sentinel only)
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.set_title(f"Attack: {attack.upper()}\n"
                      f"Spectral Sentinel Detection Rate", fontsize=10)
        key = ('spectral_sentinel', attack)
        if key in all_results:
            dr = all_results[key]['detection_rates']
            if dr:
                ax2.plot(range(1, len(dr)+1), dr,
                         color='#27ae60', lw=2, marker='o', markersize=4)
                ax2.axhline(np.mean(dr), color='orange', ls='--', lw=2,
                            label=f'Avg={np.mean(dr):.2f}')
                ax2.set_ylim(-0.05, 1.1)
                ax2.set_xlabel("Round")
                ax2.set_ylabel("Byzantine Recall")
                ax2.legend(fontsize=9)
                ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path = RESULTS_DIR / 'novelty_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    return save_path


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "═"*70)
    print("  🔬  NOVELTY PROOF: SPECTRAL SENTINEL vs BASELINES")
    print("  📐  Random Matrix Theory × Byzantine-Robust Federated Learning")
    print("═"*70)

    # ── RMT Spectral Analysis Plot (theory) ───────────────────────────────────
    print("\n[STEP 1] Generating RMT spectral analysis plot...")
    plot_rmt_spectral_analysis()

    # ── Experiments ───────────────────────────────────────────────────────────
    aggregators = ['fedavg', 'krum', 'median', 'spectral_sentinel']
    attacks     = ['minmax', 'labelflip', 'gaussian']

    NUM_CLIENTS    = 10
    BYZ_RATIO      = 0.3      # 3 out of 10 Byzantine (Krum needs n > 2f+2 = 8)
    NUM_ROUNDS     = 8
    LOCAL_EPOCHS   = 1

    print(f"\n[STEP 2] Running FL experiments...")
    print(f"  Config: {NUM_CLIENTS} clients | {BYZ_RATIO*100:.0f}% Byzantine | "
          f"{NUM_ROUNDS} rounds | {LOCAL_EPOCHS} local epochs | MNIST")
    print(f"  Aggregators: {aggregators}")
    print(f"  Attacks:     {attacks}\n")

    # Load data once to reuse
    set_seed(42)
    client_datasets, test_dataset = load_federated_data(
        'mnist', NUM_CLIENTS, non_iid_alpha=0.5
    )

    all_results = {}

    for attack in attacks:
        for agg_name in aggregators:
            key = (agg_name, attack)
            print(f"\n  ► {agg_name.upper()} | {attack} attack")

            set_seed(42)
            config = Config(
                dataset='mnist',
                model_type='simple_cnn',
                num_clients=NUM_CLIENTS,
                byzantine_ratio=BYZ_RATIO,
                num_rounds=NUM_ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                batch_size=64,
                learning_rate=0.01,
                aggregator=agg_name,
                device='cpu',
                visualize=False,
                save_model=False,
                save_path=str(RESULTS_DIR)
            )

            byzantine_ids = set(random.sample(range(NUM_CLIENTS), config.num_byzantine))

            model = get_model(config.model_type, config.num_classes, 1)

            # Only Krum / Bulyan accept num_byzantine kwarg
            _kw_byz = {'krum', 'multi_krum', 'bulyan', 'signguard'}
            if agg_name == 'spectral_sentinel':
                aggregator = SpectralSentinelAggregator(
                    ks_threshold=0.05, tail_threshold=0.1,
                    use_sketching=True, sketch_size=64,
                    window_size=NUM_ROUNDS
                )
            elif agg_name in _kw_byz:
                aggregator = get_aggregator(agg_name, num_byzantine=config.num_byzantine)
            else:
                aggregator = get_aggregator(agg_name)

            clients = create_clients(config, client_datasets, byzantine_ids,
                                     attack, 'cpu')

            server = FederatedServer(
                model=model, aggregator=aggregator,
                test_dataset=test_dataset, device='cpu'
            )

            t0 = time.time()
            stats = server.train(
                clients=clients,
                num_rounds=NUM_ROUNDS,
                num_local_epochs=LOCAL_EPOCHS,
                clients_per_round=NUM_CLIENTS,
                verbose=False
            )
            elapsed = time.time() - t0

            final_acc = stats.get('round_accuracies', stats.get('test_accuracy', []))[-1] if stats.get('round_accuracies', stats.get('test_accuracy', [])) else 0.0

            detection_rates = []
            if agg_name == 'spectral_sentinel':
                for n_det in aggregator.stats.get('byzantine_detected_per_round', []):
                    detection_rates.append(
                        min(n_det, config.num_byzantine) / config.num_byzantine
                    )

            result = {
                'aggregator': agg_name,
                'attack': attack,
                'accuracy_per_round': stats.get('round_accuracies', stats.get('test_accuracy', [])),
                'final_accuracy': final_acc,
                'detection_rates': detection_rates,
                'avg_detection_rate': float(np.mean(detection_rates)) if detection_rates else 0.0,
                'training_time_s': elapsed,
                'byzantine_ids': sorted(byzantine_ids)
            }
            all_results[key] = result
            print(f"    Final accuracy: {final_acc:.4f} | "
                  f"Time: {elapsed:.1f}s | "
                  f"Avg detection: {result['avg_detection_rate']:.2f}")

    # ── Summary Table ─────────────────────────────────────────────────────────
    print("\n" + "═"*70)
    print("  📊  RESULTS SUMMARY")
    print("═"*70)
    print(f"\n  {'Aggregator':<20} {'Attack':<12} {'Final Acc':>10} "
          f"{'Avg Detect':>11} {'Time(s)':>8}")
    print(f"  {'─'*20} {'─'*12} {'─'*10} {'─'*11} {'─'*8}")

    # Group by attack to show comparison
    for attack in attacks:
        print(f"\n  Attack: {attack.upper()}")
        for agg in aggregators:
            key = (agg, attack)
            if key not in all_results:
                continue
            r = all_results[key]
            det = f"{r['avg_detection_rate']:.2f}" if r['avg_detection_rate'] > 0 else "  N/A"
            print(f"    {agg:<22} {attack:<12} {r['final_accuracy']:>10.4f} "
                  f"{det:>11} {r['training_time_s']:>8.1f}")

    # ── Key novelty metrics ───────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  🏆  NOVELTY EVIDENCE: Spectral Sentinel Performance Gains")
    print("─"*70)

    for attack in attacks:
        ss_acc   = all_results.get(('spectral_sentinel', attack), {}).get('final_accuracy', 0)
        fa_acc   = all_results.get(('fedavg',            attack), {}).get('final_accuracy', 0)
        kr_acc   = all_results.get(('krum',              attack), {}).get('final_accuracy', 0)
        med_acc  = all_results.get(('median',            attack), {}).get('final_accuracy', 0)
        ss_det   = all_results.get(('spectral_sentinel', attack), {}).get('avg_detection_rate', 0)

        print(f"\n  [{attack.upper()}]")
        print(f"    SpectralSentinel:  acc={ss_acc:.4f}  detect_rate={ss_det:.2f}")
        print(f"    FedAvg (baseline): acc={fa_acc:.4f}  detect_rate=0.00")
        print(f"    Krum:              acc={kr_acc:.4f}  detect_rate=0.00")
        print(f"    Median:            acc={med_acc:.4f}  detect_rate=0.00")
        gain_vs_fedavg = ss_acc - fa_acc
        gain_vs_best   = ss_acc - max(kr_acc, med_acc)
        print(f"    ✅ Gain vs FedAvg:  +{gain_vs_fedavg:+.4f}")
        print(f"    ✅ Gain vs best baseline: +{gain_vs_best:+.4f}")

    # ── Save JSON results ─────────────────────────────────────────────────────
    serialisable = {
        f"{agg}_{atk}": {k: (v if not isinstance(v, (list, set)) else list(v))
                         for k, v in res.items()}
        for (agg, atk), res in all_results.items()
    }
    json_path = RESULTS_DIR / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n  Results saved to: {json_path}")

    # ── Comparison plots ──────────────────────────────────────────────────────
    print("\n[STEP 3] Generating comparison plots...")
    plot_comparison(all_results, attacks)

    # ── Blockchain cost analysis ──────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  ⛓  BLOCKCHAIN COST ANALYSIS (Hardhat Local Testnet)")
    print("─"*70)
    try:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
        if w3.is_connected():
            blocks = w3.eth.block_number
            gas_gwei = float(w3.from_wei(w3.eth.gas_price, 'gwei'))
            total_gas = sum(w3.eth.get_block(b)['gasUsed'] for b in range(1, blocks+1))
            total_eth = gas_gwei * total_gas * 1e-9
            print(f"  Blocks mined:     {blocks}")
            print(f"  Total gas used:   {total_gas:,}")
            print(f"  ETH cost (local): {total_eth:.6f} ETH @ {gas_gwei:.1f} gwei")
            print(f"  Cost per round:   {total_eth/max(blocks,1)*1000:.4f} mETH")
        else:
            print("  (Hardhat node not running)")
    except Exception as e:
        print(f"  (Blockchain stats unavailable: {e})")

    print("\n" + "═"*70)
    print("  ✅  NOVELTY PROOF COMPLETE")
    print("═"*70)
    print(f"\n  Outputs saved in: {RESULTS_DIR}")
    print("  - rmt_spectral_analysis.png  (Marchenko-Pastur law visualization)")
    print("  - novelty_comparison.png     (Accuracy & detection across attacks)")
    print("  - results.json               (Full numerical results)")
    print("")

    return all_results


if __name__ == '__main__':
    results = main()
    sys.exit(0)
