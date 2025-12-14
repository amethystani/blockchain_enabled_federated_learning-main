# Blockchain-enabled Federated Learning (Spectral Sentinel)

This repository contains a **blockchain-enabled federated learning** research codebase, including:

- A Solidity smart contract + Hardhat project (`contracts/`, `hardhat.config.js`, `scripts/`)
- The **Spectral Sentinel** Python framework for Byzantine-robust federated learning (`spectral_sentinel/`)
- Reproducible experiment runners and utilities

## Quickstart

- **Main interactive CLI**:

```bash
python apps/app.py
```

- **Quick validation (non-interactive)**:

```bash
python apps/app.py --quick
```

- **Blockchain FL demo (local Hardhat)**:

```bash
python demos/demo_blockchain_fl.py
```

- **Run the full experiment suite (long)**:

```bash
bash scripts/shell/run_all.sh
```

## Repository layout

- **`apps/`**: application entrypoints
- **`demos/`**: small runnable demos
- **`experiments/`**: additional experiment entrypoints (e.g. on-chain experiments)
- **`spectral_sentinel/`**: core Python package (attacks, aggregators, FL simulation, experiments)
- **`contracts/`** / **`scripts/`**: Solidity + Hardhat deployment tooling
- **`tests/`**: repository-level tests and debug scripts
- **`docs/`**: detailed guides, workflows, and summaries
- **`paper/`**: LaTeX report (`paper/report.tex`)
- **`results/`**: experiment outputs
- **`data/`**: datasets/cache
- **`logs/`**: logs and run artifacts

## Documentation

Start here:

- `docs/QUICKSTART.md`
- `docs/RUNNING_GUIDE.md`
- `docs/BLOCKCHAIN_SETUP_GUIDE.md`
- `docs/SPECTRAL_SENTINEL_README.md`

## Citation

BibTeX citation for the original paper:

```bibtex
@article{wilhelmi2021blockchain,
  title={Blockchain-enabled Server-less Federated Learning},
  author={Wilhelmi, Francesc and Giupponi, Lorenza and Dini, Paolo},
  journal={arXiv preprint arXiv:2112.07938},
  year={2021}
}
```

(See `CITATION.cff` for citation metadata.)
