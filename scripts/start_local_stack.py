#!/usr/bin/env python3
"""
start_local_stack.py — One-command local blockchain-FL stack launcher.

Starts everything needed for reproducible local blockchain-FL experiments:
  1. Hardhat local node  (http://127.0.0.1:8545, 25 accounts × 10,000 ETH)
  2. SpectralBFT.sol deployment (saves to local_deployment.json)
  3. Local faucet HTTP server  (http://127.0.0.1:8765)

Zero external dependencies — no testnet tokens, no API keys, no wallet setup.
Reduces experiment setup from hours to <5 minutes.

Usage:
  python scripts/start_local_stack.py                   # start everything
  python scripts/start_local_stack.py --no-faucet       # skip faucet
  python scripts/start_local_stack.py --port 8766       # custom faucet port
  python scripts/start_local_stack.py --stop            # kill Hardhat node
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import threading
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).parent.parent
HARDHAT_PID_FILE = REPO_ROOT / ".hardhat.pid"
HARDHAT_LOG_FILE = REPO_ROOT / ".hardhat.log"
DEPLOYMENT_FILE = REPO_ROOT / "local_deployment.json"

HARDHAT_RPC = "http://127.0.0.1:8545"
FAUCET_PORT = 8765


def is_port_open(host: str, port: int) -> bool:
    """Check if a TCP port is accepting connections."""
    import socket
    try:
        s = socket.create_connection((host, port), timeout=1)
        s.close()
        return True
    except OSError:
        return False


def wait_for_rpc(url: str, timeout: int = 60) -> bool:
    """Poll RPC endpoint until ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.post(url, json={"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1}, timeout=2)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def start_hardhat_node() -> subprocess.Popen:
    """Start Hardhat node in background."""
    print("Starting Hardhat local node...", flush=True)

    # Kill any existing node
    if HARDHAT_PID_FILE.exists():
        try:
            pid = int(HARDHAT_PID_FILE.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
        except Exception:
            pass
        HARDHAT_PID_FILE.unlink(missing_ok=True)

    log_f = open(HARDHAT_LOG_FILE, "w")
    proc = subprocess.Popen(
        ["npx", "hardhat", "node"],
        cwd=str(REPO_ROOT),
        stdout=log_f,
        stderr=log_f,
        preexec_fn=os.setsid  # new process group for clean shutdown
    )
    HARDHAT_PID_FILE.write_text(str(proc.pid))
    print(f"  Hardhat node PID: {proc.pid}")

    print("  Waiting for node to be ready...", end="", flush=True)
    if wait_for_rpc(HARDHAT_RPC, timeout=60):
        print(" ready!")
    else:
        print(" TIMEOUT!")
        proc.kill()
        print(f"  Check {HARDHAT_LOG_FILE} for errors.")
        sys.exit(1)

    return proc


def deploy_contract() -> dict:
    """Deploy SpectralBFT.sol to local Hardhat node."""
    print("Deploying SpectralBFT.sol...", flush=True)

    result = subprocess.run(
        ["npx", "hardhat", "run", "scripts/deploy_spectral_bft.js", "--network", "localhost"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("  DEPLOYMENT FAILED!")
        print(result.stdout[-2000:] if result.stdout else "")
        print(result.stderr[-2000:] if result.stderr else "")
        sys.exit(1)

    if not DEPLOYMENT_FILE.exists():
        print("  ERROR: local_deployment.json not created!")
        sys.exit(1)

    with open(DEPLOYMENT_FILE) as f:
        info = json.load(f)

    print(f"  Contract: {info['contractAddress']}")
    return info


def start_faucet_thread(port: int):
    """Start local faucet in a daemon thread."""
    import importlib.util
    faucet_path = REPO_ROOT / "scripts" / "local_faucet.py"
    spec = importlib.util.spec_from_file_location("local_faucet", faucet_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod.init_web3(HARDHAT_RPC)
    mod.load_deployment()

    from http.server import HTTPServer
    server = HTTPServer(("0.0.0.0", port), mod.FaucetHandler)

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def stop_stack():
    """Kill the Hardhat node."""
    if HARDHAT_PID_FILE.exists():
        pid = int(HARDHAT_PID_FILE.read_text().strip())
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            print(f"Hardhat node (PID {pid}) stopped.")
        except ProcessLookupError:
            print(f"Process {pid} not found (already stopped).")
        HARDHAT_PID_FILE.unlink(missing_ok=True)
    else:
        print("No running Hardhat node found.")


def print_summary(info: dict, faucet_port: int):
    """Print summary table."""
    accounts = info.get("accounts", [])
    print("\n" + "═" * 60)
    print("  SpectralBFT Local Stack — Ready")
    print("═" * 60)
    print(f"  RPC URL:          {HARDHAT_RPC}")
    print(f"  Chain ID:         {info.get('chainId', '31337')}")
    print(f"  Contract:         {info['contractAddress']}")
    print(f"  Faucet:           http://localhost:{faucet_port}")
    print(f"  Deployer:         {info.get('deployer', 'N/A')}")
    print(f"  Accounts:         {len(accounts)} (10,000 ETH each)")
    print("─" * 60)
    print("  Quick start:")
    print("    # Fund FL client accounts:")
    print(f"    curl -X POST http://localhost:{faucet_port}/fund_all -H 'Content-Type: application/json' -d '{{\"eth\":1.0}}'")
    print()
    print("    # Run local experiment:")
    print("    python experiments/blockchain/experiment_spectral_bft_local.py")
    print("─" * 60)
    print("  First 5 accounts:")
    for acc in accounts[:5]:
        print(f"    [{acc['index']}] {acc['address']}  ({acc.get('balance_eth', '?')} ETH)")
    if len(accounts) > 5:
        print(f"    ... and {len(accounts) - 5} more (see local_deployment.json)")
    print("═" * 60)


def main():
    parser = argparse.ArgumentParser(description="Start local blockchain-FL stack")
    parser.add_argument("--no-faucet", action="store_true", help="Skip faucet server")
    parser.add_argument("--port", type=int, default=FAUCET_PORT, help="Faucet port")
    parser.add_argument("--stop", action="store_true", help="Stop Hardhat node and exit")
    parser.add_argument("--deploy-only", action="store_true", help="Skip node start, only deploy")
    args = parser.parse_args()

    if args.stop:
        stop_stack()
        return

    if not args.deploy_only:
        hardhat_proc = start_hardhat_node()
    else:
        if not is_port_open("127.0.0.1", 8545):
            print("ERROR: No Hardhat node detected on :8545. Run without --deploy-only first.")
            sys.exit(1)

    info = deploy_contract()

    if not args.no_faucet:
        print(f"Starting local faucet on port {args.port}...", flush=True)
        faucet_server = start_faucet_thread(args.port)
        time.sleep(0.5)
        print(f"  Faucet ready at http://localhost:{args.port}")

    print_summary(info, args.port)

    if not args.deploy_only:
        print("\nStack running. Press Ctrl+C to stop.\n")
        try:
            hardhat_proc.wait()
        except KeyboardInterrupt:
            print("\nShutting down...")
            stop_stack()


if __name__ == "__main__":
    main()
