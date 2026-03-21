#!/usr/bin/env python3
"""
local_faucet.py — HTTP faucet server for local Hardhat FL experiments.

Uses Hardhat account[0] (deployer, 10,000 ETH) to distribute ETH to
any address on demand. This removes the need for testnet faucets or
manual ETH transfers when running local blockchain-FL experiments.

Endpoints:
  GET  /accounts          — List all Hardhat accounts with balances
  POST /fund              — Fund an address
       body: {"to": "0x...", "eth": "1.0"}   (JSON)
  GET  /health            — Health check
  GET  /status            — Node + contract status

Usage:
  python scripts/local_faucet.py [--port 8765] [--rpc http://127.0.0.1:8545]
"""

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from web3 import Web3

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_PORT = 8765
DEFAULT_RPC = "http://127.0.0.1:8545"
REPO_ROOT = Path(__file__).parent.parent
DEPLOYMENT_FILE = REPO_ROOT / "local_deployment.json"

# Hardhat's well-known private keys (accounts 0-9)
HARDHAT_PRIVATE_KEYS = [
    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",  # account[0]
    "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
    "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
    "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6",
    "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a",
    "0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba",
    "0x92db14e403b83dfe3df233f83dfa3a0d7096f21ca9b0d6d6b8d88b2b4ec1564",
    "0x4bbbf85ce3377467afe5d46f804f221813b2bb87f24d81f60f1fcdbf7cbf4356",
    "0xdbda1821b80551c9d65939329250132c444b91a8b1c09d3cf046965ad0eca3e9",
    "0x2a871d0798f97d79848a013d4936a73bf4cc922c825d33c1cf7073dff6d409c6",
]

w3: Web3 = None
faucet_account = None
deployment_info: dict = {}


def load_deployment():
    """Load deployment info from local_deployment.json."""
    global deployment_info
    if DEPLOYMENT_FILE.exists():
        with open(DEPLOYMENT_FILE) as f:
            deployment_info = json.load(f)
    return deployment_info


def init_web3(rpc_url: str):
    """Initialize Web3 connection and faucet account."""
    global w3, faucet_account
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        print(f"ERROR: Cannot connect to {rpc_url}")
        sys.exit(1)
    faucet_account = w3.eth.account.from_key(HARDHAT_PRIVATE_KEYS[0])
    print(f"Faucet account: {faucet_account.address}")
    bal = w3.eth.get_balance(faucet_account.address)
    print(f"Faucet balance: {w3.from_wei(bal, 'ether'):.2f} ETH")


def get_accounts_info():
    """Return info for all available Hardhat accounts."""
    accounts = []
    for i, pk in enumerate(HARDHAT_PRIVATE_KEYS):
        acc = w3.eth.account.from_key(pk)
        try:
            bal = w3.eth.get_balance(acc.address)
            accounts.append({
                "index": i,
                "address": acc.address,
                "private_key": pk,
                "balance_eth": float(w3.from_wei(bal, "ether")),
                "is_faucet": i == 0
            })
        except Exception:
            accounts.append({"index": i, "address": acc.address, "error": "Could not fetch balance"})
    return accounts


def fund_address(to_address: str, amount_eth: float) -> dict:
    """Send ETH from faucet to an address."""
    if not w3.is_address(to_address):
        return {"success": False, "error": "Invalid address"}

    to_address = Web3.to_checksum_address(to_address)
    amount_wei = w3.to_wei(amount_eth, "ether")

    faucet_bal = w3.eth.get_balance(faucet_account.address)
    if faucet_bal < amount_wei + w3.to_wei(0.001, "ether"):
        return {"success": False, "error": "Faucet balance too low"}

    try:
        nonce = w3.eth.get_transaction_count(faucet_account.address)
        tx = {
            "to": to_address,
            "value": amount_wei,
            "gas": 21000,
            "gasPrice": w3.eth.gas_price,
            "nonce": nonce,
            "chainId": w3.eth.chain_id
        }
        signed = w3.eth.account.sign_transaction(tx, private_key=HARDHAT_PRIVATE_KEYS[0])
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)

        new_balance = w3.eth.get_balance(to_address)
        return {
            "success": True,
            "tx_hash": tx_hash.hex(),
            "amount_eth": amount_eth,
            "to": to_address,
            "new_balance_eth": float(w3.from_wei(new_balance, "ether")),
            "block": receipt.blockNumber
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ─── HTTP Handler ─────────────────────────────────────────────────────────────

class FaucetHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        # Quiet logging — only errors
        if "GET /health" not in format % args:
            print(f"[faucet] {format % args}")

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/health":
            self.send_json({"status": "ok", "connected": w3.is_connected()})

        elif path == "/accounts":
            accounts = get_accounts_info()
            self.send_json({"accounts": accounts, "count": len(accounts)})

        elif path == "/status":
            info = load_deployment()
            try:
                block = w3.eth.block_number
                chain_id = w3.eth.chain_id
                faucet_bal = float(w3.from_wei(w3.eth.get_balance(faucet_account.address), "ether"))
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
                return
            self.send_json({
                "connected": True,
                "block": block,
                "chain_id": chain_id,
                "faucet_address": faucet_account.address,
                "faucet_balance_eth": faucet_bal,
                "contract_address": info.get("contractAddress", "not deployed"),
                "network": info.get("network", "unknown")
            })

        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/fund":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                to_addr = data.get("to", "")
                amount = float(data.get("eth", 1.0))
                if amount <= 0 or amount > 100:
                    self.send_json({"error": "amount must be 0 < eth ≤ 100"}, 400)
                    return
            except Exception:
                self.send_json({"error": "Invalid JSON body. Expected {\"to\": \"0x...\", \"eth\": 1.0}"}, 400)
                return

            result = fund_address(to_addr, amount)
            status = 200 if result["success"] else 400
            self.send_json(result, status)

        elif path == "/fund_all":
            # Convenience: fund all 10 FL client accounts (index 1-9) with given amount
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                amount = float(data.get("eth", 1.0))
            except Exception:
                amount = 1.0
            results = []
            for pk in HARDHAT_PRIVATE_KEYS[1:]:
                acc = w3.eth.account.from_key(pk)
                results.append(fund_address(acc.address, amount))
            success_count = sum(1 for r in results if r.get("success"))
            self.send_json({"funded": success_count, "total": len(results), "results": results})

        else:
            self.send_json({"error": "Not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Local Hardhat faucet server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--rpc", type=str, default=DEFAULT_RPC)
    args = parser.parse_args()

    print(f"\n=== Local Hardhat Faucet ===")
    print(f"RPC: {args.rpc}")

    init_web3(args.rpc)
    load_deployment()

    server = HTTPServer(("0.0.0.0", args.port), FaucetHandler)
    print(f"\nFaucet running at http://localhost:{args.port}")
    print("Endpoints:")
    print(f"  GET  http://localhost:{args.port}/accounts")
    print(f"  GET  http://localhost:{args.port}/status")
    print(f"  POST http://localhost:{args.port}/fund  (body: {{\"to\":\"0x...\",\"eth\":1.0}})")
    print(f"  POST http://localhost:{args.port}/fund_all  (body: {{\"eth\":1.0}})")
    print("\nPress Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFaucet stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
