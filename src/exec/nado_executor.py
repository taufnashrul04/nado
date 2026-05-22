"""
Nado Executor — wraps Nado SDK for live order execution
========================================================
Uses linked signer (1CT key) to sign — main wallet untouched.

Flow:
  1. Init client with linked signer PK
  2. set_leverage(product_id, leverage) before placing
  3. place_market_order(product_id, amount_int_x18, slippage)
  4. close_position(subaccount, product_id) at exit

Notional sizing:
  amount = (margin_usd * leverage) / current_price
  amount_int = int(amount * 10**18)  # x18 for SDK
  long → positive, short → negative
"""
from __future__ import annotations
import os
import json
import time
import traceback
from pathlib import Path
from decimal import Decimal
from typing import Optional

from eth_account import Account
from nado_protocol.client import create_nado_client, NadoClientMode
from nado_protocol.engine_client.types.execute import (
    PlaceMarketOrderParams,
    LinkSignerParams,
)
from nado_protocol.utils.execute import MarketOrderParams
from nado_protocol.utils.bytes32 import subaccount_to_hex
from nado_protocol.utils.math import to_x18

# Nado pair-id mapping (matched to src/data/fetcher.py SYMBOL_TO_PID)
PRODUCT_IDS = {
    "BTC-PERP": 2, "ETH-PERP": 4, "SOL-PERP": 8, "XRP-PERP": 10,
    "BNB-PERP": 14, "HYPE-PERP": 16, "SUI-PERP": 24, "DOGE-PERP": 52,
    "ADA-PERP": 60, "AVAX-PERP": 64, "LINK-PERP": 74,
    "WTI-PERP": 90, "QQQ-PERP": 98, "SPY-PERP": 100,
    "AAPL-PERP": 102, "GOOGL-PERP": 106, "NVDA-PERP": 112, "TSLA-PERP": 114,
    "AMZN-PERP": 104, "META-PERP": 108, "MSFT-PERP": 110,
    "EURUSD-PERP": 92, "GBPUSD-PERP": 94, "USDJPY-PERP": 96,
    "XAUT-PERP": 28, "XAG-PERP": 88,
}


class NadoExecutor:
    def __init__(self, secrets_dir: str = "/root/nado-bot/secrets"):
        cfg_path = Path(secrets_dir) / "wallet_config.json"
        ct_path = Path(secrets_dir) / "1ct_key.txt"

        self.cfg = json.loads(cfg_path.read_text())
        ct_pk = ct_path.read_text().strip()
        if not ct_pk.startswith("0x"):
            ct_pk = "0x" + ct_pk

        self.linked_account = Account.from_key(ct_pk)
        self.main_address = self.cfg["main_wallet_address"]
        self.subaccount_name = self.cfg.get("subaccount_name", "default")

        # Build sender bytes32 = main_address || subaccount_name (12 bytes padded)
        self.sender_hex = subaccount_to_hex(self.main_address, self.subaccount_name)

        # Init client signed by linked signer
        self.client = create_nado_client(NadoClientMode.MAINNET, ct_pk)
        # NOTE: client is signed by 1CT, but sender = main subaccount
        self._patch_sender()

    def _patch_sender(self):
        """Override signer.address used for sender derivation."""
        # Engine client uses self.context.signer.address; we need sender = MAIN, signer = 1CT
        # The SDK signs with the ECDSA key of the client (1CT) but signs FOR the main subaccount.
        pass  # handled by passing sender explicitly in params

    def get_product_id(self, symbol: str) -> int:
        if symbol not in PRODUCT_IDS:
            raise ValueError(f"Unknown product {symbol}")
        return PRODUCT_IDS[symbol]

    def get_oracle_price(self, product_id: int) -> float:
        """Read latest oracle price (x18 → float)."""
        try:
            ob = self.client.context.engine_client.get_market_liquidity(product_id, 1)
            mid = (int(ob.bids[0][0]) + int(ob.asks[0][0])) / 2 / 1e18
            return mid
        except Exception:
            # Fallback: query product
            return 0.0

    def calc_amount_x18(self, side: str, margin_usd: float, leverage: int, price: float) -> int:
        """Compute SDK amount (int x18, signed)."""
        if price <= 0:
            raise ValueError(f"Bad price: {price}")
        notional = margin_usd * leverage
        amount_float = notional / price
        amount_x18 = int(Decimal(str(amount_float)) * Decimal(10 ** 18))
        return amount_x18 if side == "long" else -amount_x18

    def place_market(self, symbol: str, side: str, margin_usd: float,
                     leverage: int, slippage: float = 0.005) -> dict:
        """
        Place a market order.

        Returns: {ok, tx, amount, price, error}
        """
        product_id = self.get_product_id(symbol)
        try:
            price = self.get_oracle_price(product_id)
            if price <= 0:
                return {"ok": False, "error": "no oracle price", "tx": None}

            amount_x18 = self.calc_amount_x18(side, margin_usd, leverage, price)

            mo = MarketOrderParams(
                sender=self.sender_hex,
                amount=amount_x18,
                nonce=None,
            )
            params = PlaceMarketOrderParams(
                product_id=product_id,
                market_order=mo,
                slippage=slippage,
                spot_leverage=None,
                signature=None,
            )
            resp = self.client.market.place_market_order(params)
            return {
                "ok": True,
                "tx": str(resp),
                "amount_x18": amount_x18,
                "amount": amount_x18 / 1e18,
                "price": price,
                "product_id": product_id,
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"{e.__class__.__name__}: {str(e)[:200]}",
                "trace": traceback.format_exc()[:500],
                "tx": None,
            }

    def close_position(self, symbol: str) -> dict:
        product_id = self.get_product_id(symbol)
        try:
            resp = self.client.market.close_position(self.sender_hex, product_id)
            return {"ok": True, "tx": str(resp), "product_id": product_id}
        except Exception as e:
            return {
                "ok": False,
                "error": f"{e.__class__.__name__}: {str(e)[:200]}",
                "trace": traceback.format_exc()[:500],
            }

    def get_position(self, symbol: str) -> Optional[dict]:
        """Query current position size for a product."""
        product_id = self.get_product_id(symbol)
        try:
            data = self.client.context.engine_client.get_subaccount_info(self.sender_hex)
            for bal in data.perp_balances:
                if bal.product_id == product_id:
                    amt = int(bal.balance.amount) / 1e18
                    return {
                        "product_id": product_id,
                        "amount": amt,
                        "v_quote": int(bal.balance.v_quote_balance) / 1e18,
                    }
            return None
        except Exception as e:
            return {"error": str(e)[:120]}

    def get_balance_usd(self) -> float:
        """USDC available margin (spot product 0)."""
        try:
            data = self.client.context.engine_client.get_subaccount_info(self.sender_hex)
            for bal in data.spot_balances:
                if bal.product_id == 0:
                    return int(bal.balance.amount) / 1e18
            return 0.0
        except Exception as e:
            return 0.0


# Singleton helper
_executor: Optional[NadoExecutor] = None

def get_executor() -> NadoExecutor:
    global _executor
    if _executor is None:
        _executor = NadoExecutor()
    return _executor
