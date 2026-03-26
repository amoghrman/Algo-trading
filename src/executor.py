import ccxt
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PaperExecutor:

    SYMBOLS = {
        "btc": "BTC/USDT:USDT",
        "eth": "ETH/USDT:USDT",
    }

    DEMO_URLS = {
        "fapiPublic"    : "https://testnet.binancefuture.com/fapi/v1",
        "fapiPublicV2"  : "https://testnet.binancefuture.com/fapi/v2",
        "fapiPrivate"   : "https://testnet.binancefuture.com/fapi/v1",
        "fapiPrivateV2" : "https://testnet.binancefuture.com/fapi/v2",
        "fapiPrivateV3" : "https://testnet.binancefuture.com/fapi/v3",
    }

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.testnet  = testnet
        self.exchange = ccxt.binanceusdm({
            "apiKey"          : api_key,
            "secret"          : api_secret,
            "enableRateLimit" : True,
            "options"         : {
                "defaultType"            : "future",
                "adjustForTimeDifference": True,
            },
        })

        if testnet:
            self.exchange.urls["api"] = self.DEMO_URLS
            print("Running on DEMO TRADING -- no real money at risk")

        # Test with public endpoint only
        try:
            server_time = self.exchange.fetch_time()
            print(f"Exchange connected. Server time: {server_time}")
        except Exception as e:
            print(f"Connection failed: {e}")
            raise

    def get_balance(self) -> float:
        try:
            # Use v2 account endpoint
            response = self.exchange.fapiPrivateV2GetAccount()
            assets   = response.get("assets", [])
            for a in assets:
                if a.get("asset") == "USDT":
                    return float(a.get("availableBalance", 0))
            # Fallback — try totalWalletBalance
            return float(response.get("totalWalletBalance", 0))
        except Exception as e:
            logger.warning(f"Balance fetch failed: {e}")
            return 0.0

    def get_position(self, asset: str) -> dict:
        try:
            symbol   = self.SYMBOLS[asset.lower()]
            sym_clean = symbol.replace("/USDT:USDT", "USDT")
            response = self.exchange.fapiPrivateGetPositionRisk(
                {"symbol": sym_clean}
            )
            for pos in response:
                amt = float(pos.get("positionAmt", 0))
                if amt != 0:
                    return {
                        "size"       : abs(amt),
                        "side"       : "long" if amt > 0 else "short",
                        "entry_price": float(pos.get("entryPrice", 0)),
                        "unrealized" : float(pos.get("unRealizedProfit", 0)),
                    }
        except Exception as e:
            logger.warning(f"Position fetch failed for {asset}: {e}")
        return {"size": 0, "side": None, "entry_price": 0, "unrealized": 0}

    def get_current_price(self, asset: str) -> float:
        symbol = self.SYMBOLS[asset.lower()]
        ticker = self.exchange.fetch_ticker(symbol)
        return float(ticker["last"])

    def place_long(self, asset: str, usdt_amount: float) -> dict:
        symbol = self.SYMBOLS[asset.lower()]
        price  = self.get_current_price(asset)
        qty    = self.exchange.amount_to_precision(
            symbol, usdt_amount / price
        )
        order = self.exchange.create_market_buy_order(
            symbol = symbol,
            amount = float(qty),
            params = {"reduceOnly": False}
        )
        logger.info(f"LONG  {asset.upper()} qty={qty} price={price:.2f}")
        return order

    def place_short(self, asset: str, usdt_amount: float) -> dict:
        symbol = self.SYMBOLS[asset.lower()]
        price  = self.get_current_price(asset)
        qty    = self.exchange.amount_to_precision(
            symbol, usdt_amount / price
        )
        order = self.exchange.create_market_sell_order(
            symbol = symbol,
            amount = float(qty),
            params = {"reduceOnly": False}
        )
        logger.info(f"SHORT {asset.upper()} qty={qty} price={price:.2f}")
        return order

    def close_position(self, asset: str) -> dict:
        symbol   = self.SYMBOLS[asset.lower()]
        position = self.get_position(asset)

        if position["size"] == 0:
            return {"status": "no_position"}

        if position["side"] == "long":
            order = self.exchange.create_market_sell_order(
                symbol = symbol,
                amount = position["size"],
                params = {"reduceOnly": True}
            )
        else:
            order = self.exchange.create_market_buy_order(
                symbol = symbol,
                amount = position["size"],
                params = {"reduceOnly": True}
            )
        logger.info(f"CLOSE {asset.upper()} size={position['size']}")
        return order