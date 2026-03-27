import ccxt
import logging

logger = logging.getLogger(__name__)


class PaperExecutor:

    SYMBOLS = {
        "btc": "BTC/USDT:USDT",
        "eth": "ETH/USDT:USDT",
    }

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.testnet = testnet
        self.exchange = ccxt.binanceusdm({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
            },
        })

        if testnet:
            # Let CCXT switch every relevant endpoint to the Binance futures testnet.
            self.exchange.set_sandbox_mode(True)
            print("Running on DEMO TRADING -- no real money at risk")

        try:
            server_time = self.exchange.fetch_time()
            print(f"Exchange connected. Server time: {server_time}")
        except Exception as e:
            print(f"Connection failed: {e}")
            raise

    def get_balance(self) -> float:
        try:
            balance = self.exchange.fetch_balance({"type": "future"})
            usdt = balance.get("USDT", {})
            free_balance = usdt.get("free")
            total_balance = usdt.get("total")

            if free_balance is not None:
                return float(free_balance)
            if total_balance is not None:
                return float(total_balance)
        except Exception as e:
            logger.warning(f"Balance fetch failed: {e}")
        return 0.0

    def get_position(self, asset: str) -> dict:
        try:
            symbol = self.SYMBOLS[asset.lower()]
            positions = self.exchange.fetch_positions([symbol], {"type": "future"})

            for pos in positions:
                contracts = pos.get("contracts")
                if not contracts:
                    continue

                side = (pos.get("side") or "").lower()
                if side not in {"long", "short"}:
                    continue

                return {
                    "size": float(abs(contracts)),
                    "side": side,
                    "entry_price": float(pos.get("entryPrice") or 0),
                    "unrealized": float(pos.get("unrealizedPnl") or 0),
                }
        except Exception as e:
            logger.warning(f"Position fetch failed for {asset}: {e}")
        return {"size": 0.0, "side": None, "entry_price": 0.0, "unrealized": 0.0}

    def get_current_price(self, asset: str) -> float:
        symbol = self.SYMBOLS[asset.lower()]
        ticker = self.exchange.fetch_ticker(symbol)
        return float(ticker["last"])

    def place_long(self, asset: str, usdt_amount: float) -> dict:
        try:
            symbol = self.SYMBOLS[asset.lower()]
            price = self.get_current_price(asset)
            qty = self.exchange.amount_to_precision(symbol, usdt_amount / price)
            order = self.exchange.create_order(
                symbol=symbol,
                type="MARKET",
                side="buy",
                amount=float(qty),
                params={"reduceOnly": False},
            )
            logger.info(f"LONG  {asset.upper()} qty={qty} price={price:.2f}")
            return order
        except Exception as e:
            logger.error(f"place_long failed for {asset}: {e}")
            return {}

    def place_short(self, asset: str, usdt_amount: float) -> dict:
        try:
            symbol = self.SYMBOLS[asset.lower()]
            price = self.get_current_price(asset)
            qty = self.exchange.amount_to_precision(symbol, usdt_amount / price)
            order = self.exchange.create_order(
                symbol=symbol,
                type="MARKET",
                side="sell",
                amount=float(qty),
                params={"reduceOnly": False},
            )
            logger.info(f"SHORT {asset.upper()} qty={qty} price={price:.2f}")
            return order
        except Exception as e:
            logger.error(f"place_short failed for {asset}: {e}")
            return {}

    def close_position(self, asset: str, size: float | None = None, side: str | None = None) -> dict:
        try:
            symbol = self.SYMBOLS[asset.lower()]
            position = None

            if not size or not side:
                position = self.get_position(asset)
                size = position["size"]
                side = position["side"]

            if not size or not side:
                return {"status": "no_position"}

            order_side = "sell" if side == "long" else "buy"
            order = self.exchange.create_order(
                symbol=symbol,
                type="MARKET",
                side=order_side,
                amount=float(size),
                params={"reduceOnly": True},
            )
            logger.info(f"CLOSE {asset.upper()} size={size}")
            return order
        except Exception as e:
            logger.error(f"close_position failed for {asset}: {e}")
            return {}
