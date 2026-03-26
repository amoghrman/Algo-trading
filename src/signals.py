import numpy as np
import pickle
import os
from stable_baselines3 import PPO
from sklearn.preprocessing import RobustScaler
import warnings

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

class TradingSignalEngine:
    """
    Loads the saved supervised models + RL agents.
    Given live features, returns:
      - action (0=SELL short, 1=HOLD, 2=BUY long)
      - position_size (0.25, 0.50, or 0.75)
      - probabilities [p_sell, p_hold, p_buy]
    """

    # Thresholds from Phase 3 threshold analysis
    THRESHOLDS = {
        "BTC": {"buy": 0.55, "sell": 0.65},
        "ETH": {"buy": 0.50, "sell": 0.60},
    }

    # RL action map
    ACTION_MAP = {
        0: ("hold",  0.00),
        1: ("long",  0.25),
        2: ("long",  0.50),
        3: ("long",  0.75),
        4: ("short", 0.25),
        5: ("short", 0.50),
        6: ("short", 0.75),
        7: ("close", 0.00),
    }

    def __init__(self, models_dir: str, rl_dir: str, processed_dir: str):
        self.models_dir    = models_dir
        self.rl_dir        = rl_dir
        self.processed_dir = processed_dir

        self.clf    = {}   # supervised classifiers
        self.agents = {}   # RL agents
        self.scaler = {}   # feature scalers

        self._load_all()

    def _load_all(self):
        for asset in ["btc", "eth"]:
            # Load supervised model
            clf_path = os.path.join(self.models_dir, f"{asset}_best_model.pkl")
            with open(clf_path, "rb") as f:
                self.clf[asset] = pickle.load(f)
            print(f"Loaded {asset.upper()} classifier")

            # Load RL agent
            agent_path = os.path.join(self.rl_dir, f"{asset}_ppo_agent")
            self.agents[asset] = PPO.load(agent_path)
            print(f"Loaded {asset.upper()} PPO agent")

            # Load scaler from Phase 2
            splits_path = os.path.join(self.processed_dir, f"{asset}_features.pkl")
            with open(splits_path, "rb") as f:
                splits = pickle.load(f)
            self.scaler[asset] = splits["scaler"]
            print(f"Loaded {asset.upper()} scaler")

    def get_signal(self,
                   asset         : str,
                   feature_row   : np.ndarray,
                   position      : float = 0.0,
                   unrealized_pnl: float = 0.0,
                   hold_hours    : int   = 0,
                   capital_ratio : float = 1.0) -> dict:
        """
        Given a single row of raw features, returns trading signal.

        Args:
            asset          : "btc" or "eth"
            feature_row    : 1D numpy array of 31 raw features
            position       : current position indicator (-1=short, 0=flat, 1=long)
            unrealized_pnl : current unrealized PnL as fraction
            hold_hours     : how many hours current position has been open
            capital_ratio  : current capital / initial capital

        Returns:
            dict with keys:
              direction    : "long", "short", "hold", "close"
              size         : 0.0, 0.25, 0.50, 0.75
              prob_buy     : float
              prob_sell    : float
              prob_hold    : float
              raw_action   : int (0-7)
        """
        asset_key = asset.lower()
        ASSET_KEY = asset.upper()

        # Step 1 — scale features
        scaled = self.scaler[asset_key].transform(
            feature_row.reshape(1, -1)
        )[0]

        # Step 2 — get probabilities from supervised model
        probs    = self.clf[asset_key].predict_proba(scaled.reshape(1, -1))[0]
        p_sell   = float(probs[0])
        p_hold   = float(probs[1])
        p_buy    = float(probs[2])

        # Step 3 — build RL observation
        pos_indicator = position   # -1, 0, or 1
        hold_norm     = min(hold_hours / 6, 1.0)

        obs = np.concatenate([
            scaled,
            np.array([p_sell, p_hold, p_buy], dtype=np.float32),
            np.array([
                float(pos_indicator),
                float(unrealized_pnl),
                float(hold_norm),
                float(capital_ratio)
            ], dtype=np.float32)
        ]).astype(np.float32)

        # Step 4 — RL agent decides action
        raw_action, _ = self.agents[asset_key].predict(obs, deterministic=True)
        direction, size = self.ACTION_MAP[int(raw_action)]

        # Step 5 — probability gate
        # Even if RL says buy/sell, block if probability is below threshold
        thresholds = self.THRESHOLDS[ASSET_KEY]
        if direction == "long"  and p_buy  < thresholds["buy"]:
            direction, size = "hold", 0.0
        if direction == "short" and p_sell < thresholds["sell"]:
            direction, size = "hold", 0.0

        return {
            "direction"  : direction,
            "size"       : size,
            "prob_buy"   : p_buy,
            "prob_sell"  : p_sell,
            "prob_hold"  : p_hold,
            "raw_action" : int(raw_action),
        }