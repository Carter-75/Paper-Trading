"""debug_broadcast.py

Replays the per-symbol analysis pipeline on historical data to reproduce
numpy shape/broadcast errors even when market is closed.

Uses the same data fetch util as the bot (Alpaca first).
"""

import os
import traceback

# Avoid file handler collisions; launcher already redirects stdout when scheduled.
os.environ.setdefault("SCHEDULED_TASK_MODE", "1")

from runner_data_utils import make_client, fetch_closes_with_volume
from strategies.decision_engine import DecisionEngine
from ml_predictor import get_ml_predictor


def main():
    print("[DBG] Starting debug broadcast replay")
    api = make_client(go_live=False)
    de = DecisionEngine()
    ml = get_ml_predictor()

    symbols = [
        "LLY", "AMD", "V", "UNH", "WMT", "JPM", "META", "NVDA", "TSLA", "BRK.B",
        "AAPL", "MSFT", "AMZN", "GOOGL", "PEP", "COST", "ORCL"
    ]

    interval_seconds = 60
    limit_bars = 250

    for sym in symbols:
        print(f"\n[DBG] === {sym} ===")
        try:
            closes, volumes = fetch_closes_with_volume(api, sym, interval_seconds, limit_bars)
            lc = len(closes) if closes else 0
            lv = len(volumes) if volumes else 0
            print(f"[DBG] lens closes={lc} volumes={lv}")

            if closes is None or volumes is None or lc == 0 or lv == 0:
                print("[DBG] no data; skipping")
                continue
            if lc != lv:
                n = min(lc, lv)
                closes = closes[-n:]
                volumes = volumes[-n:]
                print(f"[DBG] aligned to n={n}")

            sig = de.analyze(sym, closes, volumes)
            print(f"[DBG] signal action={sig.action} conf={sig.confidence:.3f} regime={sig.regime}")

            pred, conf = ml.predict(closes, volumes)
            print(f"[DBG] ml pred={pred} conf={conf:.3f}")

        except Exception as e:
            print(f"[DBG][ERROR] {sym}: {type(e).__name__}: {e}")
            print(traceback.format_exc())

    print("\n[DBG] Done")


if __name__ == "__main__":
    main()
