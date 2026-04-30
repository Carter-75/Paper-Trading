import sys
import os
sys.path.append(os.getcwd())
from portfolio_manager import PortfolioManager
from runner_data_utils import make_client, fetch_ohlcv
from config_validated import get_config
from strategies.decision_engine import DecisionEngine

def repair_targets():
    print("Starting one-time target repair...")
    pm = PortfolioManager()
    config = get_config()
    api = make_client(go_live=config.wants_live_mode())
    de = DecisionEngine()
    
    positions = pm.get_all_positions()
    repaired = False
    
    for symbol, pos in positions.items():
        sl = float(pos.get('stop_loss', 0.0))
        tp = float(pos.get('take_profit', 0.0))
        
        if (sl == 0 or tp == 0) and symbol != "TSLA":
            print(f"Repairing {symbol}...")
            try:
                # Fetch recent data for ATR
                _, highs, lows, closes, _ = fetch_ohlcv(api, symbol, interval_seconds=60, limit_bars=100)
                if not closes:
                    print(f"  No data for {symbol}, skipping.")
                    continue
                
                current_price = float(closes[-1])
                atr = de.calculate_atr(closes, highs, lows)
                if atr == 0:
                    atr = current_price * 0.02
                
                sl_mult = getattr(config, 'atr_stop_multiplier', 2.0)
                tp_mult = getattr(config, 'atr_tp_multiplier', 3.0)
                
                new_sl = current_price - (atr * sl_mult)
                new_tp = current_price + (atr * tp_mult)
                
                # Update PM
                pm.update_position(
                    symbol, 
                    float(pos['qty']), 
                    float(pos['avg_entry']), 
                    float(pos['market_value']), 
                    float(pos['unrealized_pl']),
                    confidence=float(pos.get('confidence', 0.5)),
                    stop_loss=new_sl,
                    take_profit=new_tp
                )
                print(f"  Repaired {symbol}: SL=${new_sl:.2f}, TP=${new_tp:.2f}")
                repaired = True
            except Exception as e:
                print(f"  Failed to repair {symbol}: {e}")
                
    if repaired:
        print("Repair complete. Dashboard should update shortly.")
    else:
        print("No eligible positions found for repair (excluding TSLA).")

if __name__ == "__main__":
    repair_targets()
