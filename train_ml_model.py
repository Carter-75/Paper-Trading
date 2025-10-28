#!/usr/bin/env python3
"""
Train ML model using historical data
Run after collecting 3+ months of trading data
"""

import sys
from ml_predictor import TradingMLPredictor
import config

# Import after loading config
def get_runner_functions():
    """Import runner functions after config is loaded"""
    from runner import make_client, fetch_closes_with_volume
    return make_client, fetch_closes_with_volume

def collect_training_data(symbols: list, interval_seconds: int, bars: int = 500):
    """Collect historical data for training"""
    print(f"Collecting data for {len(symbols)} symbols...")
    
    # Import runner functions
    make_client, fetch_closes_with_volume = get_runner_functions()
    
    client = make_client(allow_missing=False, go_live=False)
    training_data = []
    
    for symbol in symbols:
        print(f"  Fetching {symbol}...")
        try:
            closes, volumes = fetch_closes_with_volume(client, symbol, interval_seconds, bars)
            
            if len(closes) < 50:
                continue
            
            # Create labels: 1 if next price is higher, 0 if lower
            for i in range(40, len(closes) - 1):
                window_closes = closes[:i]
                window_volumes = volumes[:i]
                
                # Label: did price go up next bar?
                label = 1 if closes[i+1] > closes[i] else 0
                
                training_data.append((window_closes, window_volumes, label))
        
        except Exception as e:
            print(f"  Error with {symbol}: {e}")
    
    print(f"\nCollected {len(training_data)} training samples")
    return training_data


def main():
    # Symbols to train on (use diverse stocks)
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META",
        "JPM", "V", "WMT", "JNJ", "PG", "UNH", "HD", "DIS",
        "SPY", "QQQ"  # Include ETFs for diverse patterns
    ]
    
    interval_seconds = int(config.DEFAULT_INTERVAL_SECONDS)
    
    # Collect data
    training_data = collect_training_data(symbols, interval_seconds, bars=500)
    
    if len(training_data) < 100:
        print("Not enough training data collected. Need 100+ samples.")
        return 1
    
    # Train model
    predictor = TradingMLPredictor(config.ML_MODEL_PATH if hasattr(config, 'ML_MODEL_PATH') else "ml_model.pkl")
    success = predictor.train(training_data, test_size=0.3)
    
    if success:
        print("\n[OK] Model training complete!")
        print(f"Model saved to {predictor.model_path}")
        print("\nTo use the model:")
        print("1. Set ENABLE_ML_PREDICTION=1 in .env (already enabled by default)")
        print("2. Run your bot normally - it will use ML predictions")
        return 0
    else:
        print("\n[ERROR] Model training failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

