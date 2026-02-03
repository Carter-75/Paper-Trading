#!/usr/bin/env python3
"""
Machine Learning Predictor for Trading Bot
Uses Random Forest to predict next price move
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
import traceback


class TradingMLPredictor:
    """Random Forest predictor for stock price movement"""
    
    def __init__(self, model_path: str = "ml_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        
    def extract_features(self, closes: List[float], volumes: List[float] = None,
                        rsi: float = None) -> List[float]:
        """
        Extract features from price data.
        Features:
        - Last 10 returns (%)
        - RSI
        - Volume trend
        - Price momentum
        - Volatility
        """
        if len(closes) < 15:
            return None
        
        features = []
        
        # 1. Recent returns (last 10)
        returns = [(closes[i] - closes[i-1]) / closes[i-1] 
                   for i in range(len(closes)-10, len(closes))]
        features.extend(returns)
        
        # 2. RSI (normalized)
        if rsi is not None:
            features.append(rsi / 100.0)
        else:
            features.append(0.5)  # Neutral
        
        # 3. Volume trend (if available)
        if volumes and len(volumes) >= 10:
            recent_vol = sum(volumes[-5:]) / 5
            avg_vol = sum(volumes[-15:-5]) / 10
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0
            features.append(min(3.0, vol_ratio) / 3.0)  # Normalize to [0, 1]
        else:
            features.append(0.5)
        
        # 4. Price momentum (20-bar)
        if len(closes) >= 20:
            momentum = (closes[-1] - closes[-20]) / closes[-20]
            features.append(momentum)
        else:
            features.append(0.0)
        
        # 5. Volatility (10-bar std)
        if len(closes) >= 10:
            recent_returns = [(closes[i] - closes[i-1]) / closes[i-1] 
                             for i in range(len(closes)-10, len(closes))]
            volatility = np.std(recent_returns)
            features.append(min(volatility * 10, 1.0))  # Normalize
        else:
            features.append(0.0)
        
        return features
    
    def train(self, historical_data: List[Tuple[List[float], List[float], int]],
             test_size: float = 0.3):
        """
        Train the model on historical data.
        historical_data: List of (closes, volumes, label) where label is 1=up, 0=down
        """
        if len(historical_data) < 50:
            print("Not enough data to train (need 50+ samples)")
            return False
        
        X = []
        y = []
        
        for closes, volumes, label in historical_data:
            features = self.extract_features(closes, volumes)
            if features:
                X.append(features)
                y.append(label)
        
        if len(X) < 50:
            print("Not enough valid features extracted")
            return False
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"ML Model Trained:")
        print(f"  Train accuracy: {train_score:.2%}")
        print(f"  Test accuracy: {test_score:.2%}")
        print(f"  Training samples: {len(X_train)}")
        
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        return True
    
    def predict(self, closes: List[float], volumes: List[float] = None,
               rsi: float = None) -> Tuple[int, float]:
        """
        Predict next move.
        Returns (prediction, confidence) where:
        - prediction: 1=up, 0=down
        - confidence: 0.0-1.0
        """
        if not self.is_trained or self.model is None:
            return (1, 0.5)  # Neutral
        
        features = self.extract_features(closes, volumes, rsi)
        if not features:
            return (1, 0.5)
        
        X = np.array([features])
        
        # Get prediction and probability
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        return (int(prediction), float(confidence))
    
    def save_model(self):
        """Save trained model to disk"""
        if self.model:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {self.model_path}")
    
    def load_model(self) -> bool:
        """Load trained model from disk"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                print(f"Model loaded from {self.model_path}")
                return True
            except Exception as e:
                print(f"Failed to load model: {e}")
                return False
        return False

    def check_and_retrain(self) -> bool:
        """
        Check if model is stale (>24h) and retrain if needed.
        Returns True if retraining occurred.
        """
        if not os.path.exists(self.model_path):
             # Model missing, try to train
             from ml_predictor import auto_train_model_if_needed
             return auto_train_model_if_needed(self)
             
        import time
        try:
            mod_time = os.path.getmtime(self.model_path)
            age_hours = (time.time() - mod_time) / 3600
            
            if age_hours > 24:
                print(f"[ML] Model is stale ({age_hours:.1f}h). Triggering Morning Retraining...")
                # We need to call the global auto-train function
                # Note: imports inside method to avoid circular dependency issues at top level
                from ml_predictor import auto_train_model_if_needed
                return auto_train_model_if_needed(self)
                
        except Exception as e:
            print(f"[ML] Start check failed: {e}")
            
        return False


# Global instance
_ml_predictor = None
_auto_train_attempted = False

def auto_train_model_if_needed(predictor: TradingMLPredictor) -> bool:
    """
    Auto-train ML model if it doesn't exist.
    Uses FULL training set (17 symbols, 500 bars) for best accuracy.
    User can press Ctrl+C to skip if needed.
    """
    global _auto_train_attempted
    
    # Only try once per process
    if _auto_train_attempted:
        return False
    _auto_train_attempted = True
    
    # Check if model already exists
    if os.path.exists(predictor.model_path):
        return False
    
    print("\n" + "="*70)
    print("ML MODEL NOT FOUND - AUTO-TRAINING NOW (FULL DATASET)")
    print("="*70)
    print("Training on 17 symbols with 500 bars each (~5-10 minutes)...")
    print("")
    print("[!]  Press Ctrl+C at any time to TRAIN WITH WHAT'S COLLECTED SO FAR")
    print("    (ML stays enabled, just with fewer symbols)")
    print("="*70)
    print("")
    
    try:
        # Import here to avoid circular dependency
        import config
        from runner import make_client, fetch_closes_with_volume
        
        # FULL training set (17 diverse symbols - same as train_ml_model.py)
        symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META",
            "JPM", "V", "WMT", "JNJ", "PG", "UNH", "HD", "DIS",
            "SPY", "QQQ"  # Include ETFs for diverse patterns
        ]
        interval_seconds = int(getattr(config, 'DEFAULT_INTERVAL_SECONDS', 900))
        training_data = []
        
        print(f"Connecting to market data...")
        client = make_client(allow_missing=False, go_live=False)
        print("")
        
        for idx, symbol in enumerate(symbols, 1):
            print(f"  [{idx}/{len(symbols)}] Fetching {symbol}...", end=" ", flush=True)
            try:
                closes, volumes = fetch_closes_with_volume(client, symbol, interval_seconds, 500)
                
                if len(closes) < 50:
                    print("(insufficient data, skipping)")
                    continue
                
                # Create labels: 1 if next price is higher, 0 if lower
                samples_added = 0
                for i in range(40, len(closes) - 1):
                    window_closes = closes[:i]
                    window_volumes = volumes[:i]
                    label = 1 if closes[i+1] > closes[i] else 0
                    training_data.append((window_closes, window_volumes, label))
                    samples_added += 1
                
                print(f"({samples_added} samples)")
            
            except KeyboardInterrupt:
                # User pressed Ctrl+C - stop fetching and train with what we have
                print("\n")
                print("="*70)
                print("[!]  TRAINING INTERRUPTED - USING WHAT WE HAVE SO FAR")
                print("="*70)
                print(f"Collected {len(training_data)} samples from {idx-1}/{len(symbols)} symbols")
                print("")
                
                # Try to train with what we have (require at least 50 samples minimum)
                if len(training_data) >= 50:
                    print(f"Training model with {len(training_data)} samples...")
                    print("(This is less than the full dataset but still usable)")
                    print("")
                    success = predictor.train(training_data, test_size=0.3)
                    if success:
                        print("")
                        print("="*70)
                        print("[OK] MODEL TRAINED WITH PARTIAL DATA!")
                        print("="*70)
                        print(f"Model saved to: {predictor.model_path}")
                        print(f"Trained on {idx-1}/{len(symbols)} symbols")
                        print("ML prediction is now ENABLED!")
                        print("")
                        print("For full accuracy with all 17 symbols, run:")
                        print("  python train_ml_model.py")
                        print("="*70 + "\n")
                        return True
                    else:
                        print("[X] Training failed with collected data.")
                        print("ML will be DISABLED. To train later: python train_ml_model.py")
                        print("="*70 + "\n")
                        return False
                else:
                    print(f"[X] Not enough data for training ({len(training_data)} samples, need 50+)")
                    print("ML will be DISABLED for this run.")
                    print("To train later: python train_ml_model.py")
                    print("="*70 + "\n")
                    return False
            
            except Exception as e:
                print(f"(error: {e})")
                print(traceback.format_exc())
        
        if len(training_data) < 100:
            print(f"\n[X] Not enough data collected ({len(training_data)} samples, need 100+)")
            print("ML prediction will be DISABLED for this run.")
            print("To manually train: python train_ml_model.py")
            print("")
            return False
        
        print("")
        print(f"Training Random Forest model on {len(training_data)} samples...")
        print("(This may take 1-2 minutes...)")
        success = predictor.train(training_data, test_size=0.3)
        
        if success:
            print("")
            print("="*70)
            print("[OK] AUTO-TRAINING COMPLETE (FULL DATASET)!")
            print("="*70)
            print(f"Model saved to: {predictor.model_path}")
            print("ML prediction is now ENABLED and ready to use!")
            print("="*70 + "\n")
            return True
        else:
            print("[X] Auto-training failed. ML will be disabled.")
            return False
    
    except KeyboardInterrupt:
        # User pressed Ctrl+C during initial setup (before any data collected)
        print("\n")
        print("="*70)
        print("[!]  TRAINING INTERRUPTED BEFORE DATA COLLECTION")
        print("="*70)
        print("No data was collected yet, so ML will be DISABLED for this run.")
        print("To train later: python train_ml_model.py")
        print("="*70 + "\n")
        return False
    
    except Exception as e:
        print(f"\n[X] Auto-training error: {e}")
        print(traceback.format_exc())
        print("ML prediction will be DISABLED for this run.")
        print("To manually train: python train_ml_model.py")
        print("")
        return False

def get_ml_predictor() -> TradingMLPredictor:
    """Get global ML predictor instance"""
    global _ml_predictor
    if _ml_predictor is None:
        _ml_predictor = TradingMLPredictor()
        
        _ml_predictor = TradingMLPredictor()
        
        # Check if model exists and is fresh
        need_training = True
        if _ml_predictor.load_model():
            # Check age
            import time
            try:
                mod_time = os.path.getmtime(_ml_predictor.model_path)
                age_hours = (time.time() - mod_time) / 3600
                if age_hours < 24:
                    need_training = False
                    print(f"ML Model is fresh ({age_hours:.1f} hours old).")
                else:
                    print(f"ML Model is stale ({age_hours:.1f} hours old). Retraining...")
            except Exception:
                pass
                
        if need_training:
            # Model doesn't exist or is old - try auto-training
            auto_train_model_if_needed(_ml_predictor)
    
    return _ml_predictor

