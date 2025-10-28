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


# Global instance
_ml_predictor = None

def get_ml_predictor() -> TradingMLPredictor:
    """Get global ML predictor instance"""
    global _ml_predictor
    if _ml_predictor is None:
        _ml_predictor = TradingMLPredictor()
        _ml_predictor.load_model()  # Try to load existing model
    return _ml_predictor

