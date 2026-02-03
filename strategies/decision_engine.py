
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

# Import sub-modules
try:
    from strategies.regime_detection import detect_market_regime, get_regime_parameters
    from ml_predictor import get_ml_predictor
    from config_validated import get_config
except ImportError:
    # Fallback for direct testing
    from .regime_detection import detect_market_regime, get_regime_parameters
    # ML predictor might be in parent directory if running as module
    import sys
    sys.path.append("..")
    from ml_predictor import get_ml_predictor
    from config_validated import get_config

@dataclass
class TradeSignal:
    symbol: str
    action: str  # "buy", "sell", "hold"
    confidence: float  # 0.0 to 1.0
    reasoning: List[str]  # Human-readable reasons
    regime: str
    regime: str
    predicted_direction: int # 1 (up) or 0 (down)
    atr: float = 0.0
    news_score: float = 0.5 # 0.0 (bad) to 1.0 (good)

class DecisionEngine:
    """
    The Brain of the trading bot.
    Synthesizes inputs from:
    1. Technical Analysis (RSI, MACD, MA)
    2. Machine Learning (Random Forest)
    3. Market Regime (Volatility, Trend Strength)
    
    To produce a final 'reasoned' decision.
    """
    
    def __init__(self):
        self.config = get_config()
        self.ml_predictor = get_ml_predictor()
        self.logger = logging.getLogger("DecisionEngine")
        
        # Initialize News Sentinel
        try:
            from strategies.news_sentinel import NewsSentinel
            # We need an API client. We can get it from config or pass it in. 
            # For now, let's create a temporary one or expect it to be passed? 
            # Easier to lazy load or expect runner to pass it? 
            # Actually, DecisionEngine doesn't hold the API client usually.
            # Let's import the global make_client from runner_data_utils or similar if possible.
            # Or better: Just instantiate it if we can.
            from runner_data_utils import make_client # Assuming this exists or we can duplicate
            self.api = make_client()
            self.news_sentinel = NewsSentinel(self.api)
        except Exception as e:
            self.logger.warning(f"Could not initialize NewsSentinel: {e}")
            self.news_sentinel = None
    
    def analyze(self, symbol: str, closes: List[float], volumes: List[float]) -> TradeSignal:
        """
        Analyze a stock and return a reasoned decision.
        """
        if not closes or len(closes) < 50:
            return TradeSignal(symbol, "hold", 0.0, ["Insufficient data"], "unknown", 0)

        # 1. Market Regime Detection
        regime_data = detect_market_regime(closes)
        regime = regime_data['regime']
        regime_conf = regime_data['confidence']
        
        # 2. ML Prediction
        ml_pred, ml_conf = self.ml_predictor.predict(closes, volumes)
        
        # 3. ATR Calculation
        atr = self.calculate_atr(closes)
        
        # 4. News Check
        news_score = 1.0
        if self.news_sentinel:
            news_score = self.news_sentinel.check_sentiment(symbol)
        
        # 3. Technical Analysis (Classic Indicators)
        tech_action, tech_score, tech_reasons = self._analyze_technicals(closes, volumes)
        
        # --- SYNTHESIS LOGIC ---
        
        reasoning = []
        final_action = "hold"
        confidence = 0.0
        
        # Base confidence comes from technicals
        confidence = tech_score
        
        # Modify based on Regime (The "Context")
        params = get_regime_parameters(regime)
        if regime in ["strong_uptrend", "weak_uptrend"]:
            if tech_action == "buy":
                confidence *= 1.2
                reasoning.append(f"Regime is {regime} (Trend Following)")
            elif tech_action == "sell":
                confidence *= 0.8 # Don't fight the trend
                reasoning.append(f"Regime {regime} opposes sell signal")
        elif regime in ["sideways"]:
            # In sideways, standard trend following fails. Prefer mean reversion (RSI).
            if "RSI Overbought" in tech_reasons:
                 confidence *= 1.3
                 reasoning.append("Sideways market supports Mean Reversion Sell")
            elif "RSI Oversold" in tech_reasons:
                 confidence *= 1.3
                 reasoning.append("Sideways market supports Mean Reversion Buy")
        elif regime in ["high_volatility"]:
            confidence *= 0.7 # Be cautious
            reasoning.append("High volatility reduces confidence")

        # Modify based on ML (The "Prediction")
        if self.config.enable_ml_prediction:
            if ml_pred == 1: # Pred UP
                if tech_action == "buy":
                    confidence += (ml_conf * 0.2)
                    reasoning.append(f"ML confirms UP ({ml_conf:.2f} conf)")
                elif tech_action == "sell":
                    confidence -= (ml_conf * 0.2)
                    reasoning.append(f"ML disagrees (predicts UP {ml_conf:.2f})")
            else: # Pred DOWN
                if tech_action == "sell":
                     confidence += (ml_conf * 0.2)
                     reasoning.append(f"ML confirms DOWN ({ml_conf:.2f} conf)")
                elif tech_action == "buy":
                     confidence -= (ml_conf * 0.2)
                     confidence -= (ml_conf * 0.2)
                     reasoning.append(f"ML disagrees (predicts DOWN {ml_conf:.2f})")

        # Modify based on News
        if news_score < 0.5:
            confidence = 0.0
            reasoning.append(f"News Sentinel TRIGGERED (Score {news_score:.2f}) - BLOCKING TRADE")
        elif news_score < 0.9:
            confidence *= 0.8
            reasoning.append(f"News sentiment mixed ({news_score:.2f})")

        # Final Threshold Check
        min_conf = self.config.min_confidence_to_trade
        
        # Normalize confidence to 0-1 (soft clamp)
        confidence = max(0.0, min(1.0, confidence))
        
        if tech_action == "buy" and confidence > min_conf:
            final_action = "buy"
        elif tech_action == "sell" and confidence > min_conf:
            final_action = "sell"
        else:
            final_action = "hold"
            if tech_action != "hold":
                reasoning.append(f"Confidence {confidence:.2f} too low (thresh {min_conf})")

        # Merge specific reasons
        reasoning.extend(tech_reasons)
        
        return TradeSignal(
            symbol=symbol,
            action=final_action,
            confidence=confidence,
            reasoning=reasoning,
            regime=regime,
            predicted_direction=ml_pred,
            atr=atr,
            news_score=news_score
        )

    def _analyze_technicals(self, closes: List[float], volumes: List[float]) -> Tuple[str, float, List[str]]:
        """
        Classic Technical Analysis.
        Returns: (Action, Score 0-1, Reasons)
        """
        short_w = self.config.short_window
        long_w = self.config.long_window
        
        sma_short = np.mean(closes[-short_w:])
        sma_long = np.mean(closes[-long_w:])
        
        rsi = self._calculate_rsi(closes)
        
        action = "hold"
        score = 0.5
        reasons = []
        
        # 1. MA Crossover
        if sma_short > sma_long * 1.001:
            action = "buy"
            score = 0.6
            reasons.append(f"SMA Crossover (Bullish)")
        elif sma_short < sma_long * 0.999:
            action = "sell"
            score = 0.6
            reasons.append(f"SMA Crossover (Bearish)")
            
        # 2. RSI Filter
        if rsi > self.config.rsi_overbought:
            if action == "buy":
                action = "hold" # Don't buy top
                reasons.append(f"RSI Overbought ({rsi:.1f}) - suppressed buy")
            elif action == "hold":
                action = "sell" # Potential reversal
                score = 0.6
                reasons.append(f"RSI Overbought ({rsi:.1f}) - signal sell")
        elif rsi < self.config.rsi_oversold:
            if action == "sell":
                action = "hold" # Don't sell bottom
                reasons.append(f"RSI Oversold ({rsi:.1f}) - suppressed sell")
            elif action == "hold":
                action = "buy"
                score = 0.6
                reasons.append(f"RSI Oversold ({rsi:.1f}) - signal buy")

        # 3. Volume Confirmation
        if len(volumes) > 20:
             avg_vol = np.mean(volumes[-20:])
             cur_vol = volumes[-1]
             if cur_vol > avg_vol * self.config.volume_confirmation_threshold:
                 score += 0.1
                 reasons.append(f"High Volume ({cur_vol/avg_vol:.1f}x)")
        
        return action, score, reasons

    def _calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        
        deltas = np.diff(closes)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gain[-period:])
        avg_loss = np.mean(loss[-period:])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def calculate_atr(self, closes: List[float], period: int = 14) -> float:
        """
        Calculate Average True Range (ATR).
        Approximation using close-to-close volatility if High/Low not available.
        (Since we are passing only closes/volumes here).
         Ideally we'd have H/L.
        """
        if len(closes) < period + 1:
            return 0.0
            
        # True Range approx = |Close - PrevClose|
        # This is strictly just daily volatility if we don't have High/Low.
        # It's better than nothing.
        deltas = np.abs(np.diff(closes))
        atr = np.mean(deltas[-period:])
        return float(atr)
