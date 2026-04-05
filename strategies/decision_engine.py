
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
    
    def analyze(self, symbol: str, closes: List[float], volumes: List[float], highs: List[float] = None, lows: List[float] = None) -> TradeSignal:
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
        atr = self.calculate_atr(closes, highs, lows)
        
        # 4. News Check
        news_score = 1.0
        if self.news_sentinel:
            news_score = self.news_sentinel.check_sentiment(symbol)
        
        # 3. Technical Analysis (Classic Indicators)
        tech_action, tech_score, tech_reasons = self._analyze_technicals(closes, volumes)
        
        # --- SYNTHESIS LOGIC (Multi-Indicator Confirmation) ---
        
        reasoning = []
        final_action = "hold"
        
        # Confirmation Logic:
        # 1. Technical signal (EMA crossover or RSI extreme)
        # 2. ML Prediction (Direction)
        # 3. RSI Alignment (Trend)
        
        ml_aligns = (ml_pred == 1 and tech_action == "buy") or (ml_pred == 0 and tech_action == "sell")
        
        # Confidence balancing
        confidence = tech_score
        
        if ml_aligns:
            confidence += (ml_conf * 0.25)
            reasoning.append(f"ML confirms technical signal ({ml_conf:.2f} conf)")
        else:
            confidence -= (ml_conf * 0.20)
            reasoning.append(f"ML disagrees with technicals - reducing confidence")

        # RSI Trend Alignment
        rsi = self._calculate_rsi(closes)
        if tech_action == "buy" and rsi < 60: # Not overbought yet
            confidence += 0.1
        elif tech_action == "sell" and rsi > 40: # Not oversold yet
            confidence += 0.1

        # Modify based on Regime
        params = get_regime_parameters(regime)
        if regime in ["strong_uptrend", "weak_uptrend"]:
            if tech_action == "buy":
                confidence *= 1.2
                reasoning.append(f"Regime {regime} supports bullish trend")
            elif tech_action == "sell":
                confidence *= 0.6
                reasoning.append(f"Regime {regime} opposes bearish signal")
        elif regime == "high_volatility":
            confidence *= 0.7
            reasoning.append("High volatility - reducing conviction")

        # News Guard
        if news_score < 0.4:
            confidence = 0.0
            reasoning.append(f"CRITICAL NEWS DANGER: Blocked by News Sentinel")
        
        # Final Decision
        min_conf = self.config.min_confidence_to_trade
        confidence = max(0.0, min(1.0, confidence))
        
        # Confirmation Requirement: 
        # For a trade to be valid, we now require confidence > threshold 
        # and ML to at least NOT disagree strongly (or confirm if we want stricter)
        
        if tech_action == "buy" and confidence > min_conf:
            if ml_pred == 1 or not self.config.enable_ml_prediction:
                final_action = "buy"
            else:
                final_action = "hold"
                reasoning.append("Filtered: ML predicts DOWN despite technical BUY")
        elif tech_action == "sell" and confidence > min_conf:
            if ml_pred == 0 or not self.config.enable_ml_prediction:
                final_action = "sell"
            else:
                final_action = "hold"
                reasoning.append("Filtered: ML predicts UP despite technical SELL")
        else:
            final_action = "hold"

        # Add tech reasons
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
        Classic Technical Analysis using EMA and RSI.
        Returns: (Action, Score 0-1, Reasons)
        """
        short_w = self.config.short_window
        long_w = self.config.long_window
        
        ema_short = self._calculate_ema(closes, short_w)
        ema_long = self._calculate_ema(closes, long_w)
        
        rsi = self._calculate_rsi(closes)
        
        action = "hold"
        score = 0.5
        reasons = []
        
        # 1. EMA Crossover
        if ema_short > ema_long * 1.0005:
            action = "buy"
            score = 0.65
            reasons.append(f"EMA Crossover Bullish ({short_w}/{long_w})")
        elif ema_short < ema_long * 0.9995:
            action = "sell"
            score = 0.65
            reasons.append(f"EMA Crossover Bearish ({short_w}/{long_w})")
            
        # 2. RSI Confirmation (Trend alignment)
        if action == "buy":
            if rsi > self.config.rsi_overbought:
                action = "hold"
                reasons.append(f"RSI Overbought ({rsi:.1f}) - entry blocked")
            elif rsi < 50:
                score -= 0.1 # Weak upward momentum
        elif action == "sell":
            if rsi < self.config.rsi_oversold:
                action = "hold"
                reasons.append(f"RSI Oversold ({rsi:.1f}) - exit blocked")
            elif rsi > 50:
                score -= 0.1 # Weak downward momentum

        # 3. Volume Confirmation
        if len(volumes) > 20:
             avg_vol = np.mean(volumes[-20:])
             cur_vol = volumes[-1]
             if cur_vol > avg_vol * self.config.volume_confirmation_threshold:
                 score += 0.1
                 reasons.append(f"Volume confirmed ({cur_vol/avg_vol:.1f}x)")
             else:
                 score -= 0.05
                 reasons.append("Volume low - weak confirmation")
        
        return action, score, reasons

    def _calculate_ema(self, data: List[float], window: int) -> float:
        """Calculate Exponential Moving Average using Pandas for robustness."""
        if not data or len(data) < window:
            return sum(data) / len(data) if data else 0.0
        
        series = pd.Series(data)
        ema = series.ewm(span=window, adjust=False).mean()
        return float(ema.iloc[-1])

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

    def calculate_atr(self, closes: List[float], highs: List[float] = None, lows: List[float] = None, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR).
        Ensures proper True Range calculation using H/L/C.
        """
        if len(closes) < period + 1:
            return 0.0
            
        if highs is not None and lows is not None and len(highs) >= len(closes):
            # Standard ATR: TR = max(H-L, |H-Cp|, |L-Cp|)
            trs = []
            for i in range(1, len(closes)):
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i-1])
                tr3 = abs(lows[i] - closes[i-1])
                trs.append(max(tr1, tr2, tr3))
            
            # Use Wilder's smoothing or simple MA? Code originally used np.mean.
            # Wilders is standard for ATR: ATR = (PrevATR * (n-1) + TR) / n
            # For simplicity and consistency with existing code, we stick to mean but ensure TR is correct.
            atr = np.mean(trs[-period:])
        else:
            # Better fallback than simple deltas: simulate H/L with a 0.5% buffer
            trs = []
            for i in range(1, len(closes)):
                tr = abs(closes[i] - closes[i-1])
                # Add a small volatility proxy since we lack H/L
                tr = max(tr, closes[i] * 0.005) 
                trs.append(tr)
            atr = np.mean(trs[-period:])
            
        return float(atr)
