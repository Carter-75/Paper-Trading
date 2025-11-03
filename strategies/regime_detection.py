#!/usr/bin/env python3
"""
Market Regime Detection Module

Identifies market states (trending up, trending down, sideways, high volatility)
using ADX, volatility, and moving average slope analysis.

Market regimes help adapt strategy parameters:
- Strong trend: Use trend-following with wide stops
- Sideways: Use mean-reversion with tight stops
- High volatility: Reduce position size, widen stops
"""

from typing import List, Tuple, Optional, Dict
import statistics


def calculate_adx(closes: List[float], period: int = 14) -> float:
    """
    Calculate Average Directional Index (ADX) to measure trend strength.
    
    ADX values:
    - 0-25: Weak trend (sideways/ranging market)
    - 25-50: Strong trend
    - 50-75: Very strong trend
    - 75-100: Extremely strong trend
    
    Args:
        closes: List of closing prices
        period: ADX calculation period (default 14)
    
    Returns:
        ADX value (0-100)
    """
    if len(closes) < period + 1:
        return 25.0  # Neutral default
    
    # Calculate True Range (TR)
    trs = []
    for i in range(1, len(closes)):
        high = max(closes[i], closes[i-1])
        low = min(closes[i], closes[i-1])
        tr = high - low
        trs.append(tr)
    
    if len(trs) < period:
        return 25.0
    
    # Calculate +DM and -DM
    plus_dm = []
    minus_dm = []
    for i in range(1, len(closes)):
        up_move = closes[i] - closes[i-1]
        down_move = closes[i-1] - closes[i]
        
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
    
    # Smooth using simple moving average
    def smooth(values: List[float], period: int) -> List[float]:
        result = []
        for i in range(period - 1, len(values)):
            result.append(sum(values[i-period+1:i+1]) / period)
        return result
    
    smoothed_tr = smooth(trs, period)
    smoothed_plus_dm = smooth(plus_dm, period)
    smoothed_minus_dm = smooth(minus_dm, period)
    
    if not smoothed_tr or smoothed_tr[-1] == 0:
        return 25.0
    
    # Calculate +DI and -DI
    plus_di = (smoothed_plus_dm[-1] / smoothed_tr[-1]) * 100
    minus_di = (smoothed_minus_dm[-1] / smoothed_tr[-1]) * 100
    
    # Calculate DX
    di_sum = plus_di + minus_di
    if di_sum == 0:
        return 25.0
    
    dx = abs(plus_di - minus_di) / di_sum * 100
    
    # ADX is smoothed DX
    return min(100.0, dx)


def calculate_ma_slope(closes: List[float], window: int = 20) -> float:
    """
    Calculate moving average slope to determine trend direction.
    
    Returns:
        Positive value: Uptrend
        Negative value: Downtrend
        Near zero: Sideways
    """
    if len(closes) < window + 5:
        return 0.0
    
    # Calculate MA values at two points
    ma_now = sum(closes[-window:]) / window
    ma_prev = sum(closes[-window-5:-5]) / window
    
    # Calculate slope (percentage change)
    if ma_prev == 0:
        return 0.0
    
    slope = (ma_now - ma_prev) / ma_prev * 100
    return slope


def calculate_volatility_percentile(closes: List[float], window: int = 30, lookback: int = 100) -> float:
    """
    Calculate current volatility as a percentile of historical volatility.
    
    Returns:
        0.0-1.0 where:
        - 0.0-0.3: Low volatility
        - 0.3-0.7: Normal volatility
        - 0.7-1.0: High volatility
    """
    if len(closes) < lookback + window:
        return 0.5  # Neutral default
    
    # Calculate current volatility
    recent = closes[-window:]
    if len(recent) < 2:
        return 0.5
    
    current_vol = statistics.stdev(recent) / statistics.mean(recent) if statistics.mean(recent) > 0 else 0.0
    
    # Calculate historical volatilities
    historical_vols = []
    for i in range(lookback, window, -5):  # Sample every 5 bars
        if i <= len(closes):
            period_data = closes[-i:-i+window] if i > window else closes[-window:]
            if len(period_data) >= 2:
                vol = statistics.stdev(period_data) / statistics.mean(period_data) if statistics.mean(period_data) > 0 else 0.0
                historical_vols.append(vol)
    
    if not historical_vols:
        return 0.5
    
    # Calculate percentile
    sorted_vols = sorted(historical_vols)
    rank = sum(1 for v in sorted_vols if v <= current_vol)
    percentile = rank / len(sorted_vols)
    
    return percentile


def detect_market_regime(closes: List[float], adx_threshold: float = 25.0) -> Dict[str, any]:
    """
    Detect current market regime using multiple indicators.
    
    Args:
        closes: List of closing prices
        adx_threshold: ADX value to distinguish trending from sideways (default 25)
    
    Returns:
        Dictionary with:
            - regime: "strong_uptrend", "weak_uptrend", "sideways", "weak_downtrend", "strong_downtrend", "high_volatility"
            - adx: ADX value
            - slope: MA slope value
            - volatility_percentile: Current volatility percentile
            - confidence: Confidence in regime detection (0-1)
    """
    if len(closes) < 50:
        return {
            "regime": "unknown",
            "adx": 0.0,
            "slope": 0.0,
            "volatility_percentile": 0.5,
            "confidence": 0.0,
        }
    
    # Calculate indicators
    adx = calculate_adx(closes, period=14)
    slope = calculate_ma_slope(closes, window=20)
    vol_percentile = calculate_volatility_percentile(closes, window=30, lookback=100)
    
    # Determine regime
    is_trending = adx > adx_threshold
    is_high_vol = vol_percentile > 0.75
    
    # Regime classification
    if is_high_vol:
        regime = "high_volatility"
        confidence = vol_percentile
    elif is_trending:
        if slope > 0.5:
            regime = "strong_uptrend"
            confidence = min(1.0, adx / 50.0)
        elif slope > 0.1:
            regime = "weak_uptrend"
            confidence = min(1.0, adx / 50.0) * 0.7
        elif slope < -0.5:
            regime = "strong_downtrend"
            confidence = min(1.0, adx / 50.0)
        elif slope < -0.1:
            regime = "weak_downtrend"
            confidence = min(1.0, adx / 50.0) * 0.7
        else:
            regime = "sideways"
            confidence = 0.6
    else:
        regime = "sideways"
        confidence = 0.7
    
    return {
        "regime": regime,
        "adx": adx,
        "slope": slope,
        "volatility_percentile": vol_percentile,
        "confidence": confidence,
    }


def get_regime_parameters(regime: str) -> Dict[str, float]:
    """
    Get recommended strategy parameters for each market regime.
    
    Args:
        regime: Market regime string
    
    Returns:
        Dictionary with recommended parameter multipliers:
            - position_size_mult: Multiplier for position size
            - stop_loss_mult: Multiplier for stop loss
            - take_profit_mult: Multiplier for take profit
    """
    parameters = {
        "strong_uptrend": {
            "position_size_mult": 1.2,  # Larger positions in strong trends
            "stop_loss_mult": 1.3,      # Wider stops to avoid noise
            "take_profit_mult": 1.5,    # Higher targets in trends
        },
        "weak_uptrend": {
            "position_size_mult": 1.0,
            "stop_loss_mult": 1.1,
            "take_profit_mult": 1.2,
        },
        "strong_downtrend": {
            "position_size_mult": 0.5,  # Reduce exposure in downtrends (long-only)
            "stop_loss_mult": 1.3,
            "take_profit_mult": 0.8,    # Lower targets, exit quickly
        },
        "weak_downtrend": {
            "position_size_mult": 0.7,
            "stop_loss_mult": 1.1,
            "take_profit_mult": 0.9,
        },
        "sideways": {
            "position_size_mult": 0.8,  # Smaller positions in ranging markets
            "stop_loss_mult": 0.9,      # Tighter stops for mean reversion
            "take_profit_mult": 0.9,    # Realistic targets
        },
        "high_volatility": {
            "position_size_mult": 0.6,  # Reduce risk in volatile markets
            "stop_loss_mult": 1.5,      # Much wider stops
            "take_profit_mult": 1.3,    # Can capture larger moves
        },
        "unknown": {
            "position_size_mult": 1.0,
            "stop_loss_mult": 1.0,
            "take_profit_mult": 1.0,
        },
    }
    
    return parameters.get(regime, parameters["unknown"])


if __name__ == "__main__":
    # Test regime detection
    print("Testing market regime detection...\n")
    
    # Test 1: Strong uptrend
    uptrend_data = [100 + i * 0.5 for i in range(100)]
    result = detect_market_regime(uptrend_data)
    print(f"Uptrend Data:")
    print(f"  Regime: {result['regime']}")
    print(f"  ADX: {result['adx']:.2f}")
    print(f"  Slope: {result['slope']:.2f}%")
    print(f"  Confidence: {result['confidence']:.2f}")
    params = get_regime_parameters(result['regime'])
    print(f"  Recommended: Position {params['position_size_mult']}x, SL {params['stop_loss_mult']}x, TP {params['take_profit_mult']}x\n")
    
    # Test 2: Sideways
    sideways_data = [100 + (i % 10) * 2 for i in range(100)]
    result = detect_market_regime(sideways_data)
    print(f"Sideways Data:")
    print(f"  Regime: {result['regime']}")
    print(f"  ADX: {result['adx']:.2f}")
    print(f"  Slope: {result['slope']:.2f}%")
    print(f"  Confidence: {result['confidence']:.2f}")
    params = get_regime_parameters(result['regime'])
    print(f"  Recommended: Position {params['position_size_mult']}x, SL {params['stop_loss_mult']}x, TP {params['take_profit_mult']}x\n")
    
    # Test 3: High volatility
    import random
    volatile_data = [100 + random.uniform(-10, 10) for _ in range(100)]
    result = detect_market_regime(volatile_data)
    print(f"Volatile Data:")
    print(f"  Regime: {result['regime']}")
    print(f"  ADX: {result['adx']:.2f}")
    print(f"  Vol Percentile: {result['volatility_percentile']:.2f}")
    print(f"  Confidence: {result['confidence']:.2f}")
    params = get_regime_parameters(result['regime'])
    print(f"  Recommended: Position {params['position_size_mult']}x, SL {params['stop_loss_mult']}x, TP {params['take_profit_mult']}x")

