#!/usr/bin/env python3
"""
Parameter Stability Analysis

Analyzes optimization_history.csv to detect:
- Parameter drift over time (intervals, capitals changing)
- Overfitting signals (parameters unstable)
- Performance degradation
- Statistical stability metrics

Usage:
    python parameter_stability.py
    python parameter_stability.py --window 30  # Last 30 days
    python parameter_stability.py --symbol AAPL
"""

import sys
import argparse
import os
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import statistics


def load_optimization_history(filepath: str = "optimization_history.csv") -> List[Dict]:
    """
    Load optimization history from CSV file.
    Returns list of dicts with optimization runs.
    """
    if not os.path.exists(filepath):
        print(f"[X] File not found: {filepath}")
        print(f"    Run optimizer first: python optimizer.py")
        return []
    
    results = []
    
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse and convert types
                try:
                    results.append({
                        'date': row['Date'],
                        'time': row['Time'],
                        'symbol': row['Symbol'],
                        'interval_sec': int(row['Interval_Sec']),
                        'interval_hours': float(row['Interval_Hours']),
                        'capital': float(row['Capital']),
                        'daily_return': float(row['Daily_Return']),
                        'consistency': float(row['Consistency']),
                        'sharpe': float(row['Sharpe']),
                        'sortino': float(row['Sortino']),
                        'win_rate': float(row['Win_Rate']),
                        'max_drawdown': float(row['Max_Drawdown']),
                        'var_95': float(row['VaR_95']),
                        'cvar_95': float(row['CVaR_95']),
                        'datetime': datetime.strptime(f"{row['Date']} {row['Time']}", "%Y-%m-%d %H:%M:%S")
                    })
                except (ValueError, KeyError) as e:
                    # Skip malformed rows
                    continue
        
        return results
    
    except Exception as e:
        print(f"[X] Error loading optimization history: {e}")
        return []


def filter_by_window(data: List[Dict], days: int = None) -> List[Dict]:
    """Filter data to last N days."""
    if days is None or not data:
        return data
    
    cutoff = datetime.now() - timedelta(days=days)
    return [r for r in data if r['datetime'] >= cutoff]


def filter_by_symbol(data: List[Dict], symbol: str = None) -> List[Dict]:
    """Filter data by symbol."""
    if symbol is None or not data:
        return data
    
    return [r for r in data if r['symbol'].upper() == symbol.upper()]


def analyze_parameter_stability(data: List[Dict], param: str) -> Dict:
    """
    Analyze stability of a parameter (interval_sec, capital, etc).
    Returns stability metrics.
    """
    if not data:
        return {"stable": False, "reason": "No data"}
    
    values = [r[param] for r in data]
    
    if len(values) < 2:
        return {"stable": True, "reason": "Insufficient data for analysis"}
    
    mean_val = statistics.mean(values)
    stdev_val = statistics.stdev(values) if len(values) > 1 else 0
    median_val = statistics.median(values)
    
    # Coefficient of variation (CV) - measures relative variability
    cv = (stdev_val / mean_val) * 100 if mean_val != 0 else 0
    
    # Determine stability based on CV
    if cv < 10:
        stability = "VERY STABLE"
        stable = True
    elif cv < 25:
        stability = "STABLE"
        stable = True
    elif cv < 50:
        stability = "MODERATE"
        stable = True
    else:
        stability = "UNSTABLE"
        stable = False
    
    # Detect drift - compare first half vs second half
    midpoint = len(values) // 2
    first_half = values[:midpoint]
    second_half = values[midpoint:]
    
    if len(first_half) > 0 and len(second_half) > 0:
        first_mean = statistics.mean(first_half)
        second_mean = statistics.mean(second_half)
        drift_pct = ((second_mean - first_mean) / first_mean * 100) if first_mean != 0 else 0
        has_drift = abs(drift_pct) > 20  # >20% change
    else:
        drift_pct = 0
        has_drift = False
    
    return {
        "stable": stable,
        "stability": stability,
        "mean": mean_val,
        "median": median_val,
        "stdev": stdev_val,
        "cv": cv,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "drift_pct": drift_pct,
        "has_drift": has_drift,
        "num_samples": len(values)
    }


def detect_overfitting_signals(data: List[Dict]) -> Dict:
    """
    Detect potential overfitting by analyzing:
    1. Performance degradation over time
    2. High variance in parameters
    3. Inconsistent performance metrics
    """
    if len(data) < 5:
        return {"overfitting_risk": "UNKNOWN", "signals": ["Insufficient data"]}
    
    signals = []
    risk_score = 0
    
    # 1. Check parameter stability
    interval_stability = analyze_parameter_stability(data, 'interval_sec')
    capital_stability = analyze_parameter_stability(data, 'capital')
    
    if not interval_stability['stable']:
        signals.append(f"Interval parameters UNSTABLE (CV={interval_stability['cv']:.1f}%)")
        risk_score += 2
    
    if not capital_stability['stable']:
        signals.append(f"Capital parameters UNSTABLE (CV={capital_stability['cv']:.1f}%)")
        risk_score += 2
    
    if interval_stability['has_drift']:
        signals.append(f"Interval DRIFTING ({interval_stability['drift_pct']:+.1f}% change)")
        risk_score += 1
    
    if capital_stability['has_drift']:
        signals.append(f"Capital DRIFTING ({capital_stability['drift_pct']:+.1f}% change)")
        risk_score += 1
    
    # 2. Check performance consistency
    consistency_values = [r['consistency'] for r in data]
    avg_consistency = statistics.mean(consistency_values)
    
    if avg_consistency < 0.5:
        signals.append(f"Low average consistency ({avg_consistency:.2f})")
        risk_score += 2
    
    # 3. Check if returns are degrading
    returns = [r['daily_return'] for r in data]
    midpoint = len(returns) // 2
    if len(returns) >= 10:
        first_half_return = statistics.mean(returns[:midpoint])
        second_half_return = statistics.mean(returns[midpoint:])
        
        if second_half_return < first_half_return * 0.7:  # >30% drop
            degradation = ((second_half_return - first_half_return) / first_half_return * 100)
            signals.append(f"Performance DEGRADING ({degradation:+.1f}% recent vs old)")
            risk_score += 3
    
    # 4. Check Sharpe ratio stability
    sharpe_values = [r['sharpe'] for r in data]
    if sharpe_values:
        sharpe_stdev = statistics.stdev(sharpe_values) if len(sharpe_values) > 1 else 0
        sharpe_mean = statistics.mean(sharpe_values)
        sharpe_cv = (sharpe_stdev / abs(sharpe_mean) * 100) if sharpe_mean != 0 else 0
        
        if sharpe_cv > 50:
            signals.append(f"Sharpe ratio highly variable (CV={sharpe_cv:.1f}%)")
            risk_score += 1
    
    # Determine overall risk
    if risk_score >= 6:
        risk = "HIGH"
    elif risk_score >= 3:
        risk = "MODERATE"
    else:
        risk = "LOW"
    
    if not signals:
        signals.append("No overfitting signals detected")
    
    return {
        "overfitting_risk": risk,
        "risk_score": risk_score,
        "signals": signals
    }


def generate_stability_report(data: List[Dict], symbol: str = None, window_days: int = None):
    """
    Generate comprehensive parameter stability report.
    """
    if not data:
        print("[X] No data to analyze")
        return
    
    print(f"\n{'='*70}")
    print(f"PARAMETER STABILITY ANALYSIS")
    print(f"{'='*70}")
    
    if symbol:
        print(f"Symbol: {symbol}")
    
    if window_days:
        print(f"Window: Last {window_days} days")
    
    print(f"Total Optimizations: {len(data)}")
    print(f"Date Range: {data[0]['date']} to {data[-1]['date']}")
    print(f"Symbols Analyzed: {', '.join(sorted(set(r['symbol'] for r in data)))}")
    print(f"{'='*70}\n")
    
    # 1. Interval Stability
    print("INTERVAL PARAMETER STABILITY:")
    print(f"{'='*70}")
    interval_stats = analyze_parameter_stability(data, 'interval_sec')
    
    print(f"  Status: {interval_stats['stability']}")
    print(f"  Mean: {interval_stats['mean']:.0f}s ({interval_stats['mean']/3600:.3f}h)")
    print(f"  Median: {interval_stats['median']:.0f}s ({interval_stats['median']/3600:.3f}h)")
    print(f"  StdDev: {interval_stats['stdev']:.0f}s")
    print(f"  Range: {interval_stats['min']:.0f}s - {interval_stats['max']:.0f}s")
    print(f"  Coefficient of Variation: {interval_stats['cv']:.1f}%")
    
    if interval_stats['has_drift']:
        print(f"  ⚠️  DRIFT DETECTED: {interval_stats['drift_pct']:+.1f}% change over time")
    else:
        print(f"  ✅ No significant drift")
    
    # 2. Capital Stability
    print(f"\nCAPITAL PARAMETER STABILITY:")
    print(f"{'='*70}")
    capital_stats = analyze_parameter_stability(data, 'capital')
    
    print(f"  Status: {capital_stats['stability']}")
    print(f"  Mean: ${capital_stats['mean']:,.2f}")
    print(f"  Median: ${capital_stats['median']:,.2f}")
    print(f"  StdDev: ${capital_stats['stdev']:,.2f}")
    print(f"  Range: ${capital_stats['min']:,.2f} - ${capital_stats['max']:,.2f}")
    print(f"  Coefficient of Variation: {capital_stats['cv']:.1f}%")
    
    if capital_stats['has_drift']:
        print(f"  ⚠️  DRIFT DETECTED: {capital_stats['drift_pct']:+.1f}% change over time")
    else:
        print(f"  ✅ No significant drift")
    
    # 3. Performance Metrics
    print(f"\nPERFORMANCE METRICS:")
    print(f"{'='*70}")
    
    returns = [r['daily_return'] for r in data]
    sharpes = [r['sharpe'] for r in data]
    consistencies = [r['consistency'] for r in data]
    win_rates = [r['win_rate'] for r in data]
    
    print(f"  Average Daily Return: ${statistics.mean(returns):.2f}")
    print(f"  Average Sharpe Ratio: {statistics.mean(sharpes):.3f}")
    print(f"  Average Consistency: {statistics.mean(consistencies):.3f}")
    print(f"  Average Win Rate: {statistics.mean(win_rates)*100:.1f}%")
    
    # 4. Overfitting Detection
    print(f"\nOVERFITTING ANALYSIS:")
    print(f"{'='*70}")
    
    overfitting = detect_overfitting_signals(data)
    
    print(f"  Risk Level: {overfitting['overfitting_risk']}")
    print(f"  Risk Score: {overfitting['risk_score']}/10")
    print(f"\n  Signals:")
    for signal in overfitting['signals']:
        if "UNSTABLE" in signal or "DRIFTING" in signal or "DEGRADING" in signal:
            print(f"    ⚠️  {signal}")
        else:
            print(f"    ✅ {signal}")
    
    # 5. Overall Assessment
    print(f"\n{'='*70}")
    print(f"OVERALL ASSESSMENT:")
    print(f"{'='*70}")
    
    if overfitting['overfitting_risk'] == "LOW" and interval_stats['stable'] and capital_stats['stable']:
        print(f"✅ STABLE - Parameters are consistent and reliable")
        print(f"   Strategy appears robust and not overfit to recent data")
    elif overfitting['overfitting_risk'] == "MODERATE":
        print(f"⚠️  CAUTION - Some instability detected")
        print(f"   Monitor parameters and consider retraining if performance degrades")
    else:
        print(f"❌ UNSTABLE - High risk of overfitting")
        print(f"   Parameters are changing frequently - strategy may not be robust")
        print(f"   Consider:")
        print(f"     • Using longer backtest periods")
        print(f"     • Reducing parameter search space")
        print(f"     • Adding robustness constraints")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Parameter Stability Analysis")
    parser.add_argument("-s", "--symbol", type=str,
                       help="Analyze specific symbol only")
    parser.add_argument("-w", "--window", type=int,
                       help="Analyze last N days only")
    parser.add_argument("-f", "--file", type=str, default="optimization_history.csv",
                       help="Path to optimization history CSV (default: optimization_history.csv)")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading optimization history from {args.file}...")
    data = load_optimization_history(args.file)
    
    if not data:
        return 1
    
    print(f"Loaded {len(data)} optimization runs")
    
    # Apply filters
    if args.window:
        data = filter_by_window(data, args.window)
        print(f"Filtered to last {args.window} days: {len(data)} runs")
    
    if args.symbol:
        data = filter_by_symbol(data, args.symbol)
        print(f"Filtered to {args.symbol}: {len(data)} runs")
    
    if not data:
        print("[X] No data after filtering")
        return 1
    
    # Generate report
    generate_stability_report(data, args.symbol, args.window)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

