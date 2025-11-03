#!/usr/bin/env python3
"""
Black Swan Stress Testing Module

Simulates strategy performance during historical market crises:
- 2008 Financial Crisis (Sep 2008 - Mar 2009)
- 2020 COVID Crash (Feb 2020 - Mar 2020)
- 2022 Bear Market (Jan 2022 - Oct 2022)

Usage:
    python stress_test.py --symbol SPY
    python stress_test.py --symbols SPY QQQ AAPL --interval 0.25
"""

import sys
import argparse
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import config
from runner import (
    make_client,
    fetch_closes,
    simulate_signals_and_projection,
)

# Historical crisis periods (start_date, end_date, name, description)
CRISIS_PERIODS = [
    {
        "name": "2008 Financial Crisis",
        "start": "2008-09-01",
        "end": "2009-03-31",
        "description": "Lehman Brothers collapse, S&P 500 dropped 56.8% from peak",
        "severity": "EXTREME"
    },
    {
        "name": "2020 COVID Crash",
        "start": "2020-02-01",
        "end": "2020-04-30",
        "description": "Pandemic panic, S&P 500 dropped 33.9% in 33 days",
        "severity": "SEVERE"
    },
    {
        "name": "2022 Bear Market",
        "start": "2022-01-01",
        "end": "2022-10-31",
        "description": "Fed rate hikes, S&P 500 dropped 25.4%",
        "severity": "MODERATE"
    },
    {
        "name": "2011 Debt Crisis",
        "start": "2011-07-01",
        "end": "2011-10-31",
        "description": "US debt downgrade, European debt crisis",
        "severity": "MODERATE"
    },
    {
        "name": "2018 Q4 Selloff",
        "start": "2018-10-01",
        "end": "2018-12-31",
        "description": "Fed policy fears, S&P 500 dropped 19.8%",
        "severity": "MODERATE"
    }
]


def fetch_crisis_data(symbol: str, start_date: str, end_date: str, interval_seconds: int) -> List[float]:
    """
    Fetch historical price data for a specific crisis period.
    Uses yfinance since we need historical data from years ago.
    """
    try:
        import yfinance as yf
        from datetime import datetime
        
        ticker = yf.Ticker(symbol)
        
        # Convert interval_seconds to yfinance period
        if interval_seconds <= 300:
            period_str = "1m"
        elif interval_seconds <= 900:
            period_str = "5m"
        elif interval_seconds <= 3600:
            period_str = "1h"
        elif interval_seconds <= 86400:
            period_str = "1d"
        else:
            period_str = "1d"
        
        # Fetch data for the crisis period
        data = ticker.history(start=start_date, end=end_date, interval=period_str)
        
        if data.empty:
            return []
        
        closes = data['Close'].tolist()
        return closes
    
    except Exception as e:
        print(f"Error fetching crisis data for {symbol}: {e}")
        return []


def run_stress_test_on_crisis(symbol: str, crisis: Dict, interval_seconds: int, 
                               capital: float, verbose: bool = False) -> Dict:
    """
    Run strategy simulation on a specific crisis period.
    Returns dict with performance metrics.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Testing: {crisis['name']}")
        print(f"Period: {crisis['start']} to {crisis['end']}")
        print(f"Description: {crisis['description']}")
        print(f"{'='*70}")
    
    # Fetch data for crisis period
    closes = fetch_crisis_data(symbol, crisis['start'], crisis['end'], interval_seconds)
    
    if not closes or len(closes) < config.LONG_WINDOW + 10:
        if verbose:
            print(f"[X] Insufficient data: {len(closes) if closes else 0} bars")
        return {
            "symbol": symbol,
            "crisis": crisis['name'],
            "success": False,
            "reason": "Insufficient data"
        }
    
    if verbose:
        print(f"[OK] Fetched {len(closes)} bars")
    
    # Calculate buy-and-hold performance for comparison
    buy_hold_return = ((closes[-1] - closes[0]) / closes[0]) * 100
    
    # Run strategy simulation
    sim_result = simulate_signals_and_projection(
        closes, 
        interval_seconds,
        override_cap_usd=capital
    )
    
    # Extract key metrics
    strategy_return = sim_result.get("total_return_pct", 0.0)
    expected_daily = sim_result.get("expected_daily_usd", 0.0)
    max_drawdown = sim_result.get("max_drawdown_pct", 0.0)
    num_trades = sim_result.get("num_trades", 0)
    win_rate = sim_result.get("win_rate", 0.0)
    sharpe = sim_result.get("sharpe_ratio", 0.0)
    
    # Calculate alpha (outperformance vs buy-and-hold)
    alpha = strategy_return - buy_hold_return
    
    # Determine survival
    survived = max_drawdown < 50.0  # Didn't lose more than 50%
    
    if verbose:
        print(f"\nResults:")
        print(f"  Strategy Return: {strategy_return:+.2f}%")
        print(f"  Buy & Hold Return: {buy_hold_return:+.2f}%")
        print(f"  Alpha (outperformance): {alpha:+.2f}%")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")
        print(f"  Trades: {num_trades}")
        print(f"  Win Rate: {win_rate*100:.1f}%")
        print(f"  Sharpe Ratio: {sharpe:.3f}")
        
        if survived:
            print(f"\n[OK] Strategy SURVIVED this crisis")
        else:
            print(f"\n[X] Strategy FAILED this crisis (>50% drawdown)")
    
    return {
        "symbol": symbol,
        "crisis": crisis['name'],
        "severity": crisis['severity'],
        "success": True,
        "survived": survived,
        "strategy_return": strategy_return,
        "buy_hold_return": buy_hold_return,
        "alpha": alpha,
        "max_drawdown": max_drawdown,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "bars_tested": len(closes)
    }


def generate_stress_test_report(results: List[Dict], symbol: str, interval_hours: float, capital: float):
    """
    Generate a comprehensive stress test report.
    """
    print(f"\n{'='*70}")
    print(f"BLACK SWAN STRESS TEST REPORT")
    print(f"{'='*70}")
    print(f"Symbol: {symbol}")
    print(f"Interval: {interval_hours}h ({int(interval_hours*3600)}s)")
    print(f"Capital: ${capital:,.2f}")
    print(f"Crises Tested: {len(results)}")
    print(f"{'='*70}\n")
    
    # Filter successful tests
    successful_tests = [r for r in results if r.get("success", False)]
    
    if not successful_tests:
        print("[X] No successful tests - all crisis periods had insufficient data")
        return
    
    # Summary statistics
    survived_count = sum(1 for r in successful_tests if r.get("survived", False))
    avg_strategy_return = sum(r["strategy_return"] for r in successful_tests) / len(successful_tests)
    avg_buy_hold = sum(r["buy_hold_return"] for r in successful_tests) / len(successful_tests)
    avg_alpha = sum(r["alpha"] for r in successful_tests) / len(successful_tests)
    avg_drawdown = sum(r["max_drawdown"] for r in successful_tests) / len(successful_tests)
    worst_drawdown = max(r["max_drawdown"] for r in successful_tests)
    
    # Overall assessment
    survival_rate = (survived_count / len(successful_tests)) * 100
    
    print("OVERALL ASSESSMENT:")
    print(f"  Survival Rate: {survived_count}/{len(successful_tests)} ({survival_rate:.0f}%)")
    print(f"  Average Strategy Return: {avg_strategy_return:+.2f}%")
    print(f"  Average Buy & Hold: {avg_buy_hold:+.2f}%")
    print(f"  Average Alpha: {avg_alpha:+.2f}%")
    print(f"  Average Max Drawdown: {avg_drawdown:.2f}%")
    print(f"  Worst Drawdown: {worst_drawdown:.2f}%")
    
    # Verdict
    print(f"\n{'='*70}")
    if survival_rate == 100 and avg_alpha > 0:
        print("✅ EXCELLENT - Strategy survived ALL crises AND outperformed buy-and-hold")
    elif survival_rate == 100:
        print("✅ GOOD - Strategy survived ALL crises")
    elif survival_rate >= 75:
        print("⚠️  ACCEPTABLE - Strategy survived most crises")
    elif survival_rate >= 50:
        print("⚠️  RISKY - Strategy failed multiple crises")
    else:
        print("❌ DANGEROUS - Strategy failed most crises")
    print(f"{'='*70}\n")
    
    # Detailed results per crisis
    print("DETAILED RESULTS BY CRISIS:")
    print(f"{'='*70}")
    
    for r in successful_tests:
        status = "✅ SURVIVED" if r["survived"] else "❌ FAILED"
        print(f"\n{r['crisis']} ({r['severity']})")
        print(f"  Status: {status}")
        print(f"  Strategy: {r['strategy_return']:+.2f}% | Buy&Hold: {r['buy_hold_return']:+.2f}% | Alpha: {r['alpha']:+.2f}%")
        print(f"  Max DD: {r['max_drawdown']:.2f}% | Trades: {r['num_trades']} | Win Rate: {r['win_rate']*100:.1f}%")
    
    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Black Swan Stress Testing")
    parser.add_argument("-s", "--symbol", type=str, default="SPY",
                       help="Stock symbol to test (default: SPY)")
    parser.add_argument("--symbols", nargs="+", type=str,
                       help="Multiple symbols to test")
    parser.add_argument("-t", "--interval", type=float, default=0.25,
                       help="Trading interval in hours (default: 0.25 = 15min)")
    parser.add_argument("-m", "--capital", type=float, default=10000.0,
                       help="Starting capital (default: $10,000)")
    parser.add_argument("--crisis", type=str,
                       help="Test specific crisis only (e.g., '2008', '2020', '2022')")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Show detailed output")
    
    args = parser.parse_args()
    
    # Determine symbols to test
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = [args.symbol]
    
    interval_seconds = int(args.interval * 3600)
    
    # Filter crises if specific one requested
    if args.crisis:
        crises = [c for c in CRISIS_PERIODS if args.crisis in c['name']]
        if not crises:
            print(f"[X] No crisis matching '{args.crisis}' found")
            print(f"Available crises: {', '.join([c['name'] for c in CRISIS_PERIODS])}")
            return 1
    else:
        crises = CRISIS_PERIODS
    
    print(f"\n{'='*70}")
    print(f"BLACK SWAN STRESS TESTING")
    print(f"{'='*70}")
    print(f"Testing {len(symbols)} symbol(s) across {len(crises)} crisis period(s)")
    print(f"This may take a few minutes...\n")
    
    # Run tests for each symbol
    for symbol in symbols:
        results = []
        
        for crisis in crises:
            result = run_stress_test_on_crisis(
                symbol, crisis, interval_seconds, args.capital, verbose=args.verbose
            )
            results.append(result)
        
        # Generate report for this symbol
        generate_stress_test_report(results, symbol, args.interval, args.capital)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

