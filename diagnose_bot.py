#!/usr/bin/env python3
"""
Bot Diagnostic Tool - Analyzes bot.log to identify issues

Usage: python diagnose_bot.py
"""

import re
from collections import defaultdict
from datetime import datetime


def parse_log_file(log_path="bot.log"):
    """Parse bot.log and extract key metrics"""
    
    issues = []
    stats = {
        "total_stocks_scanned": 0,
        "viable_stocks": 0,
        "avg_confidence": [],
        "signals": defaultdict(int),
        "skipped_reasons": defaultdict(int),
        "positions_taken": 0,
        "positions_held": 0,
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[X] bot.log not found at {log_path}")
        return None, None
    
    for line in lines:
        # Count scanned stocks
        if "Evaluating" in line and "day" in line:
            stats["total_stocks_scanned"] += 1
            
            # Extract expected daily return
            match = re.search(r'\$\s*([-\d.]+)/day', line)
            if match:
                daily_return = float(match.group(1))
                if daily_return > 0:
                    stats["viable_stocks"] += 1
        
        # Extract confidence levels
        if "conf=" in line:
            match = re.search(r'conf=([\d.]+)', line)
            if match:
                stats["avg_confidence"].append(float(match.group(1)))
        
        # Count signals
        if " | BUY |" in line:
            stats["signals"]["buy"] += 1
        elif " | SELL |" in line:
            stats["signals"]["sell"] += 1
        elif " | HOLD |" in line:
            stats["signals"]["hold"] += 1
        
        # Track skipped stocks
        if "skipped" in line.lower():
            if "low volume" in line.lower():
                stats["skipped_reasons"]["low_volume"] += 1
            elif "below" in line.lower() and "threshold" in line.lower():
                stats["skipped_reasons"]["below_threshold"] += 1
        
        # Check for low confidence warnings
        if "Low confidence:" in line:
            match = re.search(r'Low confidence: ([\d.]+)', line)
            if match:
                conf = float(match.group(1))
                if conf < 0.01:
                    issues.append(f"CRITICAL: Very low confidence detected: {conf:.4f}")
        
        # Check for positions
        if "Portfolio:" in line and "$" in line:
            match = re.search(r'Portfolio: \$([\d.]+)', line)
            if match:
                portfolio_value = float(match.group(1))
                if portfolio_value > 0:
                    stats["positions_held"] = 1
    
    # Calculate averages
    if stats["avg_confidence"]:
        avg_conf = sum(stats["avg_confidence"]) / len(stats["avg_confidence"])
    else:
        avg_conf = 0.0
    
    # Identify issues
    if stats["total_stocks_scanned"] == 0:
        issues.append("CRITICAL: No stocks were scanned - bot may not be running")
    
    if stats["viable_stocks"] == 0:
        issues.append("WARNING: No viable stocks found - market may be bearish or strategy unsuitable")
    elif stats["viable_stocks"] < 5:
        issues.append(f"WARNING: Only {stats['viable_stocks']} viable stocks - limited opportunities")
    
    if avg_conf < 0.005:
        issues.append(f"CRITICAL: Average confidence is very low ({avg_conf:.4f}) - signals are weak")
    
    if stats["signals"]["hold"] > stats["signals"]["buy"] * 3:
        issues.append("WARNING: Too many HOLD signals - strategy may be too conservative")
    
    if stats["positions_held"] == 0 and stats["total_stocks_scanned"] > 0:
        issues.append("WARNING: No positions taken despite scanning - confidence too low or no buy signals")
    
    return stats, issues


def print_diagnosis():
    """Print diagnostic report"""
    print("=" * 70)
    print("BOT DIAGNOSTIC REPORT")
    print("=" * 70)
    print()
    
    stats, issues = parse_log_file()
    
    if stats is None:
        return
    
    # Print stats
    print("[STATISTICS]")
    print(f"  - Stocks Scanned: {stats['total_stocks_scanned']}")
    print(f"  - Viable Stocks: {stats['viable_stocks']} ({stats['viable_stocks']/max(1,stats['total_stocks_scanned'])*100:.1f}%)")
    
    if stats["avg_confidence"]:
        avg_conf = sum(stats["avg_confidence"]) / len(stats["avg_confidence"])
        print(f"  - Average Confidence: {avg_conf:.4f}")
    
    print(f"\n  Signals Generated:")
    print(f"    - BUY:  {stats['signals']['buy']}")
    print(f"    - SELL: {stats['signals']['sell']}")
    print(f"    - HOLD: {stats['signals']['hold']}")
    
    if stats["skipped_reasons"]:
        print(f"\n  Stocks Skipped:")
        for reason, count in stats["skipped_reasons"].items():
            print(f"    - {reason}: {count}")
    
    # Print issues
    print()
    print("=" * 70)
    if issues:
        print("[!] ISSUES DETECTED:")
        print()
        for i, issue in enumerate(issues, 1):
            icon = "[X]" if "CRITICAL" in issue else "[!]"
            print(f"  {icon} {issue}")
    else:
        print("[OK] NO ISSUES DETECTED - Bot appears healthy!")
    
    print()
    print("=" * 70)
    print("[RECOMMENDATIONS]")
    print()
    
    # Provide recommendations based on issues
    if any("low confidence" in i.lower() for i in issues):
        print("  1. Low Confidence Fix:")
        print("     - Try a different time interval: python optimizer.py -s AAPL -v")
        print("     - Consider using 1-hour or 2-hour intervals instead of 4-hour")
        print()
    
    if any("no viable stocks" in i.lower() for i in issues):
        print("  2. No Viable Stocks Fix:")
        print("     - Market may be bearish - wait for better conditions")
        print("     - Or reduce capital and try more frequent trading:")
        print("       python optimizer.py --preset conservative")
        print()
    
    if any("no positions taken" in i.lower() for i in issues):
        print("  3. No Positions Fix:")
        print("     - Lower MIN_CONFIDENCE_TO_TRADE in config:")
        print("       Set MIN_CONFIDENCE_TO_TRADE=0.001 in .env file")
        print("     - Or wait for clearer market signals")
        print()
    
    if stats["positions_held"] > 0:
        print("  [OK] Bot is actively trading - good!")
        print()
    
    print("  General Tips:")
    print("    - Check VIX level - high VIX (>30) pauses trading")
    print("    - Verify market hours - bot only trades 9:30 AM - 4:00 PM ET")
    print("    - Review optimization_history.csv for performance trends")
    print()
    print("=" * 70)


if __name__ == "__main__":
    print_diagnosis()

