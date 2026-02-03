
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import pandas as pd

# Import our new engines
try:
    from strategies.decision_engine import DecisionEngine, TradeSignal
    from risk.allocation_engine import AllocationEngine, AllocationResult
    from portfolio_manager import PortfolioManager 
    from config_validated import get_config
except ImportError:
    # Handle running from separate directories if needed
    import sys
    sys.path.append("..")
    from strategies.decision_engine import DecisionEngine, TradeSignal
    from risk.allocation_engine import AllocationEngine, AllocationResult
    from portfolio_manager import PortfolioManager
    from config_validated import get_config

def run_backtest_simulation(closes: List[float], 
                            volumes: List[float], 
                            interval_seconds: int, 
                            start_capital: float = 10000.0) -> Dict[str, Any]:
    """
    Simulates the DecisionEngine and AllocationEngine over historical data.
    Replaces the old 'simulate_signals_and_projection'.
    """
    
    # Initialize Engines
    # We need to mock the PortfolioManager since we are just simulating
    pm = PortfolioManager(storage_file=None) # Mock or ephemeral PM
    # Initialize engines
    decision_engine = DecisionEngine()
    allocation_engine = AllocationEngine(pm)
    
    # Mock config override if needed for optimization? 
    # For now, we assume global config is what we test, or we can patch it.
    config = get_config()
    fee_per_trade = config.fee_per_trade_usd if config.simulate_fees_enabled else 0.0
    slippage_pct = (config.slippage_percent / 100.0) if config.simulate_slippage_enabled else 0.0
    
    cash = start_capital
    shares = 0
    equity = start_capital
    
    trades = []
    equity_curve = [start_capital]
    
    # Use a rolling window for analysis
    # We need enough history for indicators (config.long_window)
    # But we iterate through the list behaving as if 'now' is moving forward
    
    min_history = 200 # Safety buffer
    
    if len(closes) < min_history:
        return {"error": "Insufficient data"}

    # Optimization: Pre-calculate indicators? 
    # Real DecisionEngine calculates on the fly. For speed, this might be slow loop in Python.
    # But for correctness, we call .analyze()
    
    # Simulate
    for i in range(min_history, len(closes)):
        current_closes = closes[:i+1] # potentially expensive slicing
        current_volumes = volumes[:i+1] if volumes else []
        current_price = closes[i]
        
        # 1. Decision
        signal = decision_engine.analyze("BACKTEST", current_closes, current_volumes)
        
        # 2. Allocation
        # We need to update PM's state roughly
        # PM expects realtime data, here we 'mock' it by manually tracking position
        # We can patch PM:
        pm._positions = {}
        if shares > 0:
            pm.update_position("BACKTEST", shares, current_price, shares*current_price, 0)
        
        allocation = allocation_engine.calculate_allocation(signal, current_price, equity)
        
        # 3. Execution Logic (Simulation)
        if allocation.target_quantity > 0 and signal.action == "buy":
            # Check if we can afford (logic already in allocation, but double check cash)
            # Add slippage: Buy Price = Price * (1 + slippage)
            exec_price = current_price * (1 + slippage_pct)
            cost = (allocation.target_quantity * exec_price) + fee_per_trade
            if cost <= cash:
                shares += allocation.target_quantity
                cash -= cost
                trades.append({
                    "action": "buy",
                    "price": exec_price,
                    "qty": allocation.target_quantity,
                    "reason": allocation.reason,
                    "fee": fee_per_trade
                })
        
        elif signal.action == "sell" or (allocation.is_allowed and allocation.target_quantity == 0 and shares > 0):
             # Sell signal logic from AllocationEngine usually returns target=0 or specific sell
             # Here if signal says sell, we sell all (AllocationEngine current logic for sell is full liquidate)
             if shares > 0:
                 # Add slippage: Sell Price = Price * (1 - slippage)
                 exec_price = current_price * (1 - slippage_pct)
                 revenue = (shares * exec_price) - fee_per_trade
                 cash += revenue
                 shares = 0
                 trades.append({
                     "action": "sell",
                     "price": current_price,
                     "reason": signal.reasoning
                 })
                 
        # Update Equity
        equity = cash + (shares * current_price)
        equity_curve.append(equity)

    # Calculate Metrics
    total_return_pct = ((equity - start_capital) / start_capital) * 100.0
    
    # Sharpe
    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe = 0.0
    if len(returns) > 0 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * (6.5*3600/interval_seconds)) # roughly annualized

    # Max Drawdown
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown_pct = abs(drawdown.min()) * 100
    
    return {
        "final_equity": equity,
        "total_return_pct": total_return_pct,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown_pct,
        "num_trades": len(trades),
        "expected_daily_usd": (equity - start_capital) / (len(closes) * interval_seconds / (6.5*3600)),
        "win_rate": 0.5, # Placeholder, real calculation would check trade results
        "trade_returns": [t.get("profit") for t in trades if "profit" in t] # NOTE: Need to track trade profit in main loop to support this
    }

def monte_carlo_projection(trade_returns: List[float], 
                          starting_capital: float, 
                          trades_per_day: float, 
                          days: int = 30, 
                          num_simulations: int = 1000) -> Dict[str, float]:
    """
    Perform Monte Carlo simulation on trade returns.
    """
    if not trade_returns or len(trade_returns) < 5:
        return {"var_95": 0.0, "cvar_95": 0.0}
        
    sim_results = []
    num_trades = int(trades_per_day * days)
    
    for _ in range(num_simulations):
        # Sample with replacement
        daily_trades = np.random.choice(trade_returns, size=num_trades, replace=True)
        # Sum of returns (assuming dollar returns for simplicity, or pct?)
        # Optimizer seems to expect dollar returns potentially? 
        # Actually standard VaR uses pct returns.
        # Let's assume trade_returns are dollar amounts for now or check usage.
        # If returns are dollars:
        total_pnl = np.sum(daily_trades)
        sim_results.append(total_pnl)
        
    sim_results = np.array(sim_results)
    
    # Value at Risk (95%)
    var_95 = np.percentile(sim_results, 5) 
    
    # CVaR (Expected Shortfall)
    cvar_95 = sim_results[sim_results <= var_95].mean()
    
    return {
        "var_95": var_95,
        "cvar_95": cvar_95,
        "median_return": np.median(sim_results)
    }

def calculate_overnight_gap_risk(symbol: str, client, days: int = 60) -> Dict[str, float]:
    # Placeholder for logic removed from runner
    return {"gap_frequency": 0.0, "avg_gap_size": 0.0}

def calculate_market_beta(symbol: str, days: int = 60) -> float:
    # Placeholder
    return 1.0

def calculate_correlation_matrix(symbols: List[str], days: int = 60):
    return {}

