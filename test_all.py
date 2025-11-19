#!/usr/bin/env python3
"""
Consolidated test suite for the Paper-Trading project.

This single file merges all existing unit tests into one comprehensive
pytest file so you can run the whole suite with a single command.

Run: pytest test_all.py -q
"""

import os
import sys
import tempfile
import pytest
import time
import numpy as np
from unittest.mock import Mock, patch
from typing import List, Dict, Any, Tuple, Generator

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runner import (
    sma, decide_action, compute_confidence, pct_stddev,
    compute_order_qty_from_remaining, adjust_runtime_params,
    snap_interval_to_supported_seconds,
    check_exposure_limit, check_kill_switch, verify_order_safety,
    calculate_max_position_size_for_risk, check_vix_filter
)

from stock_scanner import (
    get_stock_universe,
    FALLBACK_STOCK_UNIVERSE,
    DEFAULT_TOP_100_STOCKS
)

from portfolio_manager import PortfolioManager
from ml_predictor import TradingMLPredictor


# ------------------------
# Shared fixtures (copied from conftest.py)
# ------------------------

@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_config_values() -> Dict[str, Any]:
    return {
        "MAX_CAPITAL": 10000.0,
        "SHORT_WINDOW": 3,
        "LONG_WINDOW": 9,
        "BASE_TAKE_PROFIT": 3.0,
        "BASE_STOP_LOSS": 1.0,
        "BASE_TRADE_SIZE_FRAC": 0.5,
        "MAX_EXPOSURE_PCT": 75.0,
        "MAX_LOSS_PER_TRADE_PCT": 2.0,
        "VIX_THRESHOLD": 30.0,
        "MAX_PRICE_DEVIATION_PCT": 10.0,
    }


@pytest.fixture
def sample_price_data() -> Dict[str, List[float]]:
    return {
        "uptrend": [100 + i for i in range(30)],
        "downtrend": [130 - i for i in range(30)],
        "sideways": [100 + (i % 5) for i in range(30)],
        "volatile": [100 + (i % 2) * 10 * ((-1) ** i) for i in range(30)],
    }


@pytest.fixture(autouse=True)
def reset_test_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    monkeypatch.setenv("TEST_MODE", "1")
    yield


# ------------------------
# Runner tests (from test_runner.py)
# ------------------------

class TestSMA:
    def test_sma_basic(self):
        closes = [100.0, 102.0, 104.0, 106.0, 108.0]
        assert sma(closes, 3) == 106.0
        assert sma(closes, 5) == 104.0

    def test_sma_insufficient_data(self):
        closes = [100.0, 102.0]
        assert sma(closes, 5) == 102.0

    def test_sma_empty_list(self):
        closes: List[float] = []
        assert sma(closes, 5) == 0.0

    def test_sma_single_value(self):
        closes = [100.0]
        assert sma(closes, 5) == 100.0

    def test_sma_window_size_one(self):
        closes = [100.0, 102.0, 104.0]
        assert sma(closes, 1) == 104.0


class TestDecideAction:
    def test_decide_action_buy(self):
        closes = [100.0 + i for i in range(12)]
        action = decide_action(closes, short_w=3, long_w=9)
        assert action == "buy"

    def test_decide_action_sell(self):
        closes = [120.0 - i for i in range(12)]
        action = decide_action(closes, short_w=3, long_w=9)
        assert action == "sell"

    def test_decide_action_hold(self):
        closes = [100.0, 101.0] * 6
        action = decide_action(closes, short_w=3, long_w=9)
        assert action == "hold"

    def test_decide_action_insufficient_data(self):
        closes = [100.0, 102.0]
        action = decide_action(closes, short_w=3, long_w=9)
        assert action == "hold"


class TestComputeConfidence:
    def test_compute_confidence_uptrend(self):
        closes = [100.0 + i for i in range(21)]
        conf = compute_confidence(closes)
        assert conf > 0.05

    def test_compute_confidence_flat(self):
        closes = [100.0] * 20
        conf = compute_confidence(closes)
        assert conf >= 0.0

    def test_compute_confidence_downtrend(self):
        closes = [140.0 - i for i in range(21)]
        conf = compute_confidence(closes)
        assert conf >= 0.0

    def test_compute_confidence_insufficient_data(self):
        closes = [100.0, 102.0]
        conf = compute_confidence(closes)
        assert conf == 0.0


class TestPctStddev:
    def test_pct_stddev_low_volatility(self):
        closes = [100.0, 101.0, 100.0, 101.0, 100.0]
        vol = pct_stddev(closes)
        assert 0 < vol < 0.01

    def test_pct_stddev_high_volatility(self):
        closes = [100.0, 120.0, 80.0, 110.0, 90.0]
        vol = pct_stddev(closes)
        assert vol > 0.1

    def test_pct_stddev_zero_volatility(self):
        closes = [100.0] * 10
        vol = pct_stddev(closes)
        assert vol == 0.0

    def test_pct_stddev_single_value(self):
        closes = [100.0]
        vol = pct_stddev(closes)
        assert vol == 0.0


class TestComputeOrderQty:
    def test_compute_order_qty_basic(self):
        qty = compute_order_qty_from_remaining(100.0, 1000.0, 0.5)
        assert qty == 5.0
        qty = compute_order_qty_from_remaining(100.0, 550.0, 1.0)
        assert qty == 5.5

    def test_compute_order_qty_zero_capital(self):
        qty = compute_order_qty_from_remaining(100.0, 0.0, 0.5)
        assert qty == 0.0

    def test_compute_order_qty_small_fraction(self):
        qty = compute_order_qty_from_remaining(100.0, 1000.0, 0.1)
        assert qty == 1.0


class TestAdjustRuntimeParams:
    def test_adjust_runtime_params_basic(self):
        tp, sl, frac = adjust_runtime_params(
            confidence=0.05, base_tp=3.0, base_sl=1.0, base_frac=0.5
        )
        assert tp > 0.0 and sl > 0.0 and frac > 0.0

    def test_adjust_runtime_params_bounds(self):
        tp, sl, frac = adjust_runtime_params(
            confidence=10.0, base_tp=20.0, base_sl=15.0, base_frac=0.95
        )
        assert tp > 0.0 and sl > 0.0 and 0.0 < frac <= 1.0

    def test_adjust_runtime_params_zero_confidence(self):
        tp, sl, frac = adjust_runtime_params(
            confidence=0.0, base_tp=3.0, base_sl=1.0, base_frac=0.5
        )
        assert tp > 0.0 and sl > 0.0 and frac > 0.0

    def test_adjust_runtime_params_high_confidence(self):
        tp, _, frac = adjust_runtime_params(
            confidence=1.0, base_tp=3.0, base_sl=1.0, base_frac=0.5
        )
        assert tp >= 3.0 and frac >= 0.5


class TestSnapInterval:
    def test_snap_interval_exact_match(self):
        assert snap_interval_to_supported_seconds(3600) == 3600
        assert snap_interval_to_supported_seconds(900) == 900

    def test_snap_interval_needs_snapping(self):
        result = snap_interval_to_supported_seconds(1000)
        assert result in [60, 300, 900, 1800, 3600, 7200, 14400, 21600]

    def test_snap_interval_very_small(self):
        result = snap_interval_to_supported_seconds(30)
        assert result >= 60

    def test_snap_interval_very_large(self):
        result = snap_interval_to_supported_seconds(100000)
        assert result <= 23400


# ------------------------
# Stock scanner tests
# ------------------------

def test_get_stock_universe_with_user_symbols():
    user_symbols = ["AAPL", "msft", "googl"]
    result = get_stock_universe(user_symbols=user_symbols, use_top_100=False)
    assert len(result) == 3
    assert "AAPL" in result
    assert "MSFT" in result
    assert "GOOGL" in result


def test_get_stock_universe_fallback():
    result = get_stock_universe(user_symbols=None, use_top_100=False)
    assert len(result) > 0
    assert result == FALLBACK_STOCK_UNIVERSE


def test_get_stock_universe_top_100():
    result = get_stock_universe(user_symbols=None, use_top_100=True)
    assert len(result) > 0
    assert isinstance(result, list)


def test_fallback_stock_universe_validity():
    assert len(FALLBACK_STOCK_UNIVERSE) > 0
    assert "SPY" in FALLBACK_STOCK_UNIVERSE
    assert "QQQ" in FALLBACK_STOCK_UNIVERSE


def test_default_top_100_stocks_validity():
    assert len(DEFAULT_TOP_100_STOCKS) >= 100
    assert "AAPL" in DEFAULT_TOP_100_STOCKS


def test_get_stock_universe_empty_user_symbols():
    result = get_stock_universe(user_symbols=[], use_top_100=False)
    assert len(result) > 0
    assert result == FALLBACK_STOCK_UNIVERSE


def test_get_stock_universe_user_symbols_priority():
    user_symbols = ["CUSTOM1", "CUSTOM2"]
    result = get_stock_universe(user_symbols=user_symbols, use_top_100=True)
    assert result == ["CUSTOM1", "CUSTOM2"]


# ------------------------
# Risk controls tests (from test_risk_controls.py)
# ------------------------

class TestExposureLimit:
    def test_exposure_limit_within_bounds(self):
        can_continue, scaled_value, _ = check_exposure_limit(
            total_invested=5000.0, max_capital=10000.0, new_order_value=1000.0
        )
        assert can_continue is True
        assert scaled_value == 1000.0

    def test_exposure_limit_at_max(self):
        can_continue, scaled_value, _ = check_exposure_limit(
            total_invested=7500.0, max_capital=10000.0, new_order_value=1000.0
        )
        assert can_continue is False
        assert scaled_value == 0.0

    def test_exposure_limit_partial_order(self):
        can_continue, scaled_value, _ = check_exposure_limit(
            total_invested=7000.0, max_capital=10000.0, new_order_value=1000.0
        )
        assert can_continue is True
        assert 0 < scaled_value < 1000.0

    def test_exposure_limit_disabled(self):
        can_continue, scaled_value, _ = check_exposure_limit(
            total_invested=5000.0, max_capital=0.0, new_order_value=1000.0
        )
        assert can_continue is False
        assert scaled_value == 0.0

    def test_exposure_limit_zero_capital(self):
        can_continue, scaled_value, _ = check_exposure_limit(
            total_invested=0.0, max_capital=0.0, new_order_value=1000.0
        )
        assert can_continue is False
        assert scaled_value == 0.0


class TestKillSwitch:
    @pytest.fixture
    def kill_switch_file(self) -> Generator[str, None, None]:
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            temp_file = f.name
        yield temp_file
        if os.path.exists(temp_file):
            os.remove(temp_file)

    def test_kill_switch_not_present(self, kill_switch_file: str):
        if os.path.exists(kill_switch_file):
            os.remove(kill_switch_file)
        with patch('config.KILL_SWITCH_FILE', kill_switch_file):
            with patch('config.KILL_SWITCH_ENABLED', True):
                can_continue, _ = check_kill_switch()
                assert can_continue is True

    def test_kill_switch_present(self, kill_switch_file: str):
        with open(kill_switch_file, 'w') as f:
            f.write("STOP")
        with patch('config.KILL_SWITCH_FILE', kill_switch_file):
            with patch('config.KILL_SWITCH_ENABLED', True):
                can_continue, _ = check_kill_switch()
                assert can_continue is False

    def test_kill_switch_disabled(self, kill_switch_file: str):
        with open(kill_switch_file, 'w') as f:
            f.write("STOP")
        with patch('config.KILL_SWITCH_FILE', kill_switch_file):
            with patch('config.KILL_SWITCH_ENABLED', False):
                can_continue, _ = check_kill_switch()
                assert can_continue is True


class TestOrderVerification:
    def test_order_verification_disabled(self):
        mock_client = Mock()
        with patch('config.ORDER_VERIFICATION_ENABLED', False):
            is_safe, _ = verify_order_safety(mock_client, "AAPL", "buy", 10.0, 150.0, 150.0)
            assert is_safe is True

    def test_order_verification_price_within_limits(self):
        mock_client = Mock()
        with patch('config.ORDER_VERIFICATION_ENABLED', True):
            with patch('config.MAX_PRICE_DEVIATION_PCT', 10.0):
                is_safe, _ = verify_order_safety(mock_client, "AAPL", "buy", 10.0, 155.0, 150.0)
                assert is_safe is True

    def test_order_verification_price_too_high(self):
        mock_client = Mock()
        with patch('config.ORDER_VERIFICATION_ENABLED', True):
            with patch('config.MAX_PRICE_DEVIATION_PCT', 5.0):
                is_safe, _ = verify_order_safety(mock_client, "AAPL", "buy", 10.0, 200.0, 150.0)
                assert is_safe is False

    def test_order_verification_no_last_price(self):
        mock_client = Mock()
        with patch('config.ORDER_VERIFICATION_ENABLED', True):
            is_safe, _ = verify_order_safety(mock_client, "AAPL", "buy", 10.0, 150.0, None)
            assert is_safe is True


class TestMaxLossPerTrade:
    def test_max_loss_calculation_basic(self):
        with patch('config.MAX_LOSS_PER_TRADE_PCT', 2.0):
            max_position, _ = calculate_max_position_size_for_risk(
                total_capital=10000.0, stop_loss_pct=1.0, available_capital=5000.0
            )
            assert max_position == 5000.0

    def test_max_loss_limits_position(self):
        with patch('config.MAX_LOSS_PER_TRADE_PCT', 1.0):
            max_position, _ = calculate_max_position_size_for_risk(
                total_capital=10000.0, stop_loss_pct=2.0, available_capital=10000.0
            )
            assert max_position == 5000.0

    def test_max_loss_zero_stop_loss(self):
        with patch('config.MAX_LOSS_PER_TRADE_PCT', 2.0):
            max_position, _ = calculate_max_position_size_for_risk(
                total_capital=10000.0, stop_loss_pct=0.0, available_capital=5000.0
            )
            assert max_position > 0.0


class TestVixFilter:
    def test_vix_filter_disabled(self):
        with patch('config.VIX_FILTER_ENABLED', False):
            can_trade, _ = check_vix_filter()
            assert can_trade is True

    @patch('runner.get_vix_level')
    def test_vix_filter_below_threshold(self, mock_get_vix: Mock):
        mock_get_vix.return_value = (20.0, "VIX: 20.0")
        with patch('config.VIX_FILTER_ENABLED', True):
            with patch('config.VIX_THRESHOLD', 30.0):
                can_trade, _ = check_vix_filter()
                assert can_trade is True

    @patch('runner.get_vix_level')
    def test_vix_filter_above_threshold(self, mock_get_vix: Mock):
        mock_get_vix.return_value = (35.0, "VIX: 35.0")
        with patch('config.VIX_FILTER_ENABLED', True):
            with patch('config.VIX_THRESHOLD', 30.0):
                can_trade, _ = check_vix_filter()
                assert can_trade is False

    @patch('runner.get_vix_level')
    def test_vix_filter_fetch_failure(self, mock_get_vix: Mock):
        mock_get_vix.return_value = (None, "Failed to fetch VIX")
        with patch('config.VIX_FILTER_ENABLED', True):
            can_trade, _ = check_vix_filter()
            assert can_trade is True


# ------------------------
# Portfolio manager tests (from test_portfolio_manager.py)
# ------------------------

@pytest.fixture
def temp_portfolio_file() -> Generator[str, None, None]:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    yield temp_file
    if os.path.exists(temp_file):
        os.remove(temp_file)


@pytest.fixture
def portfolio(temp_portfolio_file: str) -> PortfolioManager:
    return PortfolioManager(temp_portfolio_file)


def test_portfolio_initialization(temp_portfolio_file: str):
    pm = PortfolioManager(temp_portfolio_file)
    assert pm.portfolio_file == temp_portfolio_file
    assert pm.positions == {}


def test_portfolio_load_empty():
    pm = PortfolioManager("nonexistent_file.json")
    assert pm.positions == {}


def test_portfolio_save_and_load(temp_portfolio_file: str):
    pm1 = PortfolioManager(temp_portfolio_file)
    pm1.update_position("AAPL", 10.0, 150.0, 1500.0, 50.0)
    pm1.save()
    pm2 = PortfolioManager(temp_portfolio_file)
    assert "AAPL" in pm2.positions


def test_update_position(portfolio: PortfolioManager):
    portfolio.update_position("TSLA", 5.0, 200.0, 1000.0, 100.0)
    assert "TSLA" in portfolio.positions


def test_update_existing_position_preserves_first_opened(portfolio: PortfolioManager):
    portfolio.update_position("AAPL", 10.0, 150.0, 1500.0, 0.0)
    first_opened_1 = portfolio.positions["AAPL"]["first_opened"]
    time.sleep(0.01)
    portfolio.update_position("AAPL", 15.0, 155.0, 2325.0, 75.0)
    first_opened_2 = portfolio.positions["AAPL"]["first_opened"]
    assert first_opened_1 == first_opened_2


def test_remove_position(portfolio: PortfolioManager):
    portfolio.update_position("NVDA", 20.0, 400.0, 8000.0, 200.0)
    portfolio.remove_position("NVDA")
    assert "NVDA" not in portfolio.positions


def test_remove_nonexistent_position(portfolio: PortfolioManager):
    portfolio.remove_position("NONEXISTENT")
    assert "NONEXISTENT" not in portfolio.positions


def test_get_position(portfolio: PortfolioManager):
    portfolio.update_position("MSFT", 8.0, 300.0, 2400.0, 80.0)
    pos = portfolio.get_position("MSFT")
    assert pos is not None


def test_get_all_positions(portfolio: PortfolioManager):
    portfolio.update_position("AAPL", 10.0, 150.0, 1500.0, 50.0)
    portfolio.update_position("GOOGL", 5.0, 2800.0, 14000.0, 200.0)
    all_pos = portfolio.get_all_positions()
    assert len(all_pos) == 2


def test_get_position_count(portfolio: PortfolioManager):
    assert portfolio.get_position_count() == 0
    portfolio.update_position("AAPL", 10.0, 150.0, 1500.0, 50.0)
    assert portfolio.get_position_count() == 1


def test_get_total_market_value(portfolio: PortfolioManager):
    assert portfolio.get_total_market_value() == 0.0
    portfolio.update_position("AAPL", 10.0, 150.0, 1500.0, 50.0)
    assert portfolio.get_total_market_value() == 1500.0


def test_get_total_unrealized_pl(portfolio: PortfolioManager):
    assert portfolio.get_total_unrealized_pl() == 0.0
    portfolio.update_position("AAPL", 10.0, 150.0, 1500.0, 50.0)
    assert portfolio.get_total_unrealized_pl() == 50.0


def test_get_worst_performer(portfolio: PortfolioManager):
    assert portfolio.get_worst_performer() is None
    portfolio.update_position("WINNER", 10.0, 100.0, 1200.0, 200.0)
    portfolio.update_position("LOSER", 10.0, 100.0, 800.0, -200.0)
    result = portfolio.get_worst_performer()
    assert result is not None
    worst_symbol, _ = result
    assert worst_symbol == "LOSER"


def test_has_room_for_new_position(portfolio: PortfolioManager):
    max_positions = 5
    assert portfolio.has_room_for_new_position(max_positions) is True
    for i in range(max_positions):
        portfolio.update_position(f"STOCK{i}", 10.0, 100.0, 1000.0, 0.0)
    assert portfolio.has_room_for_new_position(max_positions) is False


def test_get_symbols(portfolio: PortfolioManager):
    portfolio.update_position("AAPL", 10.0, 150.0, 1500.0, 50.0)
    symbols = portfolio.get_symbols()
    assert len(symbols) >= 1


def test_portfolio_persistence(temp_portfolio_file: str):
    pm1 = PortfolioManager(temp_portfolio_file)
    pm1.update_position("AAPL", 10.0, 150.0, 1500.0, 50.0)
    pm1.update_position("GOOGL", 5.0, 2800.0, 14000.0, 200.0)
    pm2 = PortfolioManager(temp_portfolio_file)
    assert pm2.get_position_count() == 2


def test_portfolio_handles_corrupted_file(temp_portfolio_file: str):
    with open(temp_portfolio_file, 'w') as f:
        f.write("corrupted json {{{")
    pm = PortfolioManager(temp_portfolio_file)
    assert pm.positions == {}


# ------------------------
# ML predictor tests (from test_ml_predictor.py)
# ------------------------

@pytest.fixture
def temp_model_file() -> Generator[str, None, None]:
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_file = f.name
    yield temp_file
    if os.path.exists(temp_file):
        os.remove(temp_file)


@pytest.fixture
def predictor(temp_model_file: str) -> TradingMLPredictor:
    return TradingMLPredictor(temp_model_file)


def test_predictor_initialization(temp_model_file: str):
    pred = TradingMLPredictor(temp_model_file)
    assert pred.model_path == temp_model_file
    assert pred.model is None


def test_extract_features_basic(predictor: TradingMLPredictor):
    closes = [100.0 + i for i in range(30)]
    volumes = [1000000.0 + i*10000.0 for i in range(30)]
    features = predictor.extract_features(closes, volumes, rsi=50.0)
    assert features is not None


def test_extract_features_insufficient_data(predictor: TradingMLPredictor):
    closes = [100.0, 101.0, 102.0]
    features = predictor.extract_features(closes)
    assert features is None


def test_train_insufficient_data(predictor: TradingMLPredictor):
    training_data = [(list(range(20, 40)), list(range(20)), 1) for _ in range(10)]
    # Convert to floats
    training_data_float: List[Tuple[List[float], List[float], int]] = []
    for c, v, l in training_data:
        training_data_float.append(([float(x) for x in c], [float(x) for x in v], l))
    
    success = predictor.train(training_data_float)
    assert success is False


def test_train_success(predictor: TradingMLPredictor):
    np.random.seed(42) # type: ignore
    training_data: List[Tuple[List[float], List[float], int]] = []
    for i in range(100):
        if i % 2 == 0:
            closes: List[float] = [100.0 + j + float(np.random.randn()) for j in range(30)] # type: ignore
            label = 1
        else:
            closes: List[float] = [100.0 - j + float(np.random.randn()) for j in range(30)] # type: ignore
            label = 0
        volumes: List[float] = [1000000.0 + float(np.random.randint(-10000, 10000)) for _ in range(30)] # type: ignore
        training_data.append((closes, volumes, label))
    success = predictor.train(training_data, test_size=0.3)
    assert success is True


def test_predict_without_training(predictor: TradingMLPredictor):
    closes = [100.0 + i for i in range(30)]
    prediction, confidence = predictor.predict(closes)
    assert isinstance(prediction, int)
    assert 0.0 <= confidence <= 1.0


def test_save_and_load_model(temp_model_file: str):
    pred1 = TradingMLPredictor(temp_model_file)
    np.random.seed(42) # type: ignore
    training_data: List[Tuple[List[float], List[float], int]] = []
    for i in range(100):
        closes: List[float] = [100.0 + j + float(np.random.randn()) for j in range(30)] # type: ignore
        volumes: List[float] = [1000000.0 for _ in range(30)]
        label = 1 if i % 2 == 0 else 0
        training_data.append((closes, volumes, label))
    pred1.train(training_data)
    pred1.save_model()
    pred2 = TradingMLPredictor(temp_model_file)
    loaded = pred2.load_model()
    assert loaded is True


def test_load_nonexistent_model(temp_model_file: str):
    if os.path.exists(temp_model_file):
        os.remove(temp_model_file)
    pred = TradingMLPredictor(temp_model_file)
    loaded = pred.load_model()
    assert loaded is False


def test_feature_extraction_with_volatility(predictor: TradingMLPredictor):
    closes_volatile = [100.0, 120.0, 80.0, 140.0, 70.0, 130.0, 75.0, 125.0, 85.0, 115.0] + [100.0] * 20
    features_volatile = predictor.extract_features(closes_volatile)
    closes_stable = [100.0] * 30
    features_stable = predictor.extract_features(closes_stable)
    assert features_volatile is not None and features_stable is not None


def test_model_prediction_confidence_bounds(predictor: TradingMLPredictor):
    np.random.seed(42) # type: ignore
    training_data: List[Tuple[List[float], List[float], int]] = []
    for i in range(100):
        closes: List[float] = [100.0 + j + float(np.random.randn()) for j in range(30)] # type: ignore
        volumes: List[float] = [1000000.0 for _ in range(30)]
        label = 1 if i % 2 == 0 else 0
        training_data.append((closes, volumes, label))
    predictor.train(training_data)
    for _ in range(10):
        test_closes: List[float] = [100.0 + float(np.random.randn()) for _ in range(30)] # type: ignore
        prediction, confidence = predictor.predict(test_closes)
        assert prediction in [0, 1]
        assert 0.0 <= confidence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-q"]) 
