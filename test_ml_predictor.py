#!/usr/bin/env python3
"""
Unit tests for ML Predictor
Run with: pytest test_ml_predictor.py -v
"""

import pytest
import os
import tempfile
import numpy as np
from ml_predictor import TradingMLPredictor


@pytest.fixture
def temp_model_file():
    """Create a temporary model file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_file = f.name
    yield temp_file
    # Cleanup
    if os.path.exists(temp_file):
        os.remove(temp_file)


@pytest.fixture
def predictor(temp_model_file):
    """Create a TradingMLPredictor instance"""
    return TradingMLPredictor(temp_model_file)


def test_predictor_initialization(temp_model_file):
    """Test predictor initialization"""
    pred = TradingMLPredictor(temp_model_file)
    assert pred.model_path == temp_model_file
    assert pred.model is None
    assert pred.is_trained is False


def test_extract_features_basic(predictor):
    """Test feature extraction with basic data"""
    closes = [100 + i for i in range(30)]  # Uptrend
    volumes = [1000000 + i*10000 for i in range(30)]
    
    features = predictor.extract_features(closes, volumes, rsi=50.0)
    
    assert features is not None
    assert len(features) == 14  # 10 returns + RSI + vol + momentum + volatility


def test_extract_features_insufficient_data(predictor):
    """Test feature extraction with insufficient data"""
    closes = [100, 101, 102]  # Too short
    features = predictor.extract_features(closes)
    assert features is None


def test_extract_features_no_rsi(predictor):
    """Test feature extraction without RSI"""
    closes = [100 + i for i in range(30)]
    features = predictor.extract_features(closes, rsi=None)
    
    assert features is not None
    assert features[10] == 0.5  # Default RSI value


def test_extract_features_no_volume(predictor):
    """Test feature extraction without volume data"""
    closes = [100 + i for i in range(30)]
    features = predictor.extract_features(closes, volumes=None)
    
    assert features is not None
    assert features[11] == 0.5  # Default volume ratio


def test_train_insufficient_data(predictor):
    """Test training with insufficient data"""
    # Too few samples
    training_data = [(list(range(20, 40)), list(range(20)), 1) for _ in range(10)]
    
    success = predictor.train(training_data)
    assert success is False
    assert predictor.is_trained is False


def test_train_success(predictor):
    """Test successful model training"""
    # Generate synthetic training data
    np.random.seed(42)
    training_data = []
    
    for i in range(100):
        # Uptrend = label 1, downtrend = label 0
        if i % 2 == 0:
            closes = [100 + j + np.random.randn() for j in range(30)]
            label = 1
        else:
            closes = [100 - j + np.random.randn() for j in range(30)]
            label = 0
        
        volumes = [1000000 + np.random.randint(-10000, 10000) for _ in range(30)]
        training_data.append((closes, volumes, label))
    
    success = predictor.train(training_data, test_size=0.3)
    
    assert success is True
    assert predictor.is_trained is True
    assert predictor.model is not None


def test_predict_without_training(predictor):
    """Test prediction without training returns neutral"""
    closes = [100 + i for i in range(30)]
    prediction, confidence = predictor.predict(closes)
    
    assert prediction == 1  # Neutral prediction
    assert confidence == 0.5  # Neutral confidence


def test_predict_with_trained_model(predictor):
    """Test prediction with trained model"""
    # Train model first
    np.random.seed(42)
    training_data = []
    
    for i in range(100):
        if i % 2 == 0:
            closes = [100 + j + np.random.randn() for j in range(30)]
            label = 1
        else:
            closes = [100 - j + np.random.randn() for j in range(30)]
            label = 0
        
        volumes = [1000000 for _ in range(30)]
        training_data.append((closes, volumes, label))
    
    predictor.train(training_data)
    
    # Make prediction
    test_closes = [100 + i for i in range(30)]  # Uptrend
    prediction, confidence = predictor.predict(test_closes)
    
    assert prediction in [0, 1]
    assert 0.0 <= confidence <= 1.0


def test_predict_insufficient_data(predictor):
    """Test prediction with insufficient data"""
    closes = [100, 101]  # Too short
    prediction, confidence = predictor.predict(closes)
    
    assert prediction == 1  # Neutral
    assert confidence == 0.5


def test_save_and_load_model(temp_model_file):
    """Test saving and loading trained model"""
    # Train and save
    pred1 = TradingMLPredictor(temp_model_file)
    
    np.random.seed(42)
    training_data = []
    for i in range(100):
        closes = [100 + j + np.random.randn() for j in range(30)]
        volumes = [1000000 for _ in range(30)]
        label = 1 if i % 2 == 0 else 0
        training_data.append((closes, volumes, label))
    
    pred1.train(training_data)
    pred1.save_model()
    
    # Load in new instance
    pred2 = TradingMLPredictor(temp_model_file)
    loaded = pred2.load_model()
    
    assert loaded is True
    assert pred2.is_trained is True
    assert pred2.model is not None
    
    # Predictions should be consistent
    test_closes = [100 + i for i in range(30)]
    p1, c1 = pred1.predict(test_closes)
    p2, c2 = pred2.predict(test_closes)
    
    assert p1 == p2
    assert abs(c1 - c2) < 0.01  # Nearly identical confidence


def test_load_nonexistent_model(temp_model_file):
    """Test loading non-existent model"""
    # Remove file if exists
    if os.path.exists(temp_model_file):
        os.remove(temp_model_file)
    
    pred = TradingMLPredictor(temp_model_file)
    loaded = pred.load_model()
    
    assert loaded is False
    assert pred.is_trained is False


def test_feature_extraction_with_volatility(predictor):
    """Test that volatility is calculated correctly"""
    # High volatility data (more dramatic swings)
    closes_volatile = [100, 120, 80, 140, 70, 130, 75, 125, 85, 115] + [100] * 20
    features_volatile = predictor.extract_features(closes_volatile)
    
    # Low volatility data (nearly flat)
    closes_stable = [100.0] * 30
    features_stable = predictor.extract_features(closes_stable)
    
    assert features_volatile is not None
    assert features_stable is not None
    # Volatility feature is at index 13 (last feature)
    # Volatile data should have higher volatility than flat data
    assert features_volatile[13] >= features_stable[13]


def test_feature_extraction_with_momentum(predictor):
    """Test that momentum is calculated correctly"""
    # Strong uptrend
    closes_up = [100 + i*2 for i in range(30)]
    features_up = predictor.extract_features(closes_up)
    
    # Strong downtrend
    closes_down = [100 - i*2 for i in range(30)]
    features_down = predictor.extract_features(closes_down)
    
    assert features_up is not None
    assert features_down is not None
    # Momentum feature is at index 12
    assert features_up[12] > 0  # Positive momentum
    assert features_down[12] < 0  # Negative momentum


def test_model_prediction_confidence_bounds(predictor):
    """Test that prediction confidence is always in valid range"""
    np.random.seed(42)
    training_data = []
    for i in range(100):
        closes = [100 + j + np.random.randn() for j in range(30)]
        volumes = [1000000 for _ in range(30)]
        label = 1 if i % 2 == 0 else 0
        training_data.append((closes, volumes, label))
    
    predictor.train(training_data)
    
    # Test with various inputs
    for _ in range(20):
        test_closes = [100 + np.random.randn() for _ in range(30)]
        prediction, confidence = predictor.predict(test_closes)
        
        assert prediction in [0, 1]
        assert 0.0 <= confidence <= 1.0


def test_train_with_volume_patterns(predictor):
    """Test training with volume patterns"""
    np.random.seed(42)
    training_data = []
    
    for i in range(100):
        closes = [100 + j for j in range(30)]
        # High volume for uptrends
        if i % 2 == 0:
            volumes = [2000000 for _ in range(30)]
            label = 1
        else:
            volumes = [500000 for _ in range(30)]
            label = 0
        training_data.append((closes, volumes, label))
    
    success = predictor.train(training_data)
    assert success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

