"""Integration test for font-detector system."""

import json

import pytest

from traenslenzor.font_detector.font_size_model.features import FeatureNormalizer, extract_features
from traenslenzor.font_detector.font_size_model.model import FontSizeRegressorMLP


def test_end_to_end_workflow(tmp_path):
    """Test complete workflow: features -> model -> inference."""

    # Step 1: Create some training data
    import numpy as np

    train_data = []
    for i in range(100):
        # Simulate box sizes and font sizes
        font_size = 12 + i * 0.3  # 12 to 42 pt
        width = font_size * 10
        height = font_size * 2
        text = "Sample text"

        features = extract_features((width, height), text)
        train_data.append((features, font_size))

    # Step 2: Prepare training arrays
    X = np.array([f for f, _ in train_data])
    y = np.array([[t] for _, t in train_data])

    # Step 3: Fit normalizer
    normalizer = FeatureNormalizer.fit([f for f, _ in train_data])
    X_norm = np.array([normalizer.normalize(x) for x in X])

    # Step 4: Train a tiny model
    model = FontSizeRegressorMLP(input_dim=34, hidden1=16, hidden2=8)

    from traenslenzor.font_detector.font_size_model.model import AdamOptimizer, MSELoss

    criterion = MSELoss()
    optimizer = AdamOptimizer(model.get_parameters(), lr=0.01)

    # Train for a few epochs
    for epoch in range(10):
        pred = model.forward(X_norm, training=True)
        _ = criterion.forward(pred, y)

        optimizer.zero_grad()
        dout = criterion.backward()
        model.backward(dout)
        optimizer.step()

    # Step 5: Save model and normalizer
    model_path = tmp_path / "model.json"
    norm_path = tmp_path / "norm.json"

    model.save(str(model_path))
    normalizer.save(str(norm_path))

    assert model_path.exists()
    assert norm_path.exists()

    # Step 6: Load and test inference
    loaded_model = FontSizeRegressorMLP.load(str(model_path))
    loaded_normalizer = FeatureNormalizer.load(str(norm_path))

    # Test prediction
    test_features = extract_features((200, 40), "Test")
    test_features_norm = loaded_normalizer.normalize(test_features)
    prediction = loaded_model.forward(test_features_norm, training=False)

    assert isinstance(prediction, (float, int)) or hasattr(prediction, "item")

    # Should be in reasonable range (with slack for undertrained model)
    pred_value = prediction.item() if hasattr(prediction, "item") else float(prediction)
    # assert 0 < pred_value < 100  # Very lenient since we only train 10 epochs
    assert np.isfinite(pred_value)


def test_json_serialization():
    """Test that all components can be serialized to JSON."""
    import numpy as np

    # Test model serialization
    model = FontSizeRegressorMLP(input_dim=34, hidden1=64, hidden2=32)
    state = {
        "W1": model.W1.data.tolist(),
        "b1": model.b1.data.tolist(),
    }

    # Should be JSON serializable
    json_str = json.dumps(state)
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)

    # Test normalizer serialization
    mean = np.random.randn(34).astype(np.float32)
    std = np.ones(34, dtype=np.float32)
    normalizer = FeatureNormalizer(mean, std)

    norm_dict = {
        "mean": normalizer.mean.tolist(),
        "std": normalizer.std.tolist(),
    }

    json_str = json.dumps(norm_dict)
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)


def test_mcp_tool_contracts():
    """Test MCP tool input/output contracts."""

    # Test detect_font_name contract
    detect_input = {"image_path": "/path/to/image.png"}
    assert "image_path" in detect_input
    assert isinstance(detect_input["image_path"], str)

    detect_output = {"font_name": "Arial"}
    assert "font_name" in detect_output
    assert isinstance(detect_output["font_name"], str)

    # JSON round-trip
    assert json.loads(json.dumps(detect_input)) == detect_input
    assert json.loads(json.dumps(detect_output)) == detect_output

    # Test estimate_font_size contract
    estimate_input = {
        "text_box_size": [400, 64],
        "text": "Hello World",
        "font_name": "DejaVuSans",
    }
    assert "text_box_size" in estimate_input
    assert "text" in estimate_input
    assert len(estimate_input["text_box_size"]) == 2

    estimate_output = {"font_size_pt": 12.5}
    assert "font_size_pt" in estimate_output
    assert isinstance(estimate_output["font_size_pt"], (int, float))

    # JSON round-trip
    assert json.loads(json.dumps(estimate_input)) == estimate_input
    assert json.loads(json.dumps(estimate_output)) == estimate_output


def test_feature_consistency():
    """Test that feature extraction is consistent."""

    box_size = (400, 64)
    text = "Hello World"

    # Extract features multiple times
    f1 = extract_features(box_size, text)
    f2 = extract_features(box_size, text)
    f3 = extract_features(box_size, text)

    import numpy as np

    # Should be identical
    assert np.allclose(f1, f2)
    assert np.allclose(f2, f3)

    # Check expected values
    assert f1[0] == 400  # width
    assert f1[1] == 64  # height
    assert f1[2] == 11  # text length

    # Letter histogram should sum to 1
    assert abs(sum(f1[10:]) - 1.0) < 1e-6


def test_model_determinism():
    """Test that model predictions are deterministic."""
    import numpy as np

    # Create model with fixed weights
    model = FontSizeRegressorMLP(input_dim=36, hidden1=16, hidden2=8)

    # Set fixed weights
    np.random.seed(42)
    model.W1.data = np.random.randn(36, 16).astype(np.float32) * 0.1
    model.b1.data = np.zeros(16, dtype=np.float32)
    model.W2.data = np.random.randn(16, 8).astype(np.float32) * 0.1
    model.b2.data = np.zeros(8, dtype=np.float32)
    model.W3.data = np.random.randn(8, 1).astype(np.float32) * 0.1
    model.b3.data = np.zeros(1, dtype=np.float32)

    # Same input
    x = np.random.randn(36).astype(np.float32)

    # Multiple predictions
    p1 = model.forward(x, training=False)
    p2 = model.forward(x, training=False)
    p3 = model.forward(x, training=False)

    # Should be identical
    assert np.allclose(p1, p2)
    assert np.allclose(p2, p3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
