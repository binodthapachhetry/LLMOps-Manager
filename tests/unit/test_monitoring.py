"""
Unit tests for the monitoring system.
"""
import pytest
import time
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from src.monitoring.metrics import Alert, DriftDetector, MetricsCollector, AlertManager, MonitoringSystem


def test_alert_lifecycle():
    """Test the lifecycle of an alert."""
    alert = Alert(
        metric_name="latency",
        threshold=500.0,
        current_value=600.0,
        deployment_id="test-deployment-id",
        severity="warning",
    )
    
    # Initial state
    assert alert.metric_name == "latency"
    assert alert.threshold == 500.0
    assert alert.current_value == 600.0
    assert alert.deployment_id == "test-deployment-id"
    assert alert.severity == "warning"
    assert alert.resolved is False
    assert alert.resolved_timestamp is None
    
    # Resolve the alert
    alert.resolve()
    assert alert.resolved is True
    assert alert.resolved_timestamp is not None
    
    # Convert to dictionary
    alert_dict = alert.to_dict()
    assert alert_dict["metric_name"] == "latency"
    assert alert_dict["threshold"] == 500.0
    assert alert_dict["current_value"] == 600.0
    assert alert_dict["deployment_id"] == "test-deployment-id"
    assert alert_dict["severity"] == "warning"
    assert alert_dict["resolved"] is True
    assert alert_dict["resolved_timestamp"] is not None


def test_drift_detector():
    """Test the drift detector."""
    config = {
        "window_size": 10,
        "drift_threshold": 0.05,
    }
    
    detector = DriftDetector(config)
    
    # Initial state
    assert detector.reference_data is None
    assert len(detector.current_window) == 0
    assert detector.window_size == 10
    assert detector.drift_threshold == 0.05
    
    # Add samples
    for i in range(15):
        detector.add_sample({"input": f"sample_{i}", "output": f"result_{i}"})
    
    # Window should be capped at window_size
    assert len(detector.current_window) == 10
    assert detector.current_window[0]["input"] == "sample_5"
    
    # Detect drift without reference data
    drift = detector.detect_drift()
    assert drift["input_drift"] == 0.0
    assert drift["output_drift"] == 0.0
    
    # Set reference data and detect drift
    detector.set_reference_data([{"input": f"ref_{i}", "output": f"ref_result_{i}"} for i in range(20)])
    drift = detector.detect_drift()
    assert "input_drift" in drift
    assert "output_drift" in drift


def test_metrics_collector():
    """Test the metrics collector."""
    provider = MagicMock()
    provider.get_metrics.return_value = {
        "latency": [{"timestamp": datetime.now().isoformat(), "value": 150.0}],
        "throughput": [{"timestamp": datetime.now().isoformat(), "value": 100.0}],
    }
    
    config = {
        "collection_interval_seconds": 1,  # Short interval for testing
        "enabled_metrics": ["latency", "throughput"],
    }
    
    collector = MetricsCollector(
        deployment_id="test-deployment-id",
        provider=provider,
        config=config,
    )
    
    # Initial state
    assert collector.deployment_id == "test-deployment-id"
    assert collector.provider == provider
    assert collector.collection_interval == 1
    assert collector.enabled_metrics == ["latency", "throughput"]
    assert collector.metrics == {}
    
    # Start collecting metrics
    collector.start()
    assert collector.running is True
    
    # Wait for collection
    time.sleep(2)
    
    # Stop collecting
    collector.stop()
    assert collector.running is False
    
    # Check collected metrics
    assert "latency" in collector.metrics
    assert "throughput" in collector.metrics
    
    # Get metrics
    metrics = collector.get_metrics(
        metric_names=["latency"],
        start_time=(datetime.now() - timedelta(hours=1)).isoformat(),
        end_time=datetime.now().isoformat(),
    )
    
    assert "latency" in metrics
    assert len(metrics["latency"]) > 0


def test_alert_manager():
    """Test the alert manager."""
    config = {
        "thresholds": {
            "latency": {
                "warning": 500.0,
                "error": 1000.0,
                "critical": 2000.0,
                "direction": "above",
            },
            "error_rate": {
                "warning": 0.01,
                "error": 0.05,
                "critical": 0.1,
                "direction": "above",
            },
            "gpu_utilization": {
                "warning": 20.0,
                "direction": "below",
            },
        },
    }
    
    manager = AlertManager(config)
    
    # Initial state
    assert manager.thresholds == config["thresholds"]
    assert len(manager.alerts) == 0
    assert len(manager.alert_handlers) == 0
    
    # Register an alert handler
    handler_called = False
    def test_handler(alert):
        nonlocal handler_called
        handler_called = True
    
    manager.register_alert_handler(test_handler)
    assert len(manager.alert_handlers) == 1
    
    # Check thresholds - should trigger alerts
    metrics = {
        "latency": [
            {"timestamp": datetime.now().isoformat(), "value": 1500.0}
        ],
        "error_rate": [
            {"timestamp": datetime.now().isoformat(), "value": 0.02}
        ],
        "gpu_utilization": [
            {"timestamp": datetime.now().isoformat(), "value": 10.0}
        ],
    }
    
    alerts = manager.check_thresholds(metrics, "test-deployment-id")
    
    # Should have triggered 3 alerts (error for latency, warning for error_rate, warning for gpu_utilization)
    assert len(alerts) == 3
    assert handler_called is True
    
    # Get active alerts
    active_alerts = manager.get_active_alerts()
    assert len(active_alerts) == 3
    
    # Filter by deployment ID
    filtered_alerts = manager.get_active_alerts(deployment_id="test-deployment-id")
    assert len(filtered_alerts) == 3
    
    # Filter by severity
    warning_alerts = manager.get_active_alerts(severity="warning")
    assert len(warning_alerts) == 2
    
    error_alerts = manager.get_active_alerts(severity="error")
    assert len(error_alerts) == 1
    
    # Resolve an alert
    manager.resolve_alert(0)
    assert manager.alerts[0].resolved is True
    
    # Get active alerts again
    active_alerts = manager.get_active_alerts()
    assert len(active_alerts) == 2


def test_monitoring_system():
    """Test the monitoring system."""
    provider = MagicMock()
    config = {
        "alerts": {
            "thresholds": {
                "latency": {
                    "warning": 500.0,
                },
            },
        },
        "collectors": {
            "collection_interval_seconds": 1,  # Short interval for testing
            "enabled_metrics": ["latency", "throughput"],
        },
        "threshold_check_interval_seconds": 1,  # Short interval for testing
    }
    
    system = MonitoringSystem(provider, config)
    
    # Initial state
    assert system.provider == provider
    assert system.config == config
    assert len(system.collectors) == 0
    assert system.alert_manager is not None
    
    # Register a deployment
    system.register_deployment(
        deployment_id="test-deployment-id",
        model_id="test-model",
        model_version="v1",
    )
    
    assert "test-deployment-id" in system.collectors
    
    # Start the system
    system.start()
    assert system.running is True
    
    # Wait for collection and threshold checking
    time.sleep(2)
    
    # Stop the system
    system.stop()
    assert system.running is False
    
    # Unregister the deployment
    system.unregister_deployment("test-deployment-id")
    assert "test-deployment-id" not in system.collectors
