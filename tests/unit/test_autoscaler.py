"""
Unit tests for the auto-scaler.
"""
import pytest
from unittest.mock import MagicMock, patch
import time
from datetime import datetime, timedelta

from src.scaling.autoscaler import ScalingPolicy, DeploymentState, AutoScaler, RequestQueueManager


def test_scaling_policy():
    """Test the scaling policy."""
    policy = ScalingPolicy(
        min_replicas=2,
        max_replicas=10,
        target_gpu_utilization=70.0,
        scale_up_threshold=80.0,
        scale_down_threshold=30.0,
        scale_up_factor=2.0,
        scale_down_factor=0.5,
        cooldown_period_seconds=300,
        max_queue_length=100,
    )
    
    # Check initial values
    assert policy.min_replicas == 2
    assert policy.max_replicas == 10
    assert policy.target_gpu_utilization == 70.0
    assert policy.scale_up_threshold == 80.0
    assert policy.scale_down_threshold == 30.0
    assert policy.scale_up_factor == 2.0
    assert policy.scale_down_factor == 0.5
    assert policy.cooldown_period_seconds == 300
    assert policy.max_queue_length == 100
    assert policy.last_scale_time is None
    
    # Convert to dictionary
    policy_dict = policy.to_dict()
    assert policy_dict["min_replicas"] == 2
    assert policy_dict["max_replicas"] == 10
    assert policy_dict["target_gpu_utilization"] == 70.0
    assert policy_dict["scale_up_threshold"] == 80.0
    assert policy_dict["scale_down_threshold"] == 30.0
    assert policy_dict["scale_up_factor"] == 2.0
    assert policy_dict["scale_down_factor"] == 0.5
    assert policy_dict["cooldown_period_seconds"] == 300
    assert policy_dict["max_queue_length"] == 100


def test_deployment_state():
    """Test the deployment state."""
    state = DeploymentState(
        deployment_id="test-deployment-id",
        initial_replicas=2,
    )
    
    # Check initial values
    assert state.deployment_id == "test-deployment-id"
    assert state.current_replicas == 2
    assert state.desired_replicas == 2
    assert state.last_scaling_time is None
    assert len(state.metrics_history["gpu_utilization"]) == 0
    assert len(state.scaling_history) == 0
    
    # Update metrics
    state.update_metrics({
        "gpu_utilization": 75.0,
        "queue_length": 50,
        "throughput": 100.0,
        "latency": 200.0,
    })
    
    assert len(state.metrics_history["gpu_utilization"]) == 1
    assert state.metrics_history["gpu_utilization"][0]["value"] == 75.0
    
    # Record scaling action
    state.record_scaling_action(
        old_replicas=2,
        new_replicas=4,
        reason="High GPU utilization",
    )
    
    assert state.current_replicas == 4
    assert state.last_scaling_time is not None
    assert len(state.scaling_history) == 1
    assert state.scaling_history[0]["old_replicas"] == 2
    assert state.scaling_history[0]["new_replicas"] == 4
    assert state.scaling_history[0]["reason"] == "High GPU utilization"
    
    # Get recent metrics
    recent_metrics = state.get_recent_metrics("gpu_utilization", window_seconds=3600)
    assert len(recent_metrics) == 1
    assert recent_metrics[0]["value"] == 75.0
    
    # Get average metric
    avg_gpu = state.get_average_metric("gpu_utilization", window_seconds=3600)
    assert avg_gpu == 75.0
    
    # Check cooldown
    assert state.in_cooldown(cooldown_seconds=3600) is True
    assert state.in_cooldown(cooldown_seconds=0) is False


def test_auto_scaler():
    """Test the auto-scaler."""
    provider = MagicMock()
    provider.get_deployment_status.return_value = {"instance_count": 2}
    
    config = {
        "check_interval_seconds": 1,  # Short interval for testing
        "default_min_replicas": 1,
        "default_max_replicas": 10,
        "default_target_gpu_utilization": 70.0,
    }
    
    scaler = AutoScaler(provider, config)
    
    # Initial state
    assert scaler.provider == provider
    assert scaler.config == config
    assert len(scaler.deployments) == 0
    assert len(scaler.policies) == 0
    
    # Register a deployment
    scaler.register_deployment(
        deployment_id="test-deployment-id",
        initial_replicas=2,
    )
    
    assert "test-deployment-id" in scaler.deployments
    assert "test-deployment-id" in scaler.policies
    assert scaler.deployments["test-deployment-id"].current_replicas == 2
    assert scaler.policies["test-deployment-id"].min_replicas == 1
    assert scaler.policies["test-deployment-id"].max_replicas == 10
    
    # Update policy
    scaler.update_policy(
        deployment_id="test-deployment-id",
        min_replicas=2,
        max_replicas=8,
        target_gpu_utilization=75.0,
    )
    
    assert scaler.policies["test-deployment-id"].min_replicas == 2
    assert scaler.policies["test-deployment-id"].max_replicas == 8
    assert scaler.policies["test-deployment-id"].target_gpu_utilization == 75.0
    
    # Get policy
    policy = scaler.get_policy("test-deployment-id")
    assert policy["min_replicas"] == 2
    assert policy["max_replicas"] == 8
    assert policy["target_gpu_utilization"] == 75.0
    
    # Mock _get_deployment_metrics to return high GPU utilization
    scaler._get_deployment_metrics = MagicMock(return_value={
        "gpu_utilization": 90.0,
        "queue_length": 50,
        "throughput": 100.0,
        "latency": 200.0,
    })
    
    # Start the scaler
    scaler.start()
    
    # Wait for scaling check
    time.sleep(2)
    
    # Stop the scaler
    scaler.stop()
    
    # Check scaling history
    history = scaler.get_scaling_history("test-deployment-id")
    assert len(history) > 0
    
    # Unregister the deployment
    scaler.unregister_deployment("test-deployment-id")
    assert "test-deployment-id" not in scaler.deployments
    assert "test-deployment-id" not in scaler.policies


def test_request_queue_manager():
    """Test the request queue manager."""
    config = {
        "max_queue_size": 10,
        "default_timeout_seconds": 1,
    }
    
    manager = RequestQueueManager(config)
    
    # Initial state
    assert manager.max_queue_size == 10
    assert manager.default_timeout == 1
    assert len(manager.queues) == 0
    
    # Create a queue
    manager.create_queue("test-deployment-id")
    assert "test-deployment-id" in manager.queues
    
    # Enqueue requests
    for i in range(5):
        success = manager.enqueue_request(
            deployment_id="test-deployment-id",
            request={"id": f"request_{i}"},
        )
        assert success is True
    
    # Check queue length
    assert manager.get_queue_length("test-deployment-id") == 5
    assert manager.is_queue_full("test-deployment-id") is False
    
    # Dequeue requests
    for i in range(5):
        request = manager.dequeue_request("test-deployment-id")
        assert request is not None
        assert request["id"] == f"request_{i}"
    
    # Queue should be empty now
    assert manager.get_queue_length("test-deployment-id") == 0
    
    # Dequeue from empty queue should return None
    request = manager.dequeue_request("test-deployment-id", timeout=0.1)
    assert request is None
    
    # Delete the queue
    manager.delete_queue("test-deployment-id")
    assert "test-deployment-id" not in manager.queues
