"""
Unit tests for the deployment pipeline.
"""
import pytest
from unittest.mock import MagicMock, patch
import time
from datetime import datetime

from src.deployment.pipeline import DeploymentStage, DeploymentPipeline


def test_deployment_stage_lifecycle():
    """Test the lifecycle of a deployment stage."""
    stage = DeploymentStage(
        name="test_stage",
        config={"test_param": "test_value"},
    )
    
    # Initial state
    assert stage.name == "test_stage"
    assert stage.config == {"test_param": "test_value"}
    assert stage.status == "pending"
    assert stage.start_time is None
    assert stage.end_time is None
    assert len(stage.logs) == 0
    
    # Start the stage
    stage.start()
    assert stage.status == "running"
    assert stage.start_time is not None
    assert stage.end_time is None
    assert len(stage.logs) == 1
    assert "Stage test_stage started" in stage.logs[0]
    
    # Add a log message
    stage.log("Test message")
    assert len(stage.logs) == 2
    assert "Test message" in stage.logs[1]
    
    # Complete the stage successfully
    stage.complete(success=True)
    assert stage.status == "succeeded"
    assert stage.end_time is not None
    assert len(stage.logs) == 3
    assert "Stage test_stage succeeded" in stage.logs[2]
    
    # Create a new stage and complete it with failure
    stage = DeploymentStage(
        name="failed_stage",
        config={},
    )
    stage.start()
    stage.complete(success=False)
    assert stage.status == "failed"
    assert "Stage failed_stage failed" in stage.logs[1]


def test_deployment_pipeline_initialization():
    """Test initialization of the deployment pipeline."""
    provider = MagicMock()
    config = {
        "default_instance_type": "ml.g4dn.xlarge",
        "default_instance_count": 1,
        "canary_evaluation_duration": 600,
    }
    
    pipeline = DeploymentPipeline(provider, config)
    
    assert pipeline.provider == provider
    assert pipeline.config == config
    assert pipeline.deployments == {}


def test_deployment_pipeline_deploy():
    """Test deploying a model through the pipeline."""
    provider = MagicMock()
    provider.deploy_model.return_value = "test-deployment-id"
    
    config = {
        "default_instance_type": "ml.g4dn.xlarge",
        "default_instance_count": 1,
        "canary_evaluation_duration": 10,  # Short duration for testing
        "canary_success_criteria": {},
        "validation_tests": [],
    }
    
    pipeline = DeploymentPipeline(provider, config)
    
    # Mock the execution methods to avoid actual execution
    pipeline._execute_prepare_artifacts = MagicMock()
    pipeline._execute_validate_model = MagicMock()
    pipeline._execute_deploy_canary = MagicMock()
    pipeline._execute_evaluate_canary = MagicMock()
    pipeline._execute_deploy_full = MagicMock()
    pipeline._execute_post_deployment_validation = MagicMock()
    
    # Deploy with canary
    deployment_id = pipeline.deploy(
        model_id="test-model",
        model_version="v1",
        canary_percentage=10.0,
        config={"instance_type": "ml.g4dn.2xlarge"},
    )
    
    assert deployment_id in pipeline.deployments
    assert pipeline.deployments[deployment_id]["model_id"] == "test-model"
    assert pipeline.deployments[deployment_id]["model_version"] == "v1"
    assert pipeline.deployments[deployment_id]["canary_percentage"] == 10.0
    assert pipeline.deployments[deployment_id]["config"]["instance_type"] == "ml.g4dn.2xlarge"
    assert len(pipeline.deployments[deployment_id]["stages"]) > 0


def test_deployment_pipeline_update_canary():
    """Test updating a canary deployment."""
    provider = MagicMock()
    config = {}
    
    pipeline = DeploymentPipeline(provider, config)
    
    # Create a mock deployment
    deployment_id = "test-deployment-id"
    pipeline.deployments[deployment_id] = {
        "model_id": "test-model",
        "model_version": "v1",
        "canary_percentage": 10.0,
        "status": "deploying",
        "create_time": datetime.now().isoformat(),
        "stages": [],
        "current_stage": 0,
        "canary_deployment_id": "canary-id",
    }
    
    # Update the canary percentage
    pipeline.update_canary(deployment_id, 50.0)
    
    assert pipeline.deployments[deployment_id]["canary_percentage"] == 50.0
    provider.update_deployment.assert_called_once_with(
        deployment_id="canary-id",
        traffic_percentage=50.0,
    )
    
    # Update to 100% should mark as fully deployed
    pipeline.update_canary(deployment_id, 100.0)
    assert pipeline.deployments[deployment_id]["status"] == "deployed"


def test_deployment_pipeline_rollback():
    """Test rolling back a deployment."""
    provider = MagicMock()
    config = {}
    
    pipeline = DeploymentPipeline(provider, config)
    
    # Create a mock deployment
    deployment_id = "test-deployment-id"
    pipeline.deployments[deployment_id] = {
        "model_id": "test-model",
        "model_version": "v1",
        "status": "deployed",
        "create_time": datetime.now().isoformat(),
        "stages": [],
        "current_stage": 0,
        "canary_deployment_id": "canary-id",
        "full_deployment_id": "full-id",
    }
    
    # Rollback the deployment
    pipeline.rollback(deployment_id)
    
    assert pipeline.deployments[deployment_id]["status"] == "rolled_back"
    assert "rollback_time" in pipeline.deployments[deployment_id]
    provider.delete_deployment.assert_called_with("full-id")


def test_deployment_pipeline_get_status():
    """Test getting deployment status."""
    provider = MagicMock()
    provider.get_deployment_status.return_value = {"status": "InService"}
    config = {}
    
    pipeline = DeploymentPipeline(provider, config)
    
    # Create a mock deployment with stages
    deployment_id = "test-deployment-id"
    stage = DeploymentStage(name="test_stage", config={})
    stage.start()
    stage.complete(success=True)
    
    pipeline.deployments[deployment_id] = {
        "model_id": "test-model",
        "model_version": "v1",
        "status": "deployed",
        "create_time": datetime.now().isoformat(),
        "stages": [stage],
        "current_stage": 0,
        "full_deployment_id": "full-id",
    }
    
    # Get the status
    status = pipeline.get_deployment_status(deployment_id)
    
    assert status["deployment_id"] == deployment_id
    assert status["model_id"] == "test-model"
    assert status["model_version"] == "v1"
    assert status["status"] == "deployed"
    assert status["current_stage"] == "test_stage"
    assert "provider_status" in status
    assert len(status["stages"]) == 1
    assert status["stages"][0]["name"] == "test_stage"
    assert status["stages"][0]["status"] == "succeeded"


def test_deployment_pipeline_error_handling():
    """Test error handling in the deployment pipeline."""
    provider = MagicMock()
    config = {}
    
    pipeline = DeploymentPipeline(provider, config)
    
    # Mock the execution methods to simulate an error
    pipeline._execute_prepare_artifacts = MagicMock()
    pipeline._execute_validate_model = MagicMock(side_effect=Exception("Test error"))
    
    # Create stages for the deployment
    stages = [
        DeploymentStage(name="prepare_artifacts", config={}),
        DeploymentStage(name="validate_model", config={}),
    ]
    
    # Create a mock deployment
    deployment_id = "test-deployment-id"
    pipeline.deployments[deployment_id] = {
        "model_id": "test-model",
        "model_version": "v1",
        "status": "deploying",
        "create_time": datetime.now().isoformat(),
        "stages": stages,
        "current_stage": 0,
    }
    
    # Execute the pipeline
    pipeline._execute_pipeline(deployment_id)
    
    # Check that the deployment is marked as failed
    assert pipeline.deployments[deployment_id]["status"] == "failed"
    assert stages[1].status == "failed"
    assert any("Error: Test error" in log for log in stages[1].logs)
