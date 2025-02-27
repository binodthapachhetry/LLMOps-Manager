# Configuration Guide

This document explains how to configure the LLMOps Manager for your environment.

## Configuration File

LLMOps Manager uses a YAML configuration file to define its behavior. By default, it looks for this file at `config/llmops.yaml`, but you can specify a different path using the `LLMOPS_CONFIG_PATH` environment variable.

## Configuration Structure

The configuration file has the following structure:

```yaml
aws:
  region: us-west-2
  role_arn: arn:aws:iam::123456789012:role/LLMOpsManagerRole

gcp:
  project_id: llmops-project
  region: us-central1

azure:
  subscription_id: subscription-id
  resource_group: llmops-resource-group
  workspace_name: llmops-workspace

deployment:
  default_instance_type: ml.g4dn.xlarge
  default_instance_count: 1
  canary_evaluation_duration: 600
  validation_tests:
    - name: basic_inference
      inputs:
        text: "Hello, world!"
      expected_outputs:
        contains: "Hello"

monitoring:
  collectors:
    collection_interval_seconds: 60
    enabled_metrics:
      - latency
      - throughput
      - token_usage
      - error_rate
      - gpu_utilization
    drift_detector:
      window_size: 1000
      drift_threshold: 0.05
  alerts:
    thresholds:
      latency:
        warning: 500
        error: 1000
        critical: 2000
        direction: above
      error_rate:
        warning: 0.01
        error: 0.05
        critical: 0.1
        direction: above
      gpu_utilization:
        warning: 85
        error: 95
        direction: above
    email_alerts:
      enabled: false
      recipients:
        - alerts@example.com
    webhook_alerts:
      enabled: false
      url: https://example.com/webhook

scaling:
  check_interval_seconds: 60
  default_min_replicas: 1
  default_max_replicas: 10
  default_target_gpu_utilization: 70.0

versioning:
  model_registry:
    storage_prefix: models
    local_cache_dir: model_cache
  prompt_registry:
    storage_prefix: prompts
    local_cache_dir: prompt_cache
```

## Configuration Sections

### AWS Configuration

```yaml
aws:
  region: us-west-2
  role_arn: arn:aws:iam::123456789012:role/LLMOpsManagerRole
```

- `region`: AWS region to use for deployments
- `role_arn`: IAM role ARN with permissions for SageMaker, S3, and CloudWatch

### GCP Configuration

```yaml
gcp:
  project_id: llmops-project
  region: us-central1
```

- `project_id`: Google Cloud project ID
- `region`: Google Cloud region to use for deployments

### Azure Configuration

```yaml
azure:
  subscription_id: subscription-id
  resource_group: llmops-resource-group
  workspace_name: llmops-workspace
```

- `subscription_id`: Azure subscription ID
- `resource_group`: Azure resource group name
- `workspace_name`: Azure ML workspace name

### Deployment Configuration

```yaml
deployment:
  default_instance_type: ml.g4dn.xlarge
  default_instance_count: 1
  canary_evaluation_duration: 600
  validation_tests:
    - name: basic_inference
      inputs:
        text: "Hello, world!"
      expected_outputs:
        contains: "Hello"
```

- `default_instance_type`: Default instance type for deployments
- `default_instance_count`: Default number of instances for deployments
- `canary_evaluation_duration`: Duration in seconds to evaluate canary deployments
- `validation_tests`: Tests to run during deployment validation

### Monitoring Configuration

```yaml
monitoring:
  collectors:
    collection_interval_seconds: 60
    enabled_metrics:
      - latency
      - throughput
      - token_usage
      - error_rate
      - gpu_utilization
    drift_detector:
      window_size: 1000
      drift_threshold: 0.05
  alerts:
    thresholds:
      latency:
        warning: 500
        error: 1000
        critical: 2000
        direction: above
      error_rate:
        warning: 0.01
        error: 0.05
        critical: 0.1
        direction: above
      gpu_utilization:
        warning: 85
        error: 95
        direction: above
    email_alerts:
      enabled: false
      recipients:
        - alerts@example.com
    webhook_alerts:
      enabled: false
      url: https://example.com/webhook
```

- `collectors`: Configuration for metric collectors
  - `collection_interval_seconds`: How often to collect metrics
  - `enabled_metrics`: List of metrics to collect
  - `drift_detector`: Configuration for drift detection
- `alerts`: Configuration for alerting
  - `thresholds`: Metric thresholds for alerts
  - `email_alerts`: Email alert configuration
  - `webhook_alerts`: Webhook alert configuration

### Scaling Configuration

```yaml
scaling:
  check_interval_seconds: 60
  default_min_replicas: 1
  default_max_replicas: 10
  default_target_gpu_utilization: 70.0
```

- `check_interval_seconds`: How often to check scaling conditions
- `default_min_replicas`: Default minimum number of replicas
- `default_max_replicas`: Default maximum number of replicas
- `default_target_gpu_utilization`: Default target GPU utilization percentage

### Versioning Configuration

```yaml
versioning:
  model_registry:
    storage_prefix: models
    local_cache_dir: model_cache
  prompt_registry:
    storage_prefix: prompts
    local_cache_dir: prompt_cache
```

- `model_registry`: Configuration for the model registry
  - `storage_prefix`: Prefix for model storage paths
  - `local_cache_dir`: Directory for local model caching
- `prompt_registry`: Configuration for the prompt registry
  - `storage_prefix`: Prefix for prompt storage paths
  - `local_cache_dir`: Directory for local prompt caching

## Environment Variables

In addition to the configuration file, LLMOps Manager supports the following environment variables:

- `LLMOPS_CONFIG_PATH`: Path to the configuration file
- `LLMOPS_PROVIDER`: Default cloud provider to use (aws, gcp, or azure)
- `LLMOPS_API_KEY`: API key for authentication
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Provider Credentials

LLMOps Manager uses the standard credential mechanisms for each cloud provider:

### AWS

- AWS credentials file (`~/.aws/credentials`)
- Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- IAM roles for EC2/EKS

### GCP

- Application Default Credentials
- Service account key file (set `GOOGLE_APPLICATION_CREDENTIALS`)

### Azure

- Azure CLI authentication
- Environment variables (`AZURE_SUBSCRIPTION_ID`, `AZURE_TENANT_ID`, etc.)
- Managed Identity

## Example Configuration

Here's an example configuration for AWS:

```yaml
aws:
  region: us-west-2
  role_arn: arn:aws:iam::123456789012:role/LLMOpsManagerRole

deployment:
  default_instance_type: ml.g4dn.xlarge
  default_instance_count: 2
  canary_evaluation_duration: 300

monitoring:
  collectors:
    collection_interval_seconds: 30
    enabled_metrics:
      - latency
      - throughput
      - error_rate
      - gpu_utilization
  alerts:
    thresholds:
      latency:
        warning: 200
        error: 500
        direction: above
      error_rate:
        warning: 0.01
        error: 0.05
        direction: above

scaling:
  check_interval_seconds: 30
  default_min_replicas: 2
  default_max_replicas: 8
  default_target_gpu_utilization: 75.0

versioning:
  model_registry:
    storage_prefix: models
    local_cache_dir: model_cache
  prompt_registry:
    storage_prefix: prompts
    local_cache_dir: prompt_cache
```
