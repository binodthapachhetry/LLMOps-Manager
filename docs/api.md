# API Reference

This document provides a detailed reference for the LLMOps Manager API endpoints.

## Authentication

All API endpoints require authentication using an API key. Include the API key in the `X-API-Key` header:

```
X-API-Key: your-api-key
```

## Rate Limiting

API endpoints are rate-limited to prevent abuse. The default limit is 60 requests per minute per client IP address.

## Endpoints

### Health Check

```
GET /health
```

Returns the health status of the API.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2023-01-01T00:00:00.000000"
}
```

### Model Management

#### Register a Model

```
POST /models/register
```

Register a model with the version manager.

**Request Body**:
```json
{
  "model_id": "string",
  "model_version": "string",
  "model_path": "string",
  "metadata": {
    "key": "value"
  }
}
```

**Response**:
```json
{
  "model_id": "string",
  "model_version": "string",
  "model_uri": "string",
  "status": "registered"
}
```

#### List Models

```
GET /models
```

List all registered models.

**Response**:
```json
{
  "models": [
    "model1",
    "model2"
  ]
}
```

#### List Model Versions

```
GET /models/{model_id}/versions
```

List all versions of a specific model.

**Response**:
```json
{
  "model_id": "string",
  "versions": [
    "v1",
    "v2"
  ]
}
```

### Deployment Management

#### Deploy a Model

```
POST /models/deploy
```

Deploy a model to the cloud provider.

**Request Body**:
```json
{
  "model_id": "string",
  "model_version": "string",
  "canary_percentage": 10.0,
  "instance_type": "string",
  "instance_count": 2,
  "auto_scaling": true,
  "environment_variables": {
    "key": "value"
  }
}
```

**Response**:
```json
{
  "deployment_id": "string",
  "model_id": "string",
  "model_version": "string",
  "status": "deploying"
}
```

#### Get Deployment Status

```
GET /deployments/{deployment_id}
```

Get the status of a deployment.

**Response**:
```json
{
  "deployment_id": "string",
  "model_id": "string",
  "model_version": "string",
  "status": "deployed",
  "current_stage": "string",
  "canary_percentage": 10.0,
  "stages": [
    {
      "name": "string",
      "status": "string",
      "start_time": "string",
      "end_time": "string"
    }
  ],
  "provider_status": {
    "status": "string"
  }
}
```

#### Update Deployment

```
PUT /deployments/{deployment_id}
```

Update an existing deployment.

**Request Body**:
```json
{
  "instance_count": 3,
  "canary_percentage": 50.0
}
```

**Response**:
```json
{
  "deployment_id": "string",
  "status": "updating"
}
```

#### Delete Deployment

```
DELETE /deployments/{deployment_id}
```

Delete a deployment.

**Response**:
```json
{
  "deployment_id": "string",
  "status": "deleted"
}
```

### Monitoring

#### Get Deployment Metrics

```
GET /deployments/{deployment_id}/metrics
```

Get metrics for a deployment.

**Query Parameters**:
- `metric_names`: Comma-separated list of metrics to retrieve
- `start_time`: Start time for metrics query (ISO format)
- `end_time`: End time for metrics query (ISO format)

**Response**:
```json
{
  "latency": [
    {
      "timestamp": "string",
      "value": 150.0
    }
  ],
  "throughput": [
    {
      "timestamp": "string",
      "value": 100.0
    }
  ]
}
```

### Scaling

#### Update Scaling Policy

```
PUT /deployments/{deployment_id}/scaling
```

Update the scaling policy for a deployment.

**Request Body**:
```json
{
  "min_replicas": 2,
  "max_replicas": 8,
  "target_gpu_utilization": 75.0,
  "max_queue_length": 100
}
```

**Response**:
```json
{
  "deployment_id": "string",
  "status": "policy_updated"
}
```

### Prompt Management

#### Register a Prompt

```
POST /prompts/register
```

Register a prompt with the version manager.

**Request Body**:
```json
{
  "prompt_id": "string",
  "prompt_version": "string",
  "prompt_text": "string",
  "metadata": {
    "key": "value"
  }
}
```

**Response**:
```json
{
  "prompt_id": "string",
  "prompt_version": "string",
  "prompt_uri": "string",
  "status": "registered"
}
```

#### List Prompts

```
GET /prompts
```

List all registered prompts.

**Response**:
```json
{
  "prompts": [
    "prompt1",
    "prompt2"
  ]
}
```

#### List Prompt Versions

```
GET /prompts/{prompt_id}/versions
```

List all versions of a specific prompt.

**Response**:
```json
{
  "prompt_id": "string",
  "versions": [
    "v1",
    "v2"
  ]
}
```

#### Get Prompt

```
GET /prompts/{prompt_id}
```

Get a prompt.

**Query Parameters**:
- `version`: Prompt version (optional, defaults to latest)

**Response**:
```json
{
  "prompt_id": "string",
  "prompt_version": "string",
  "prompt_text": "string",
  "metadata": {
    "key": "value"
  }
}
```

### Inference

#### Run Inference

```
POST /inference
```

Run inference on a model.

**Request Body**:
```json
{
  "model_id": "string",
  "model_version": "string",
  "inputs": {
    "text": "string"
  },
  "parameters": {
    "temperature": 0.7
  }
}
```

**Response**:
```json
{
  "model_id": "string",
  "model_version": "string",
  "outputs": {
    "generated_text": "string",
    "token_count": 8,
    "processing_time_ms": 150
  }
}
```

## Error Handling

All API endpoints return appropriate HTTP status codes:

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Invalid or missing API key
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

Error responses include a JSON body with an error message:

```json
{
  "error": "Error message"
}
```
