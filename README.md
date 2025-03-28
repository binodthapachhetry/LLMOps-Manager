# LLMOps Manager

A comprehensive toolkit for deploying, monitoring, scaling, and managing large language models in production environments across multiple cloud providers (AWS, GCP, Azure).

## Overview

LLMOps Manager provides a unified interface for managing the entire lifecycle of LLM deployments, from model registration to deployment, monitoring, scaling, and versioning. It abstracts away the complexities of different cloud providers, allowing you to focus on your models rather than infrastructure.

## Key Features

- **Multi-Cloud Support**: Deploy and manage models on AWS, GCP, and Azure with a unified API
- **Canary Deployments**: Safely roll out new model versions with configurable traffic splitting
- **Real-Time Monitoring**: Track latency, throughput, token usage, and custom metrics
- **Drift Detection**: Monitor for input and output drift in your models
- **Auto-Scaling**: Automatically scale based on GPU utilization and request queues
- **Version Control**: Manage model artifacts and prompts with versioning
- **API-First Design**: RESTful API for integration with existing systems
- **Security**: Built-in authentication and rate limiting
- **CI/CD Pipeline**: Automated testing, building, and deployment with GitHub Actions

## Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (for local development)
- Access to at least one cloud provider (AWS, GCP, or Azure)
- Git (for version control and pre-commit hooks)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/llmops-manager.git
   cd llmops-manager
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

4. Configure your cloud provider credentials according to their respective SDKs:
   - AWS: Configure AWS credentials via AWS CLI or environment variables
   - GCP: Set up application default credentials or provide a service account key
   - Azure: Configure Azure credentials via Azure CLI or environment variables

5. Run the application:
   ```bash
   uvicorn src.api.endpoints:app --host 0.0.0.0 --port 8000
   ```

### Using Docker

You can also run the application using Docker:

```bash
docker-compose up -d
```

This will start the LLMOps Manager along with supporting services like Prometheus and Grafana for monitoring.

## Usage Examples

### Register a Model

```bash
curl -X POST "http://localhost:8000/models/register" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "gpt-model",
    "model_version": "v1",
    "model_path": "/path/to/model",
    "metadata": {"framework": "pytorch"}
  }'
```

### Deploy a Model

```bash
curl -X POST "http://localhost:8000/models/deploy" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "gpt-model",
    "model_version": "v1",
    "canary_percentage": 10.0,
    "instance_type": "ml.g4dn.xlarge",
    "instance_count": 2
  }'
```

### Get Metrics

```bash
curl -X GET "http://localhost:8000/deployments/deployment-id/metrics?metric_names=latency,throughput" \
  -H "X-API-Key: your-api-key"
```

## Documentation

For more detailed documentation, see:

- [Architecture Overview](ARCHITECTURE.md)
- [API Reference](docs/api.md)
- [Configuration Guide](docs/configuration.md)
- [Deployment Guide](docs/deployment.md)

## Development

### Code Quality

This project uses several tools to ensure code quality:

- **Black**: For code formatting
- **isort**: For import sorting
- **flake8**: For code linting
- **pre-commit**: To run these checks automatically before each commit

### CI/CD Pipeline

The project includes a GitHub Actions workflow that:

1. Runs automated tests on each push and pull request
2. Builds the Docker image
3. (Future) Deploys to staging/production environments

To view the CI pipeline, check the Actions tab in the GitHub repository.

### Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src

# Run only unit tests
pytest -m unit

# Run only tests marked for CI
pytest -m ci
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
