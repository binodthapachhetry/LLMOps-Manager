# Deployment Guide

This document provides instructions for deploying the LLMOps Manager in various environments.

## Local Development

For local development, you can run the LLMOps Manager directly:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a configuration file:
   ```bash
   mkdir -p config
   cp config/llmops.yaml.example config/llmops.yaml
   # Edit config/llmops.yaml with your settings
   ```

3. Run the application:
   ```bash
   uvicorn src.api.endpoints:app --host 0.0.0.0 --port 8000 --reload
   ```

## Docker Deployment

For a more isolated environment, you can use Docker:

1. Build the Docker image:
   ```bash
   docker build -t llmops-manager:latest .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 -v $(pwd)/config:/app/config llmops-manager:latest
   ```

## Docker Compose Deployment

For a complete local environment with supporting services:

1. Create a configuration file:
   ```bash
   mkdir -p config
   cp config/llmops.yaml.example config/llmops.yaml
   # Edit config/llmops.yaml with your settings
   ```

2. Start the services:
   ```bash
   docker-compose up -d
   ```

This will start:
- LLMOps Manager API
- Prometheus for metrics collection
- Grafana for metrics visualization
- Mock AWS services (LocalStack)
- PostgreSQL for metadata storage
- Redis for caching and queue management

## Kubernetes Deployment

For production deployments, Kubernetes is recommended:

1. Build and push the Docker image:
   ```bash
   docker build -t your-registry/llmops-manager:latest .
   docker push your-registry/llmops-manager:latest
   ```

2. Apply the Kubernetes manifests:
   ```bash
   kubectl apply -f kubernetes/namespace.yaml
   kubectl apply -f kubernetes/configmap.yaml
   kubectl apply -f kubernetes/secret.yaml
   kubectl apply -f kubernetes/deployment.yaml
   kubectl apply -f kubernetes/service.yaml
   ```

3. For monitoring, install Prometheus and Grafana:
   ```bash
   kubectl apply -f kubernetes/monitoring/
   ```

## Cloud Provider Deployments

### AWS

1. Use the provided Terraform configuration:
   ```bash
   cd infrastructure/terraform
   terraform init
   terraform apply -var="aws_region=us-west-2" -var="environment=prod" -var="app_name=llmops-manager"
   ```

2. Build and push the Docker image to ECR:
   ```bash
   aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-west-2.amazonaws.com
   docker build -t $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-west-2.amazonaws.com/llmops-manager:latest .
   docker push $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-west-2.amazonaws.com/llmops-manager:latest
   ```

### GCP

1. Use the provided Terraform configuration:
   ```bash
   cd infrastructure/terraform
   terraform init
   terraform apply -var="gcp_project_id=your-project-id" -var="gcp_region=us-central1" -var="environment=prod" -var="app_name=llmops-manager"
   ```

2. Build and push the Docker image to GCR:
   ```bash
   gcloud auth configure-docker
   docker build -t gcr.io/your-project-id/llmops-manager:latest .
   docker push gcr.io/your-project-id/llmops-manager:latest
   ```

### Azure

1. Use the provided Terraform configuration:
   ```bash
   cd infrastructure/terraform
   terraform init
   terraform apply -var="azure_subscription_id=your-subscription-id" -var="environment=prod" -var="app_name=llmops-manager"
   ```

2. Build and push the Docker image to ACR:
   ```bash
   az acr login --name yourregistry
   docker build -t yourregistry.azurecr.io/llmops-manager:latest .
   docker push yourregistry.azurecr.io/llmops-manager:latest
   ```

## Environment Variables

Set the following environment variables for deployment:

- `LLMOPS_CONFIG_PATH`: Path to the configuration file
- `LLMOPS_PROVIDER`: Cloud provider to use (aws, gcp, or azure)
- `LLMOPS_API_KEY`: API key for authentication
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Security Considerations

1. **API Key**: Generate a strong API key and keep it secure
2. **Network Security**: Use a firewall to restrict access to the API
3. **TLS**: Enable HTTPS for all communications
4. **IAM Roles**: Use least-privilege IAM roles for cloud provider access
5. **Secrets Management**: Use a secure method for managing secrets (e.g., Kubernetes Secrets, AWS Secrets Manager)

## Monitoring and Logging

1. **Prometheus**: Access the Prometheus UI at `http://your-host:9090`
2. **Grafana**: Access the Grafana UI at `http://your-host:3000` (default credentials: admin/admin)
3. **Logs**: Logs are written to stdout/stderr and can be collected by your logging infrastructure

## Scaling

The LLMOps Manager can be scaled horizontally by increasing the number of replicas:

```bash
kubectl scale deployment llmops-manager --replicas=3
```

For automatic scaling, use a Horizontal Pod Autoscaler:

```bash
kubectl autoscale deployment llmops-manager --cpu-percent=70 --min=2 --max=10
```

## Backup and Disaster Recovery

1. **Configuration**: Back up your configuration files
2. **Database**: Set up regular backups for PostgreSQL
3. **Model Artifacts**: Model artifacts are stored in cloud storage, which has built-in redundancy
4. **Stateless Design**: The LLMOps Manager is designed to be stateless, making recovery simpler

## Troubleshooting

1. **Check Logs**: `kubectl logs deployment/llmops-manager`
2. **Check API Health**: `curl http://your-host:8000/health`
3. **Check Metrics**: `curl http://your-host:8000/metrics`
4. **Check Cloud Provider Status**: Verify that your cloud provider services are operational
