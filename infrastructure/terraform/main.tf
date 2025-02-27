# LLMOps Manager Infrastructure as Code (Terraform)

# Provider configuration
provider "aws" {
  region = var.aws_region
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

provider "azurerm" {
  features {}
  subscription_id = var.azure_subscription_id
}

# Variables
variable "aws_region" {
  description = "AWS region"
  default     = "us-west-2"
}

variable "gcp_project_id" {
  description = "GCP project ID"
  default     = "llmops-project"
}

variable "gcp_region" {
  description = "GCP region"
  default     = "us-central1"
}

variable "azure_subscription_id" {
  description = "Azure subscription ID"
  default     = "subscription-id"
}

variable "environment" {
  description = "Deployment environment"
  default     = "dev"
}

variable "app_name" {
  description = "Application name"
  default     = "llmops-manager"
}

# AWS Resources
# -------------

# S3 bucket for model artifacts
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "${var.app_name}-model-artifacts-${var.environment}"

  tags = {
    Name        = "${var.app_name}-model-artifacts"
    Environment = var.environment
  }
}

# S3 bucket for monitoring data
resource "aws_s3_bucket" "monitoring_data" {
  bucket = "${var.app_name}-monitoring-data-${var.environment}"

  tags = {
    Name        = "${var.app_name}-monitoring-data"
    Environment = var.environment
  }
}

# IAM role for LLMOps Manager
resource "aws_iam_role" "llmops_manager_role" {
  name = "${var.app_name}-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "${var.app_name}-role"
    Environment = var.environment
  }
}

# IAM policy for LLMOps Manager
resource "aws_iam_policy" "llmops_manager_policy" {
  name        = "${var.app_name}-policy-${var.environment}"
  description = "Policy for LLMOps Manager"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Effect   = "Allow"
        Resource = [
          aws_s3_bucket.model_artifacts.arn,
          "${aws_s3_bucket.model_artifacts.arn}/*",
          aws_s3_bucket.monitoring_data.arn,
          "${aws_s3_bucket.monitoring_data.arn}/*"
        ]
      },
      {
        Action = [
          "sagemaker:CreateModel",
          "sagemaker:CreateEndpoint",
          "sagemaker:CreateEndpointConfig",
          "sagemaker:UpdateEndpoint",
          "sagemaker:DeleteModel",
          "sagemaker:DeleteEndpoint",
          "sagemaker:DeleteEndpointConfig",
          "sagemaker:DescribeEndpoint",
          "sagemaker:InvokeEndpoint"
        ]
        Effect   = "Allow"
        Resource = "*"
      },
      {
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricData",
          "cloudwatch:ListMetrics"
        ]
        Effect   = "Allow"
        Resource = "*"
      },
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
  })
}

# Attach policy to role
resource "aws_iam_role_policy_attachment" "llmops_manager_policy_attachment" {
  role       = aws_iam_role.llmops_manager_role.name
  policy_arn = aws_iam_policy.llmops_manager_policy.arn
}

# ECS cluster for LLMOps Manager
resource "aws_ecs_cluster" "llmops_manager_cluster" {
  name = "${var.app_name}-cluster-${var.environment}"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name        = "${var.app_name}-cluster"
    Environment = var.environment
  }
}

# ECR repository for LLMOps Manager
resource "aws_ecr_repository" "llmops_manager_repo" {
  name = "${var.app_name}-repo-${var.environment}"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name        = "${var.app_name}-repo"
    Environment = var.environment
  }
}

# Security group for LLMOps Manager
resource "aws_security_group" "llmops_manager_sg" {
  name        = "${var.app_name}-sg-${var.environment}"
  description = "Security group for LLMOps Manager"

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.app_name}-sg"
    Environment = var.environment
  }
}

# GCP Resources
# -------------

# GCS bucket for model artifacts
resource "google_storage_bucket" "model_artifacts" {
  name     = "${var.app_name}-model-artifacts-${var.environment}"
  location = var.gcp_region

  uniform_bucket_level_access = true

  labels = {
    environment = var.environment
  }
}

# GCS bucket for monitoring data
resource "google_storage_bucket" "monitoring_data" {
  name     = "${var.app_name}-monitoring-data-${var.environment}"
  location = var.gcp_region

  uniform_bucket_level_access = true

  labels = {
    environment = var.environment
  }
}

# Service account for LLMOps Manager
resource "google_service_account" "llmops_manager_sa" {
  account_id   = "${var.app_name}-sa-${var.environment}"
  display_name = "LLMOps Manager Service Account"
}

# IAM binding for GCS buckets
resource "google_storage_bucket_iam_binding" "model_artifacts_binding" {
  bucket = google_storage_bucket.model_artifacts.name
  role   = "roles/storage.objectAdmin"

  members = [
    "serviceAccount:${google_service_account.llmops_manager_sa.email}",
  ]
}

resource "google_storage_bucket_iam_binding" "monitoring_data_binding" {
  bucket = google_storage_bucket.monitoring_data.name
  role   = "roles/storage.objectAdmin"

  members = [
    "serviceAccount:${google_service_account.llmops_manager_sa.email}",
  ]
}

# IAM binding for Vertex AI
resource "google_project_iam_binding" "vertex_ai_binding" {
  project = var.gcp_project_id
  role    = "roles/aiplatform.user"

  members = [
    "serviceAccount:${google_service_account.llmops_manager_sa.email}",
  ]
}

# Cloud Run service for LLMOps Manager
resource "google_cloud_run_service" "llmops_manager" {
  name     = "${var.app_name}-service-${var.environment}"
  location = var.gcp_region

  template {
    spec {
      containers {
        image = "gcr.io/${var.gcp_project_id}/${var.app_name}:latest"
        
        env {
          name  = "LLMOPS_PROVIDER"
          value = "gcp"
        }
        
        resources {
          limits = {
            cpu    = "1000m"
            memory = "2Gi"
          }
        }
      }
      
      service_account_name = google_service_account.llmops_manager_sa.email
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# Azure Resources
# --------------

# Resource group
resource "azurerm_resource_group" "llmops_manager_rg" {
  name     = "${var.app_name}-rg-${var.environment}"
  location = "West US 2"
}

# Storage account for model artifacts
resource "azurerm_storage_account" "model_artifacts" {
  name                     = "${replace(var.app_name, "-", "")}artifacts${var.environment}"
  resource_group_name      = azurerm_resource_group.llmops_manager_rg.name
  location                 = azurerm_resource_group.llmops_manager_rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"

  tags = {
    environment = var.environment
  }
}

# Storage container for model artifacts
resource "azurerm_storage_container" "model_artifacts" {
  name                  = "model-artifacts"
  storage_account_name  = azurerm_storage_account.model_artifacts.name
  container_access_type = "private"
}

# Storage container for monitoring data
resource "azurerm_storage_container" "monitoring_data" {
  name                  = "monitoring-data"
  storage_account_name  = azurerm_storage_account.model_artifacts.name
  container_access_type = "private"
}

# Azure Container Registry
resource "azurerm_container_registry" "llmops_manager_acr" {
  name                = "${replace(var.app_name, "-", "")}acr${var.environment}"
  resource_group_name = azurerm_resource_group.llmops_manager_rg.name
  location            = azurerm_resource_group.llmops_manager_rg.location
  sku                 = "Standard"
  admin_enabled       = true

  tags = {
    environment = var.environment
  }
}

# Azure Kubernetes Service
resource "azurerm_kubernetes_cluster" "llmops_manager_aks" {
  name                = "${var.app_name}-aks-${var.environment}"
  location            = azurerm_resource_group.llmops_manager_rg.location
  resource_group_name = azurerm_resource_group.llmops_manager_rg.name
  dns_prefix          = "${var.app_name}-aks-${var.environment}"

  default_node_pool {
    name       = "default"
    node_count = 2
    vm_size    = "Standard_DS2_v2"
  }

  identity {
    type = "SystemAssigned"
  }

  tags = {
    environment = var.environment
  }
}

# Kubernetes Resources (for all providers)
# ---------------------------------------

# Kubernetes provider configuration
provider "kubernetes" {
  # Configuration depends on which cloud provider is being used
  # This would be configured dynamically in a real implementation
}

# Kubernetes namespace
resource "kubernetes_namespace" "llmops_manager" {
  metadata {
    name = "${var.app_name}-${var.environment}"
  }
}

# Kubernetes deployment
resource "kubernetes_deployment" "llmops_manager" {
  metadata {
    name      = var.app_name
    namespace = kubernetes_namespace.llmops_manager.metadata[0].name
    labels = {
      app         = var.app_name
      environment = var.environment
    }
  }

  spec {
    replicas = 2

    selector {
      match_labels = {
        app = var.app_name
      }
    }

    template {
      metadata {
        labels = {
          app         = var.app_name
          environment = var.environment
        }
      }

      spec {
        container {
          image = "${aws_ecr_repository.llmops_manager_repo.repository_url}:latest"
          name  = var.app_name

          port {
            container_port = 8000
          }

          env {
            name  = "LLMOPS_CONFIG_PATH"
            value = "/app/config/llmops.yaml"
          }

          resources {
            limits = {
              cpu    = "1"
              memory = "2Gi"
            }
            requests = {
              cpu    = "500m"
              memory = "1Gi"
            }
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 30
            period_seconds        = 10
          }

          readiness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 5
            period_seconds        = 5
          }
        }
      }
    }
  }
}

# Kubernetes service
resource "kubernetes_service" "llmops_manager" {
  metadata {
    name      = var.app_name
    namespace = kubernetes_namespace.llmops_manager.metadata[0].name
  }

  spec {
    selector = {
      app = kubernetes_deployment.llmops_manager.metadata[0].labels.app
    }

    port {
      port        = 80
      target_port = 8000
    }

    type = "LoadBalancer"
  }
}

# Monitoring Resources
# ------------------

# Prometheus and Grafana would be deployed using Helm charts in a real implementation
# This is a simplified example

# Outputs
# -------

output "aws_s3_model_artifacts_bucket" {
  value = aws_s3_bucket.model_artifacts.bucket
}

output "gcp_gcs_model_artifacts_bucket" {
  value = google_storage_bucket.model_artifacts.name
}

output "azure_storage_account" {
  value = azurerm_storage_account.model_artifacts.name
}

output "kubernetes_namespace" {
  value = kubernetes_namespace.llmops_manager.metadata[0].name
}

output "aws_ecs_cluster" {
  value = aws_ecs_cluster.llmops_manager_cluster.name
}

output "gcp_cloud_run_service" {
  value = google_cloud_run_service.llmops_manager.name
}

output "azure_kubernetes_cluster" {
  value = azurerm_kubernetes_cluster.llmops_manager_aks.name
}
