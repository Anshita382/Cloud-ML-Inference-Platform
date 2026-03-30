# Deployment Guide

## Local (Docker Compose)

```bash
# From project root
make up
```

## AWS EC2 (Phase 5 — Easy Route)

### 1. Launch EC2 Instance
- AMI: Amazon Linux 2023 or Ubuntu 22.04
- Instance type: t3.large (2 vCPU, 8GB RAM) minimum
- Security group: Open ports 8000, 3000, 9090, 22

### 2. Install Docker
```bash
# Amazon Linux 2023
sudo yum install -y docker
sudo systemctl start docker
sudo usermod -aG docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 3. Deploy
```bash
git clone https://github.com/YOUR_USERNAME/cloud-ml-inference-platform.git
cd cloud-ml-inference-platform
docker-compose up --build -d
```

### 4. Verify
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/predict/sync \
  -H "Content-Type: application/json" \
  -d '{"text": "Deployed on EC2!"}'
```

## AWS EKS (Phase 5 — Strong Route)

### 1. Create EKS Cluster
```bash
eksctl create cluster \
  --name ml-inference \
  --region us-east-1 \
  --nodegroup-name workers \
  --node-type t3.large \
  --nodes 3
```

### 2. Build & Push Images to ECR
```bash
# Create ECR repos
aws ecr create-repository --repository-name inference-api
aws ecr create-repository --repository-name inference-worker

# Login
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Build & push
docker build -f deploy/docker/Dockerfile.api -t inference-api .
docker tag inference-api:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/inference-api:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/inference-api:latest

docker build -f deploy/docker/Dockerfile.worker -t inference-worker .
docker tag inference-worker:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/inference-worker:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/inference-worker:latest
```

### 3. Install KEDA
```bash
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
helm install keda kedacore/keda --namespace keda --create-namespace
```

### 4. Deploy to EKS
```bash
# Update image references in k8s manifests to ECR URLs, then:
kubectl apply -f deploy/k8s/redis-deployment.yaml
kubectl apply -f deploy/k8s/api-deployment.yaml
kubectl apply -f deploy/k8s/worker-deployment.yaml
```

### 5. Verify Autoscaling
```bash
# Watch worker pods scale
kubectl get pods -w

# Run load test against EKS endpoint
ENDPOINT=$(kubectl get svc inference-api-service -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
locust -f loadtest/locustfile.py --host http://$ENDPOINT \
  --users 500 --spawn-rate 25 --run-time 120s --headless
```

## S3 Model Artifacts (Optional)

```bash
# Upload model to S3
aws s3 cp model/cache/ s3://your-bucket/models/distilbert-sst2/ --recursive

# Set environment variable for workers
MODEL_S3_PATH=s3://your-bucket/models/distilbert-sst2/
```
