# Kubernetes Assginment (Minikube)

## Project Overview

This project demonstrates how to deploy and manage a containerized application using Kubernetes on Minikube. It covers core concepts like Pods, ReplicaSets, Deployments, Services, Rolling Updates, and Health Probes.

The application is a simple web UI that displays different versions (**v1** and **v2**) to showcase rolling updates.

---

## Tech Stack

1) Kubernetes (Minikube)
2) Docker
3) Apache HTTP Server (`httpd`)
4) HTML (Frontend UI) 

---

## Docker Setup

### Build Images inside Minikube

```bash
minikube image build -t k8s-app:v1 -f Docker/dockerfile.v1 .
minikube image build -t k8s-app:v2 -f Docker/dockerfile.v2 .
```

### Verify Images

```bash
minikube image ls
```

---

## Kubernetes Setup

### 1 Pod (Basic Demo)
```bash
kubectl apply -f k8s/pod.yml
kubectl get pods
```

### 2️ ReplicaSet (Scaling Demo)
```bash
kubectl apply -f k8s/rs.yml
kubectl get pods
```

Demonstrates multiple replicas and self-healing.

### 3 Deploy Application (Version 1)

```bash
kubectl apply -f k8s/deployment.yml
```

### 4 Expose Application

```bash
kubectl apply -f k8s/service.yml
```

### 5 Check Resources

```bash
kubectl get pods
kubectl get svc
```

### 6 Access Application

```bash
minikube service demo-service
```

Output: **Version 1**

---

## Rolling Update (v1 → v2)

### Update Deployment

```bash
kubectl annotate deployment demo-app-deployment \
kubernetes.io/change-cause="Updated to v2" --overwrite

kubectl set image deployment/demo-app-deployment demo-app=k8s-app:v2
```

### Check Rollout Status

```bash
kubectl rollout status deployment demo-app-deployment
```

### Check Rollout History

```bash
kubectl rollout history deployment demo-app-deployment
```

Output:

```
REVISION  CHANGE-CAUSE
1         Initial deployment v1
2         Updated to v2
```

### Access Updated App

```bash
minikube service demo-service
```

 Output: **Version 2**

---

## Health Probes

### Apply Deployment with Probes

```bash
kubectl apply -f k8s/deployment_probe.yml
```

### Verify Probes

```bash
kubectl describe pod <pod-name>
```

Includes:

1) Readiness Probe
2) Liveness Probe

---

## Cleanup

### Delete All Resources

```bash
kubectl delete -f k8s/
```