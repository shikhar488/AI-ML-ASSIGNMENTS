# 🚀 Assignment 2: Backend API & Docker Deployment

## 📌 Overview

This project demonstrates how to build a simple backend API using **FastAPI** and containerize it using **Docker**. The goal is to understand API development, dependency management, and running applications inside containers.

---

## 📝 Tasks: 

### 1. Backend API Development
- Developed a RESTful API using **FastAPI** and **Uvicorn**.
- Created a root endpoint (`/`) returning a "Hello World" message.
- Created a data endpoint (`/data`) returning sample JSON metadata.

## ▶️ Run Locally

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Server

```bash
uvicorn app:app --reload
```

### 3. Access API

* http://127.0.0.1:8000
* http://127.0.0.1:8000/data

---

## 🐳 Run with Docker

### 1. Build Image

```bash
docker build -t pythonapp:1.0 .
```

### 2. Run Container

```bash
docker run -d -p 8000:8000 --name pythonwebapp pythonapp:1.0
```

### 3. Verify

Open browser:

* http://localhost:8000
* http://localhost:8000/data



