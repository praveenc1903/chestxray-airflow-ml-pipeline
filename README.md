# 🩺 Chest X-Ray Pneumonia Detection – Airflow ML Pipeline

An **end-to-end Machine Learning pipeline** for detecting **Pneumonia from Chest X-ray images**, orchestrated using **Apache Airflow**, trained with **TensorFlow/Keras**, and tracked using **MLflow**.  
The entire workflow is containerized using **Docker & Docker Compose**.

This project demonstrates **MLOps fundamentals** including workflow orchestration, model training automation, experiment tracking, and reproducible ML pipelines.

---

## 🚀 Tech Stack

- **Python**
- **Apache Airflow** – workflow orchestration
- **TensorFlow / Keras** – CNN model training
- **MLflow** – experiment tracking & model logging
- **Docker & Docker Compose**
- **Kaggle Chest X-ray Dataset**

---

## 📌 Project Objectives

- Automate the ML lifecycle using Airflow
- Train a CNN for pneumonia detection from medical images
- Track metrics, parameters, and models using MLflow
- Build a reproducible, containerized ML pipeline
- Follow best practices used in real-world MLOps systems

---
## 🔁 Airflow DAG Graph View

![Airflow DAG Graph](cnn_chest_xray_pipeline-graph.png)


---

## 🔄 Pipeline Workflow

1. **Dataset Download**
   - Chest X-ray Pneumonia dataset downloaded from Kaggle

2. **Data Preprocessing**
   - Image resizing & normalization
   - Train / validation / test split

3. **Model Training**
   - CNN architecture built using TensorFlow/Keras
   - Binary classification: Normal vs Pneumonia

4. **Model Evaluation**
   - Accuracy & loss metrics calculated

5. **Experiment Tracking**
   - Metrics, parameters, and models logged to MLflow

6. **Orchestration**
   - Entire workflow executed via Airflow DAG

---

## 📊 Dataset

- **Chest X-Ray Images (Pneumonia)**
- Source: Kaggle (Paul Mooney)
- Dataset is **not included** in this repository due to size constraints

---

## ▶️ How to Run the Project

### Prerequisites
- Docker
- Docker Compose

---


### Access Services

Service        URL
--------------------------------
- Airflow UI     http://localhost:8080
- MLflow UI      http://localhost:5000


### Default Airflow Credentials

- Username: airflow
- Password: airflow

### Start Airflow & MLflow
```bash
docker compose up -d


