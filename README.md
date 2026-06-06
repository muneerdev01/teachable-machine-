# 🤖 Teachable Machine Clone (Full-Stack AI Project)

## 📌 Overview

This project is a full-stack AI web application that allows users to build and train their own image classification models without deep ML knowledge. It replicates Google’s Teachable Machine using a professional architecture.

---

## ⚙️ Features

* Create custom image classes (e.g., Cat, Dog, Car)
* Upload images for each class
* Train model with one click
* Real-time prediction on new images
* Confidence score visualization

---

## 🏗️ Architecture

### Frontend (Streamlit)

* User interface
* Image upload & webcam input
* Train & predict buttons
* Displays results and charts

### Backend (FastAPI)

* Handles image storage
* Model training pipeline
* Prediction API
* Dataset management

### ML Pipeline

* Pre-trained MobileNetV3 for feature extraction
* Logistic Regression classifier on top
* Fast training with saved model (`model.pkl`)

---

## 📁 Project Structure

```
teachable-machine/
│
├── backend/
│   ├── main.py
│   ├── dataset/
│   ├── model.pkl
│
├── frontend/
│   ├── app.py
│
├── requirements.txt
└── docker-compose.yml
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Backend

```bash
cd backend
uvicorn main:app --reload
```

### 3. Start Frontend

```bash
cd frontend
streamlit run app.py
```

---

## 🔄 Workflow

1. Create class labels
2. Upload images
3. Train model
4. Test predictions

---

## 🎯 Goal

To build a production-style AI system with separated frontend and backend, similar to real-world machine learning applications.

---

## 👨‍💻 Tech Stack

* Streamlit
* FastAPI
* PyTorch
* Scikit-learn
* Pillow
* Requests
