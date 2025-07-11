# 🧠 Epiguard‑AI: Epidemic Prediction & Prevention Web App

**Epiguard-AI** is an intelligent, ML-powered web application that predicts potential epidemic outbreaks based on weather and demographic data.

> ✅ Live Demo: [https://epiguard.onrender.com](https://epiguard.onrender.com)  

## ✨ Features

✅ **Epidemic Prediction** based on weather (temperature, humidity), population, and regional input  
✅ **Trained ML model** (Scikit-learn) for accurate forecasting   
✅ **Precaution list** to minimize spread  
✅ **Medicine suggestions** and **vaccine info** (if available)  
✅ **User-friendly UI** powered by Flask and HTML  
✅ **Model retraining support** via `model_train.py`  
✅ **Portable deployment** using Render or Railway  
✅ **Logs and saved models** are auto-managed   
✅ **Supports uploading custom datasets**

---

## 🌐 Live Demo

Access the deployed app here:  
🔗 [https://epiguard.onrender.com](https://epiguard.onrender.com)


## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.8+
- `pip` (Python package installer)
- Git

---

### 🔧 Installation

```bash
git clone https://github.com/Farhacali/Epiguard-AI-epidemic-prediction.git
cd Epiguard-AI-epidemic-prediction
pip install -r requirements.txt

🤖 Model Overview
Framework: Scikit-learn

Inputs: Population, temperature, humidity, region

Output: Predicted epidemic

Training script: model_train.py

Serialization: joblib.dump() → models/epidemic_model.pkl

👩‍💻 Contributors
Farha C. Ali
GitHub: @Farhacali

📜 License
Licensed under the MIT License – feel free to use and modify.
