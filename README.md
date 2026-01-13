# 🚀 NotionFL — Federated Learning with Explainable AI

**NotionFL** is a Final Year Project that implements a **Trustworthy privacy-preserving Federated Learning (FL) system** with built-in **Explainable AI (XAI)** capabilities.

It enables multiple distributed clients to collaboratively train a shared machine-learning model **without sharing raw data**, while also providing **clear explanations** of how the model is trained and updated.

This project was designed to demonstrate how modern AI systems can be:
- 📊 Scalable
- 🔐 Privacy-aware
- 🔍 Interpretable

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Running the System](#running-the-system)
- [Usage](#usage)
- [Explainable AI Layer](#explainable-ai-layer)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 🔍 Overview

Traditional machine learning requires collecting all data in a single central server, which creates **privacy, security, and compliance risks**.

Federated Learning solves this by allowing each client to train locally on its own data and send only **model updates** to a central server.

**NotionFL** enhances this by adding **explainability** so users can understand:
- How each client contributes to the global model
- How the global model evolves over training rounds
- Why model performance changes

This makes the FL process transparent, auditable, and trustworthy.

---

## 🧠 Key Features

- 🔐 **Privacy-Preserving Federated Learning**
  - Raw data never leaves the client machines
  
- 🌐 **Distributed Training**
  - Multiple clients train a shared model collaboratively
  
- 🧩 **Explainable AI Layer**
  - Visual and analytical insights into model updates and learning behavior
  
- 📊 **Web Dashboard**
  - View training rounds, accuracy, client participation, and model evolution
  
- ⚙️ **Modular Architecture**
  - Easily extendable to new datasets, models, and FL strategies

---

## 🧱 System Architecture

```
┌──────────────────────────┐     ┌──────────────────────────┐
│    Client Nodes          │     │   Central Server         │
│  (Local Datasets)        │────▶│  Federated Aggregator    │
│                          │     │  Global Model            │
│  - Train Local Model     │     │  Explanation Engine      │
│  - Send Updates          │◀────│  API Layer               │
└──────────────────────────┘     └──────────────────────────┘
        │                               │
        │                               │
        ▼                               ▼
    Client Nodes              Frontend Dashboard
    (Multiple)                (Monitoring & XAI)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python (FastAPI / Flask) |
| **Federated Learning** | PyTorch / TensorFlow |
| **Frontend** | React / JavaScript |
| **API Communication** | REST APIs |
| **Visualization** | Charts, Logs, Model Metrics |
| **Data Handling** | NumPy, Pandas |
| **Model Training** | PyTorch / TensorFlow |

---

## 📁 Project Structure

```
NotionFL/
│
├── NotionFL-BE/                 # Backend (Federated Learning Server & APIs)
│   ├── models/                  # ML Models
│   ├── routes/                  # API Routes
│   ├── fl_engine/               # Federated Learning Logic
│   ├── main.py                  # Entry Point
│   └── requirements.txt
│
├── NotionFL-FE/                 # Frontend (Dashboard & Visualization)
│   ├── src/
│   ├── components/              # React Components
│   ├── pages/                   # Pages
│   └── package.json
│
├── docs/                        # Documentation & Diagrams
├── datasets/                    # Sample or Test Datasets
└── README.md
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Ratheshan03/NotionFL.git
cd NotionFL
```

### 2️⃣ Backend Setup

```bash
cd NotionFL-BE
python -m venv venv

# Activate virtual environment
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

Create a `.env` file (if required):

```ini
SERVER_HOST=localhost
SERVER_PORT=8000
MODEL_PATH=./models
```

### 3️⃣ Run Backend

```bash
uvicorn main:app --reload
```

Backend will run at: `http://localhost:8000`

### 4️⃣ Frontend Setup

```bash
cd NotionFL-FE
npm install
npm start
```

Frontend will run at: `http://localhost:3000`

---

## 🧪 Usage

1. **Start the backend server** (see step 3️⃣ above)

2. **Launch the frontend dashboard** (see step 4️⃣ above)

3. **Connect one or more clients** to the federated learning network

4. **Start federated training** via the dashboard

5. **Monitor in real-time:**
   - Model Accuracy
   - Loss Curves
   - Client Participation Rates
   - Model Updates

6. **View explainability insights** via the dashboard to understand:
   - Training progress
   - Client contributions
   - Model performance trends

---

## 🔍 Explainable AI Layer

NotionFL does not treat the FL model as a black box. It provides:

- 📈 **Training Round Summaries** - Detailed logs of each training round
- 👥 **Client Contribution Statistics** - Impact analysis of each client
- 📊 **Model Performance Graphs** - Accuracy and loss visualizations
- 🔄 **Aggregation Insights** - How global model updates are computed

This helps users understand:
- **Why** the model improved or degraded
- **Which clients** had the biggest impact
- **How** training evolved over time
- **What** changed in each round

---

## 🔮 Future Improvements

- 🔐 Differential Privacy Integration
- 🛡️ Secure Aggregation
- 📊 Real-time SHAP / LIME Explanations
- ☁️ Deployment on Cloud or Kubernetes
- 📱 Support for Mobile & IoT Clients
- 🌍 Multi-language Support
- 🧪 Enhanced Testing & Benchmarks

---

## 📄 License

This project is released under the **MIT License**.

---

## 🙌 Author

**Ratheshan Sathiyamoorthy**  
Final Year Project – BSc in Computer Science

If you find this project useful, feel free to ⭐ **star the repository!**

---

## 📧 Support & Contributions

For issues, suggestions, or contributions, please open an issue or pull request on the [GitHub repository](https://github.com/Ratheshan03/NotionFL).

Happy Learning! 🎉
