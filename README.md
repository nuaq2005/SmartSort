# SmartSort

## Problem
Recyclables, electronics, and food scraps are mixed with common trash, as a result, it is difficult to recycle and estimate carbon emissions.  
Proper waste sorting can significantly reduce environmental impact but many people don’t know how to sort their trash correctly.  

That’s why we created **SmartSort**, an AI-powered tool that helps individuals and companies to properly dispose of waste while tracking its carbon footprint.  

---

## SmartSort
**SmartSort** is a smart guide to sorting trash!  
With just the snap of a photo, users can:
- Instantly identify what type of waste it is (recyclable, compost, landfill, etc.)
- Learn where to properly dispose of it
- See the estimated **carbon emissions** of each item
- Compete on a **leaderboard** that gamifies sustainable habits

---

## ⚙️ Technical Challenges Faced
- Connecting the **Flask backend** with the **React frontend**
- Training and optimizing the **PyTorch model** to accurately classify trash images and calculate CO2 emissions

---

## 🧱 How We Built It
### 🖥️ Tech Stack
- **Frontend:** React + Tailwind CSS  
- **Backend:** Flask (Python)  
- **Database & Authentication:** Firebase  
- **AI Model:** PyTorch image classification model trained on 15k waste datasets  

### 🧩 Architecture Overview
1. The **user uploads a photo** from the frontend.  
2. The **Flask backend** receives it, processes it through the **PyTorch model**, and returns classification + emission data.  
3. **Firebase** stores user data and leaderboard status.  
4. The **frontend** displays classification results, CO₂ estimates, and leaderboard updates in real time.

### 🧩 Architecture Overview

```mermaid
graph LR
    A[User Uploads Image] --> B[React Frontend]
    B --> C[Flask API Server]
    C --> D[PyTorch Model]
    D --> E[Classification + CO₂ Estimation]
    E --> F[Firebase Database & Auth]
    F --> G[Leaderboard & User History]

    style A fill:#f4f4f4,stroke:#888,stroke-width:1px
    style B fill:#61dafb,stroke:#333,stroke-width:1px,color:#000
    style C fill:#f8d775,stroke:#333,stroke-width:1px,color:#000
    style D fill:#f38b66,stroke:#333,stroke-width:1px,color:#000
    style E fill:#c3f0ca,stroke:#333,stroke-width:1px,color:#000
    style F fill:#f9a8d4,stroke:#333,stroke-width:1px,color:#000
    style G fill:#dbeafe,stroke:#333,stroke-width:1px,color:#000


### 🧠 Key Design Decisions
- Used **Flask** for lightweight API serving and seamless Python ML integration  
- Chose **Firebase** for real-time database and authentication  
- Built a **modular AI pipeline** to easily retrain or swap the model with updated datasets  
