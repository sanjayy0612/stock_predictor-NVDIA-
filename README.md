# 📈 End-to-End Stock Price Prediction  

---

## 📌 Project Overview  
This repository documents my **end-to-end journey** of predicting the stock price of **NVIDIA (NVDA)**.  

The project started as a simple exploration and evolved into a deep dive comparing **classic Machine Learning algorithms** against a more sophisticated **Deep Learning (LSTM) architecture**.  

🔑 **Core Objective**: Not just to build a predictive model, but to understand *why* each step is necessary, and document lessons from both **successes** and **failures**.  

---

## 🧩 Project Phases  

### **Phase 1: Machine Learning Baselines**  
Using **scikit-learn** to set baselines.  

1. **Logistic Regression (Classification: Up/Down Prediction)**  
   - Accuracy: **55%**  
   - ⚠️ Flaw: Only ever predicted "Up".  
   - **Lesson:** Accuracy can be misleading — always check the **Confusion Matrix**.  

2. **Linear Regression (Regression: Predict Price)**  
   - RMSE: **$6.75**  
   - Strategy: Learned a *naive forecast* → "Tomorrow ≈ Today".  
   - **Lesson:** Even a "dumb" model can set a tough baseline to beat.  

3. **Random Forest Regressor**  
   - RMSE: **$21.92**  
   - Flaw: Failed on extrapolation (flat predictions when price rallied).  
   - **Lesson:** Tree models are bad for extrapolation in time-series.  

---

### **Phase 2: Deep Learning with LSTMs**  
Sequential models designed for **time-series forecasting**.  

1. **LSTM with Keras (TensorFlow)**  
   - RMSE: **$7.83**  
   - Captured **trend, momentum, and volatility**.  
   - **Lesson:** Right architecture > brute force. LSTM’s memory makes it ideal.  

2. **LSTM with PyTorch**  
   - RMSE: **$7.86**  
   - Identical performance to Keras after debugging tensor shapes & scaling.  
   - **Lesson:** Validated architecture + gained hands-on framework debugging.  

---

## 🧠 Key Learnings & Takeaways  

- ✅ A **naive baseline** (Linear Regression) can be surprisingly strong.  
- ✅ **Accuracy ≠ usefulness** → always inspect confusion matrix.  
- ✅ **Random Forest ≠ time-series friendly** (fails at extrapolation).  
- ✅ **LSTMs shine** when learning sequences, trends, and volatility.  
- ✅ Debugging takes 90% of the effort → fixing scaling, shapes, bugs taught me more than model training.  

---

## 🏆 Final Results  

| Model                | RMSE (Regression) | Key Takeaway |
|-----------------------|------------------:|--------------|
| **Linear Regression** | **$6.75**        | Strong naive baseline |
| **Random Forest**     | **$21.92**       | Failed due to no extrapolation |
| **LSTM (Keras)**      | **$7.83**        | Captured stock momentum & trends |
| **LSTM (PyTorch)**    | **$7.86**        | Validated effectiveness, improved debugging skills |

---

## ⚙️ Setup & Usage  

Clone the repo:  
```bash
git clone https://github.com/sanjayy0612/stock_predictor-NVIDA
cd stock_predictor-NVIDA

Install dependencies:
   pip install pandas yfinance scikit-learn tensorflow torch matplotlib

Run the code:
The project is contained within a Jupyter Notebook (.ipynb) or a Python script (.py). Open it and run the cells/script to replicate the analysis.
