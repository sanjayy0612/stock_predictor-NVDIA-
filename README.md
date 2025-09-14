# stock_predictor-NVDIA-
End-to-End Stock Price Prediction: A Journey from ML Baselines to Deep Learning
📌 Project Overview
This repository documents my end-to-end journey of building and evaluating various models to predict the stock price of NVIDIA (NVDA). The project started as a simple exploration and evolved into a deep dive comparing the capabilities of classic Machine Learning algorithms against a more sophisticated Deep Learning (LSTM) architecture.

The core objective was not just to build a predictive model, but to understand the process, learn the "why" behind each step, and document the key takeaways from each model's successes and failures.

The project is structured in two main phases:

Phase 1: Machine Learning Baselines: Using scikit-learn to establish baseline performance with Linear Regression and Random Forest for both classification and regression tasks.

Phase 2: Deep Learning with LSTMs: Implementing a Long Short-Term Memory (LSTM) network, which is designed for time-series data, using both Keras (TensorFlow) and PyTorch to compare frameworks and validate the model's effectiveness.

🧠 My Key Learnings & Takeaways
This project was a fantastic learning experience. Here are my biggest takeaways:

A "Dumb" Model Can Set a Tough Baseline: A simple Linear Regression that creates a "naive forecast" (predicting tomorrow's price is close to today's) can achieve a deceptively low error rate. Any complex model must definitively beat this simple logic to be considered valuable.

Higher Accuracy Isn't Always Better: The first classification model had 55% accuracy but was useless because it only ever predicted "Up". The second model had 52% accuracy but was far more useful because it at least tried to predict both outcomes. This taught me to always analyze the Confusion Matrix, not just the accuracy score.

Know Your Model's Weaknesses: The Random Forest, a powerful model, failed completely on the regression task because it cannot extrapolate—it can't predict prices outside the range it saw during training. This was a critical lesson in choosing the right tool for the job.

LSTMs Are Built for Time-Series: The LSTM was the only model that truly learned the momentum and trend of the stock. Its ability to process sequences of data proved far superior for this kind of problem.

Debugging is 90% of the Work: From tensor shape mismatches in PyTorch to data scaling bugs, the most valuable lessons came from diagnosing and fixing errors.

📈 The Modeling Journey: Step-by-Step Analysis
Phase 1: Machine Learning Baselines with Scikit-Learn
1. Logistic Regression (Classification: Predict Up/Down)
The first attempt was to predict if the price would go up (1) or down (0).

Output:

Classification Accuracy: 0.5517
Confusion Matrix:
 [[ 0 78]
  [ 0 96]]

Analysis: The model had 55% accuracy, which seemed okay at first. However, the confusion matrix revealed a critical flaw: it only ever predicted "Up" (1). It was a lazy model that simply learned that the stock went up more often than down in the training data.

Learning: A model's accuracy can be misleading. Always check the confusion matrix to understand its actual behavior.

2. Linear Regression (Regression: Predict the Price)
Next, I tried to predict the exact closing price.

Output:

RMSE: $6.75

Graph:

Analysis: The predictions followed the actual price very closely, but they were almost always lagging by one day. The model learned the simplest possible strategy: "tomorrow's price is probably very close to today's price."

Learning: This created a powerful baseline. Any "smarter" model I built had to achieve an RMSE lower than $6.75 to be considered an improvement.

3. Random Forest Regressor (Regression: Predict the Price)
I used a more powerful model to see if it could beat the baseline.

Output:

RMSE: $21.92

Graph:

Analysis: The Random Forest performed much worse than the simple linear model. The graph shows exactly why: when the actual stock price rallied into a new, higher price range, the model's predictions stayed flat.

Learning: Tree-based models like Random Forest are terrible at extrapolation. They cannot predict values outside the range of data they were trained on. This was a critical failure and a major lesson.

Phase 2: Deep Learning with LSTMs
Given the failure of the ML models to understand trends, I moved to an LSTM, which is designed for sequential data.

1. LSTM with Keras (TensorFlow)
Output:

RMSE: $7.83

Graph:

Analysis: A huge success! Although the RMSE was slightly higher than the naive baseline, the graph shows that the model was far more intelligent. It correctly captured the overall trend, momentum, and volatility of the stock price, even as it moved into new price ranges.

Learning: Using the right architecture for the problem domain is crucial. The LSTM's "memory" was the key to unlocking a more meaningful prediction.

2. LSTM with PyTorch
To solidify my understanding, I rebuilt the same model in PyTorch.

Output:

RMSE: $7.86

Graph:

Analysis: After solving several challenging bugs related to tensor shapes and data scaling, the PyTorch model produced a result nearly identical to the Keras model.

Learning: This validated that the LSTM architecture itself was effective. It also provided invaluable hands-on experience with the debugging process in a different deep learning framework.

🏆 Final Results & Conclusion
Model

RMSE (Regression Task)

Key Takeaway

Linear Regression

$6.75

A strong, but naive, baseline to beat.

Random Forest

$21.92

Failed completely due to inability to extrapolate.

LSTM (Keras)

$7.83

The most "intelligent" model, successfully capturing trend and momentum.

LSTM (PyTorch)

$7.86

Validated the LSTM's effectiveness and confirmed implementation skills.

While no model can predict the stock market with perfect accuracy, this project demonstrates that LSTMs are a vastly superior tool for time-series forecasting compared to standard regression models, as they can learn and adapt to underlying trends over time.

🛠️ Setup & Usage
Clone the repository:

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

Install dependencies:
It's recommended to use a virtual environment.

pip install pandas yfinance scikit-learn tensorflow torch matplotlib

Run the code:
The project is contained within a Jupyter Notebook (.ipynb) or a Python script (.py). Open it and run the cells/script to replicate the analysis.
