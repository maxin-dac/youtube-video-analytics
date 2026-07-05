# YouTube Video Performance Analysis
<img width="796" height="400" alt="85c9323b-5fd1-49ae-a902-dd41c41b88ba" src="https://github.com/user-attachments/assets/075cd32c-b243-4dcd-9fc7-2b70200ba8e9" />

## Context
- From Kaggle: [https://www.kaggle.com/datasets/shaistashahid/youtube-analytics-data]
- Dataset: youtube_recommendation_dataset -.csv

## Objective
This project delivers a rigorous data-driven approach to understanding YouTube video performance dynamics and predicting content virality using Machine Learning. The main objectives are:
* **Exploration:** Analyze engagement distributions (Views, Likes, Comments) to isolate performance drivers.
* **Prediction:** Train a classification model to forecast content virality with a high level of accuracy (Global Accuracy: **74%**).
* **Action:** Extract actionable insights to build an optimized, data-backed content strategy.

Each row represents a single YouTube video and includes:
- Video metadata
- Channel information
- User engagement metrics
- Video category and duration details
- Computed ratios (engagement rate, likes/views, comments/views)
- Video age and publication characteristics

## Notebook Structure & Workflow
* **Data Loading & Exploration:** Import the YouTube dataset, inspect structure, and understand key metrics.
* **Exploratory Data Analysis (EDA):** Visualize relationships between features and performance: Views vs. Duration, Engagement Rate, Average Views by Publish Hour, Viral Threshold Analysis.
* **Feature Engineering:** Create derived metrics such as engagement rate.
* **Machine Learning Classification:** Training of a Random Forest model to predict "High Performance" videos (above median views).
* **Model Evaluation:** Analyze accuracy, classification report, and confusion matrix.

## Key Insights & Feature Importance
* **Duration is Critical:** Viral videos are significantly shorter (average **215 seconds / 3:35**) compared to non-viral videos (average **over 1 hour 20 minutes**). Shorter, high-impact formats perform much better.
* **Engagement Rate:** Non-viral videos actually show a higher engagement rate (**2.93%**) than viral ones (**1.76%**). This is common in viral content, where a large number of passive viewers dilutes the ratio.
* **Publish Timing:** Videos published during peak hours tend to gain more interaction. The peaks were observed successively at 7am, 6am, 10am, 8pm and 12pm.
* **Viral Threshold:** A video is considered viral if it exceeds **85.9 million views** (**top 5% of the dataset**).

## Model Performance
The Random Forest classifier achieved **74% accuracy** in predicting high-performance videos. It is better at identifying non-viral content (83% recall) than viral hits (62% recall).

## Content Strategy & Optimization
Based on the analysis, the following recommendations can help creators maximize video performance:
* **Prioritize Short-Form Content:** Aim for videos around 3-4 minutes to increase virality potential.
* **Optimize Posting Schedule:** Publish during peak viewing hours identified in the hourly analysis.
* **Focus on High-Engagement Niches:** Target categories and topics with the strongest average views and interaction.
* **Boost Engagement Tactics:** Encourage likes and comments early in the video to improve algorithmic promotion.
* **Monitor Virality Signals:** Use the classification model to evaluate new video ideas based on duration, expected engagement, and timing.

## Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation:** `pandas`, `NumPy`
* **Visualization:** `Matplotlib`, `Seaborn`
* **Machine Learning:** `scikit-learn` (`RandomForestClassifier`, `train_test_split`, `classification_report`, `confusion_matrix`)
* **Environment:** Kaggle Notebook

---
*To explore the detailed code, feel free to download the notebook file on this repo, or check out my Kaggle: https://www.kaggle.com/code/maximendacleu/youtube-video-analytics.*

***By: Maxime NDACLEU | BI & Data Analyst***
