# YouTube Video Performance Analysis
<img width="796" height="400" alt="85c9323b-5fd1-49ae-a902-dd41c41b88ba" src="https://github.com/user-attachments/assets/075cd32c-b243-4dcd-9fc7-2b70200ba8e9" />

## Objective
This project delivers a rigorous data-driven approach to understanding YouTube video performance dynamics and predicting content virality using Machine Learning. The main objectives are:
* **Exploration:** Analyze engagement distributions (Views, Likes, Comments) to isolate performance drivers.
* **Prediction:** Train a classification model to forecast content virality with a high level of accuracy (Global Accuracy: **74%**).
* **Action:** Extract actionable insights to build an optimized, data-backed content strategy.

## Dataset
From Kaggle: https://www.kaggle.com/datasets/shaistashahid/youtube-analytics-data?select=youtube_recommendation_dataset+-.csv

Each row represents a single YouTube video and includes:
- Video metadata
- Channel information
- User engagement metrics
- Video category and duration details
- Computed ratios (engagement rate, likes/views, comments/views)
- Video age and publication characteristics

## Key Insights & Feature Importance
The Machine Learning model highlights that a video's success is driven by strong structural variables rather than randomness:
1. **Initial Engagement Velocity:** Early interactions (Likes/Comments ratio) serve as the primary catalyst for long-term algorithmic reach.
2. **Temporal Format:** Video duration directly impacts audience retention, drastically segmenting high-performing clusters from standard ones.

## Strategic Recommendations (Content Optimization)
Based on the empirical findings from the model, three core strategic axes are recommended:
* **Format Calibration:** Target the identified *sweet spot* in video length to maximize viewer retention and completion rates.
* **Publishing Schedule:** Schedule video releases right before peak audience activity windows to capitalize on the initial algorithmic recommendation boost.
* **Engagement Engineering:** Implement targeted Calls-to-Action (CTAs) early in the video timeline to aggressively drive up the comment-to-view ratio.

---
*To explore the detailed code, statistical visualizations, and model evaluation metrics, feel free to download the notebook file on this repo, or check out my Kaggle: https://www.kaggle.com/code/maximendacleu/youtube-video-analytics.*

***By: Maxime NDACLEU | BI & Data Analyst***
