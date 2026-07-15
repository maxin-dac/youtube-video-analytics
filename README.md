# 📊 YouTube Content Performance: EDA, Machine Learning & Strategy

## Context & Objectives

This project delivers a rigorous, data-driven approach to understanding YouTube video performance dynamics and predicting content virality using Machine Learning. Using metadata from **537 videos**, we explore audience interaction patterns, analyze category-level performance, and build a robust predictive pipeline.

### Core questions we answer:
* **Category Performance:** Which content categories drive the most reach and engagement?
* **Duration Dynamics:** What is the relationship between video duration and performance?
* **Formats Comparison:** Are "Shorts" outperforming mid/long-form content, and by how much?
* **Channel Dominance:** Which creators and channels dominate their respective niches?
* **Temporal Patterns:** How does video age affect accumulated views, and when is the best time to publish?
* **Virality Signatures:** What engagement metrics and distribution patterns distinguish viral hits from average content?
* **Investment & Strategy:** Where should creators or sponsors invest their next production budget?

## Data quality & Preprocessing workflow

Before performing analysis or training models, a comprehensive **Data Quality Assessment (QA)** was conducted to ensure empirical integrity:
1. **Type coercion:** Standardized timestamps (`published_at` parsed in UTC) and structured categorical columns.
2. **Category mapping:** Mapped raw YouTube Category IDs to human-readable names (e.g., *Music*, *Gaming*, *Entertainment*) using official API metadata.
3. **Deduplication:** Confirmed zero duplicate records based on `title + channel_title + published_at`.
4. **Consistency checks:** Verified logical bounds (no videos had `like_count` or `comment_count` greater than `view_count`).
5. **Zero-Variance feature dropping:** Dropped the `favorite_count` column as all 537 values were `0`, providing zero predictive power.
6. **Anomalous content handling:** Isolated non-performing/removed entries (e.g., a 15-day-old video with `0` views).


## Dataset key metrics (At a glance)

Below is the statistical profile of our final dataset across key performance indicators (KPIs):

| Metric | Mean (Average) | Median (50th percentile) | Max Value |
| :--- | :--- | :--- | :--- |
| **Views** | 21.45M | 6.38M | 369.73M |
| **Likes** | 433.6K | 155.05K | 10.87M |
| **Comments** | 7.12K | 2.35K | 810.64K |
| **Engagement Rate** | 2.87% | 2.44% | 21.57% |
| **Duration** | 1h 20m (4,802s) | 6.5 min (390s) | 29.2h (105,227s) |

---

## Feature engineering

To capture non-linear relationships and temporal nuances, we engineered several high-signal features:

| Feature | Extraction / Formula | Strategic Rationale |
| :--- | :--- | :--- |
| `format` | `"Short (≤60s)"` / `"Mid (1–10 min)"` / `"Long (>10 min)"` | Standardizes video duration into YouTube product categories (Shorts vs. Standard). |
| `views_per_day` | `view_count / max(video_age_days, 1)` | Measures daily view velocity, neutralizing age-bias for older videos. |
| `likes_per_comment` | `like_count / max(comment_count, 1)` | Unveils community interaction ratios (passive liking vs. active commenting). |
| `engagement_bucket`| Quantile split (`q=4`) on `engagement_rate` | Categorizes audience engagement into ordinal buckets (*Low*, *Mid*, *High*, *Very High*). |
| `viral_flag` | `(view_count > p95)` & `(engagement_rate > median)` | Defines true virality: videos reaching the top 5% of views (**>85.9M views**) with healthy engagement. |
| **Temporal Features**| `publish_hour_utc`, `publish_dow`, `publish_month` | Powers schedule optimization and cohort trend analysis. |

---

## Key Insights

* **Duration is critical:** Viral hits are significantly shorter, with an average length of **3:35 (215 seconds)**. Conversely, standard/non-viral content averages **over 1 hour 20 minutes** (4,802 seconds). Shorter, high-impact structures strongly correlate with virality.
* **The engagement dilution effect:** Non-viral videos maintain a higher average engagement rate (**2.93%**) compared to viral videos (**1.76%**). Large mass-audience exposure dilutes relative engagement, a common phenomenon as content spills to more passive viewer cohorts.
* **Optimal publishing window:** Publishing times drive early momentum. Clear performance peaks were observed in the **11:00-17:00 window, with a maximum at 14:00** (when YouTube's US/Europe audience is online).
* **The Viral Threshold:** To qualify as a high-velocity viral outlier, a video must cross **85.9 Million Views** (defining the 95th percentile threshold).

---

## Predictive Modeling & Performance

We built and evaluated a `RandomForestClassifier` to predict "High Performance" videos (videos scoring above the median view threshold of **6.38M views**). 

* **Overall Accuracy:** **74%**
* **Class Performance Highlights:**
  * **Non-Viral Class Retention (Recall):** **83%** (highly effective at identifying predictable, steady-performing content).
  * **Viral Hit Recall:** **62%** (captures nearly two-thirds of breakout hits despite their highly volatile, non-linear nature).

---

## Content strategy

1. **Aim for the 3-4 minute sweet spot:** Prioritize high-impact, short-form editing styles to boost completion and algorithmic virality.
2. **Automate the posting schedule:** Queue uploads to go live just prior to peak active windows (e.g., 6:00 AM - 10:00 AM UTC).
3. **Optimize early engagement Call-to-Actions (CTAs):** Encourage immediate comments and shares in the first 30 seconds of video playback to trigger early recommendation signals.
4. **Benchmark with ML:** Pre-evaluate planned topics and scripts using predicted duration, expected niche engagement, and timing parameters.

---

## Tech Stack & Libraries

* **Language:** ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
* **Data Wrangling:** ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
* **Statistical Analysis:** ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)
* **Visualizations:** ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-5B8DB8?style=flat) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)
* **Machine Learning:** ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white) (`RandomForestClassifier`, `Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`)
* **Environments:** ![Jupyter Notebook](https://img.shields.io/badge/Jupyter_Notebook-F37626?style=flat&logo=jupyter&logoColor=white) ![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)

---

## Resources & Links

* **Kaggle Notebook:** [View the code on Kaggle](https://www.kaggle.com/code/maximendacleu/youtube-video-analytics)
* **Local Notebook:** [Download `youtube-content-analytics.ipynb`](./youtube-content-analytics.ipynb)
* **Project PDF Report:** [Download Notebook PDF](./youtube-video-analytics.pdf)
