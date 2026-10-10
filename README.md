# IPL Batsman Performance & Runs Predictor

An end-to-end Machine Learning pipeline that analyzes historical Indian Premier League (IPL) match data (`deliveries.csv` and `matches.csv`) to predict individual batsman performance and match run totals.

## 🚀 Project Overview
This project performs rigorous Exploratory Data Analysis (EDA), automated feature engineering (calculating player form curves, strike rates, batting positions, and opposing bowling strengths), and builds/compares four different regression models to forecast batsman scores.

## 📂 Repository Structure

├── models/

│   ├── model_1_linear.pkl            # Multiple Linear Regression Pipeline

│   ├── model_2_polynomial.pkl        # Polynomial Regression Pipeline (Degree 2)

│   ├── model_3_decision_tree.pkl     # Tuned Decision Tree Regressor

│   └── model_4_random_forest.pkl     # Random Forest Regressor

├── results/

│   ├── comparison_table.png          # Side-by-side performance comparison of all models

│   ├── feature_importance.png        # Random Forest feature importance chart

│   ├── eda_runs_distribution.png     # Distribution of batsman runs per match

│   ├── eda_avg_vs_runs.png           # Player batting average vs total runs scatter plot

│   ├── eda_strike_rate_dist.png      # Density distribution of strike rates

│   ├── eda_top_venues.png            # Top venues by match count

│   └── eda_toss_decision.png         # Toss decision breakdown (Bat vs Field)

├── notebooks/

│   └── analysis.ipynb                # Main Jupyter Notebook containing code, EDA, and training

└── README.md



## 📊 Exploratory Data Analysis (EDA)

The pipeline generates 5 core EDA visualizations saved under `results/`:

1. **Runs Distribution**: Visualizes the skewness and spread of runs scored by batsmen per match.
2. **Average vs. Total Runs**: Explores the correlation between a player's career batting average and total accumulated runs.
3. **Strike Rate Density**: Analyzes the distribution density of player strike rates.
4. **Top Venues**: Highlights the most frequent match venues across historical IPL seasons.
5. **Toss Decision Breakdown**: Proportions of teams choosing to bat vs. field first after winning the toss.

## 🤖 Model Performance Summary

| Model | Root Mean Squared Error (RMSE) | R-squared ($R^2$) Score |
| --- | --- | --- |
| **Multiple Linear Regression** | 17.76 | 0.1825 |
| **Polynomial Regression (Degree 2)** | **17.74** | **0.1847** |
| **Decision Tree Regressor** | 18.36 | 0.1272 |
| **Random Forest Regressor** | 18.06 | 0.1554 |

### 🏆 Key Findings & Best Model Justification

* **Polynomial Regression (Degree 2)** achieved the best overall performance, slightly outperforming standard linear modeling with an RMSE of **17.74** and an $R^2$ score of **0.1847**.
* It successfully captured the non-linear dynamics and interaction terms between player form curves and venue statistics without overfitting.
* Tree-based models captured localized splits but showed higher variance on test splits.

## 🛠️ Requirements & Libraries

* Python 3.x
* Pandas & NumPy
* Scikit-Learn
* Matplotlib & Seaborn
