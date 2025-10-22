# Rossmann Store Sales Forecasting

This project builds a time-series forecasting model to predict daily sales for the [Rossmann Store Sales Kaggle Competition](https://www.kaggle.com/c/rossmann-store-sales).

The model uses **SARIMAX** (Seasonal AutoRegressive Integrated Moving Average with eXogenous Regressors) to forecast sales for individual stores, capturing weekly seasonality and the impact of external factors like promotions and holidays.

## 📈 Key Results (Initial Stores)

This repository initially contains the analysis for individual stores to prove the modeling concept before scaling.

### Store 1 Results

* **Model:** `SARIMAX(1, 1, 1)x(0, 0, 1, 7)`
* **Competition Score (RMSPE):** **11.92%**
* **RMSE:** **$603.03**
* **Key Findings:** The model successfully identified the `Promo` variable as a highly significant (p < 0.001) driver of sales changes. `SchoolHoliday` was not significant for this store.

![Model Forecast vs. Actual Sales - Store 1](https://i.imgur.com/GjQ8tA1.png)
*(Note: Replace with your actual Store 1 plot link/image)*

### Store 2 Results

* **Model:** `SARIMAX(1, 1, 1)x(0, 0, 1, 7)` (Same stable structure applied)
* **Competition Score (RMSPE):** **27.47%**
* **RMSE:** **$1395.65**
* **Key Findings:** The model converged successfully. `Promo`, `SchoolHoliday`, and `StateHoliday_a` were all found to be statistically significant predictors of sales changes for this store, highlighting store-specific differences. The higher RMSPE suggests this simple model structure is less accurate for Store 2 than for Store 1.

![Model Forecast vs. Actual Sales - Store 2](https://i.imgur.com/placeholder_link_store2.png)
*(Note: You'll need to generate a link/upload the plot for Store 2 and add it here)*

## 🛠️ How to Run This Project

### 1. Setup

First, clone the repository to your local machine:

```bash
# Clone the repo
git clone https://github.com/YourUsername/rossmann-store-sales.git
cd rossmann-store-sales
```

### 2. Create Environment

This project uses a dedicated virtual environment.

```bash
# Create the virtual environment
python3 -m venv ross

# Activate the environment
source ross/bin/activate
```

### 3. Install Dependencies

Install all required libraries from the requirements.txt file:

```bash
pip install -r requirements.txt
```

### 4. Run the Notebook

The entire analysis is contained in a Jupyter Notebook. The easiest way to run it is in VS Code.

1.  Open the project folder in VS Code:
    ```bash
    code .
    ```
2.  Open the `notebooks/01-Data-Prep-and-SARIMAX.ipynb` file.
3.  When prompted, select the `Python (Rossmann Sales)` kernel.
4.  Run the cells in the notebook from top to bottom.

## 🧮 Modeling Approach

The goal was to build a model that captures both the internal momentum of sales (autocorrelation) and the impact of external events.

1.  **Data:** Loaded `train.csv` and `store.csv`. The data was filtered for a single store (`Store == 1`) and days where the store was open (`Sales > 0`).
2.  **Analysis (EDA):** Plotted the time series, which clearly showed a strong **7-day (weekly) seasonality**, confirming the use of `m=7` for the SARIMAX model.
3.  **Features (Exogenous):** The `Promo` and `SchoolHoliday` columns were used as the 'X' variables.
4.  **Model Selection & Tuning:**
    - `pmdarima.auto_arima` was first used to find the best model parameters.
    - This initial model `(3,0,4)(1,0,0)[7]` was found to be **unstable** and failed to converge.
    - Through diagnostic analysis, a "unit root" (`ar.L1` ~1.0) was identified as the cause of the instability.
    - A **differencing term (d=1)** was introduced to fix this, which is the "I" in ARIMA.
5.  **Final Model:** A stable `SARIMAX(1, 1, 1)x(0, 0, 1, 7)` model was successfully trained. This model correctly identified `Promo` as a significant predictor of sales.

## 🚀 Next Steps

- **Scale the Model:** Refactor the notebook into a script that loops through all 1,115 `Store` IDs, training a unique model for each.
- **Add Yearly Seasonality:** Engineer new features to capture the annual cycle (e.g., `Month_of_Year`, `Is_December`). This will allow the model to predict the large holiday spikes without the computational cost of `m=365`.
