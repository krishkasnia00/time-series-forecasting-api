#  Time Series Forecasting API

##  Objective
Forecast next 8 weeks of sales for each state using Machine Learning models.

---

##  Models Used
- ARIMA
- Prophet
- XGBoost

---

##  Feature Engineering
- Lag Features
- Rolling Mean & Standard Deviation
- Time-based Features (Month, Week, Trend)

---

##  How to Run API

```bash
python -m uvicorn api.app:app --reload
