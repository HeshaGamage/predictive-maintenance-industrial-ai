# predictive-maintenance-industrial-AI
 
A predictive maintenance system that monitors aircraft engines, predicts failures before they happen, and generates maintenance work orders automatically.
 
Built as a personal project to explore how companies like IFS implement Industrial AI in asset-heavy industries.
 
**Live demo:** [https://predictive-maintainance-v01.streamlit.app]  
**Demo video:** [https://youtu.be/ednx9Flmqbc]
 
---
 
## What it does
 
- Predicts how many flight cycles each engine has left before failure
- Explains which sensors are driving the prediction (using SHAP)
- Automatically generates a maintenance work order for high-risk engines using an LLM
- Displays everything in an interactive dashboard
---
 
## How it works
 
```
Sensor data → XGBoost model → SHAP explanation → Llama 3 work order → Dashboard
```
 
Trained on NASA's CMAPSS turbofan engine dataset (FD001). The model predicts Remaining Useful Life (RUL) with a test RMSE of 16.24 cycles against a 30–40 cycle warning window.
 
---
 
## Stack
 
- **ML model** — XGBoost with rolling window features (5, 10, 30 cycle averages)
- **Explainability** — SHAP TreeExplainer
- **LLM agent** — Llama 3 via Groq API
- **Dashboard** — Streamlit
---
 
## Results
 
| Metric | Value |
|--------|-------|
| Validation RMSE | 13.86 cycles |
| Test RMSE | 16.24 cycles |
| Engines flagged (CRITICAL/HIGH) | 22 out of 100 |
| Top predictor | LPT outlet temperature (s4, 5-cycle rolling mean) |
 
---
 
## Setup
 
```bash
git clone https://github.com/HeshaGamage/predictive-maintenance-industrial-ai
cd predictive-maintenance-industrial-ai
pip install -r requirements.txt
```
 
Add your Groq API key to a `.env` file:
```
GROQ_API_KEY=your_key_here
```
 
Download the NASA CMAPSS dataset from [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) and place the files in `data/raw/`.
 
Run the notebooks 01 to 05 in order, then:
```bash
streamlit run app.py
```
 
---
 
## Project structure
 
```
├── notebooks/          # Data exploration → features → model → SHAP → LLM agent
├── data/raw/           # NASA CMAPS files (not committed)
├── data/processed/     # Processed features and cached work orders
├── models/             # Saved model, scaler, SHAP explainer
└── app.py              # Streamlit dashboard
```
 
---
 
## Dataset
 
NASA CMAPSS Turbofan Engine Degradation Dataset (FD001)  
100 engines run to failure, 21 sensors per flight cycle.  
Fault: High Pressure Compressor degradation.
 
---
 
*Hesha Gamage — [LinkedIn](https://www.linkedin.com/in/heshan-kavishka-655381215/)*