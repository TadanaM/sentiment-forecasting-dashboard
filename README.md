# Sentiment-Augmented Financial Forecasting Using LLMs

> BSc Computer Systems — First Class Honours Dissertation  
> Heriot-Watt University Dubai ·  
> Supervised by Dr Drishty Sobnath

---

## Overview

This project investigates whether integrating **LLM-derived sentiment signals** from financial news improves stock price forecasting accuracy over traditional time-series baselines.

Six forecasting models were built, evaluated, and compared on **1,509 trading days of Apple Inc. (AAPL) data (2018–2024)**:

- Three **baseline models**: Linear Regression, ARIMA, Random Forest
- Three **sentiment-augmented hybrids**: LR+Sentiment, ARIMAX+Sentiment, RF+Sentiment

Sentiment features were generated using **[ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert)**, a finance-domain BERT model, fine-tuned on the **FinancialPhraseBank** corpus. An interactive **Streamlit dashboard** was built to present forecasts, model comparisons, and a UX evaluation (mean SUS score: 78.5 / 100).

---

## Key Results

| Model | MAE | RMSE | Directional Accuracy |
|---|---|---|---|
| Linear Regression (baseline) | 0.011447 | 0.015823 | 0.4983 |
| ARIMA (baseline) | 0.011509 | 0.015919 | 0.5083 |
| Random Forest (baseline) | 0.012208 | 0.016586 | 0.4718 |
| **LR + Sentiment** | 0.011479 | 0.015853 | 0.5017 |
| **ARIMAX + Sentiment** | 0.011524 | 0.015945 | **0.5116 ✓ best** |
| **RF + Sentiment** | 0.012126 | 0.016446 | 0.5050 |

> **Finding:** All three sentiment-augmented hybrid models outperformed their respective baselines. ARIMAX achieved the highest directional accuracy at **51.16%**, suggesting that incorporating FinBERT-derived sentiment as an exogenous variable improves trend prediction over a pure ARIMA baseline.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Sentiment | ProsusAI/FinBERT (HuggingFace Transformers) |
| Forecasting | scikit-learn, statsmodels (ARIMA/ARIMAX) |
| Data | yfinance (AAPL), FinancialPhraseBank |
| Dashboard | Streamlit |
| Evaluation | SUS (System Usability Scale), 5 participants |
| Environment | VS Code Dev Container |

---

## Repository Structure

```
sentiment-forecasting-dashboard/
├── Dashboard/          # Streamlit app (app.py — main entry point)
├── Evaluation/         # UX study results and SUS scoring
├── data/               # AAPL historical price data + processed sentiment
├── models/             # Trained model artefacts (.pkl / saved weights)
├── sentiment/          # FinBERT inference pipeline
├── .devcontainer/      # Dev container config for reproducible environment
└── .vscode/            # Editor settings
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- pip

### Installation

```bash
git clone https://github.com/TadanaM/sentiment-forecasting-dashboard.git
cd sentiment-forecasting-dashboard
pip install -r Dashboard/requirements.txt
```

## Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sentiment-forecasting-dashboard-d2frwtt3aklnvjwzhuenyz.streamlit.app/)

> *Dashboard deployed to Streamlit Community Cloud as part of the dissertation UX evaluation.*

## Methodology

### Data
- **Price data**: AAPL daily OHLCV from Yahoo Finance, 2018–2024 (1,509 trading days)
- **Train/test split**: 80/20 chronological (no shuffling to preserve temporal order)
- **Random seed**: 42

### Sentiment Pipeline
Financial news headlines were passed through ProsusAI/FinBERT to generate three sentiment probability scores per day: `positive`, `negative`, `neutral`. The dominant label and its confidence score were used as exogenous features in the hybrid models.

### Models
- **Linear Regression**: Closing price predicted from lagged features
- **ARIMA**: Auto-regressive integrated moving average on price series
- **Random Forest**: Ensemble of decision trees on engineered lag features
- **Hybrid (+Sentiment)**: Each baseline extended with FinBERT sentiment scores as additional input features; ARIMA extended to ARIMAX (exogenous inputs)

### Evaluation
Models were assessed on MAE, RMSE, and **Directional Accuracy** (whether the model correctly predicted up/down movement). Directional accuracy is the most practically meaningful metric for trading applications.

---

## UX Evaluation

A usability study was conducted with **5 participants** using the **System Usability Scale (SUS)**:

- Mean SUS score: **78.5 / 100**
- Interpretation: *Good* usability (above the 68-point industry benchmark)

---

## Limitations & Future Work

- Dataset limited to a single stock (AAPL); generalisation to other equities untested
- Sentiment sourced from FinancialPhraseBank; real-time news feeds would improve practical utility
- ARIMAX directional accuracy (51.16%) marginally exceeds random baseline — further feature engineering and longer training windows may improve this
- Future extension: replace static sentiment with live LLM inference (e.g. GPT-4, Claude) on real-time Reuters/Bloomberg feeds

---

## Academic Context

> **Dissertation title**: Sentiment-Augmented Financial Forecasting Using LLMs  
> **Degree**: BSc Computer Systems (First Class Honours)  
> **Institution**: Heriot-Watt University Dubai  
> **Supervisor**: Dr Drishty Sobnath  
> **Year**: 2026

*This repository contains the implementation artefacts submitted as part of the above dissertation. The full written dissertation is available on request.*

---

## Author

Tadana Manombe — [LinkedIn](https://www.linkedin.com/in/tadana-manombe-87270738a/) · [GitHub](https://github.com/TadanaM)
