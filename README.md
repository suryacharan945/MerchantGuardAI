# MerchantGuard AI

[Launch the live Streamlit app](https://merchantguardai.streamlit.app/)

MerchantGuard AI is an autonomous payment-interception and fraud-investigation platform for merchants. It combines calibrated machine-learning risk scoring, behavioral anomaly detection, policy retrieval, explainability, and an auditable action layer in a Streamlit dashboard.

## What It Does

- Scores transactions with the bundled LightGBM ensemble and Platt calibration.
- Detects behavioral anomalies with the autoencoder and isolation forest models.
- Retrieves relevant operating policies from the local FAISS knowledge base.
- Produces evidence-backed investigation reports and merchant communications.
- Applies severity-based actions such as clear, review, hold, or decline.
- Records decisions and tool activity in a persistent SQLite audit ledger.
- Supports an optional Gemini-backed explanation path with PII redaction and caching.

## Quick Start

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the integrity smoke test:

```powershell
python smoke_test.py
```

Start the dashboard:

```powershell
streamlit run app.py
```

The application uses the model, data, and policy artifacts included in this repository. Run commands from the repository root.

## Optional Gemini Integration

The application works with its local and cached explanation paths without a Gemini key. To enable live Gemini explanations, set `GEMINI_API_KEY` before starting Streamlit:

```powershell
$env:GEMINI_API_KEY = "your-api-key"
streamlit run app.py
```

Do not commit API keys, `kaggle.json`, `.env` files, or the local `venv` directory.

## Repository Layout

```text
app.py                 Streamlit dashboard
src/                   Detection, policy, RAG, agent, and monitoring modules
data/                  Knowledge base, datasets, features, and model splits
models/                Trained models, calibrators, and vector indexes
reports/               Evaluation results, audit data, and generated reports
logs/                  Experiment and validation logs
docs/                  Project documentation and known limitations
smoke_test.py          End-to-end integrity check
requirements.txt       Python dependencies
```

## Validation

The smoke test verifies dataset loading, model and calibration artifacts, inference, policy retrieval, and SQLite audit persistence. The project also contains evaluation reports and experiment logs under `reports/` and `logs/`.

Known model limitations, including missed low-signal fraud and calibration overconfidence at the highest score range, are documented in [`docs/known_limitations.md`](docs/known_limitations.md).

## Data and Large Files

Large datasets and model artifacts are stored with Git LFS. Install Git LFS and run `git lfs pull` after cloning if the working tree contains LFS pointer files.

## License

No license has been specified for this project yet.