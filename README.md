# Failure Before Failure

An educational, local-first prototype for industrial predictive maintenance. It combines a FastAPI backend, a scikit-learn failure classifier, a PyTorch Autoencoder, a transparent Python investigation service, SQLite storage, and a React dashboard. The core system does not call hosted AI APIs.

> **Prototype limitation:** AI4I is synthetic research data. An anomaly is not proof of failure, model outputs are not certified safety signals, and this project does not guarantee that a breakdown will be predicted.

## What is implemented

- Validated industrial sensor readings and analysis through FastAPI.
- A majority-class and logistic-regression baseline, a trained Random Forest classifier, and a normal-pattern Autoencoder.
- A rules-and-evidence investigator based on the documented AI4I synthetic failure patterns.
- SQLite persistence for machines, readings, analysis results, and human-reviewed maintenance actions.
- A React dashboard for machine overview, sensor history, model outputs, investigation evidence, and maintenance review.

## Project layout

```text
backend/
  database/storage.py          SQLite access and schema
  main.py                      FastAPI application
  routes/analysis.py           API endpoints
  schemas/analysis.py          Pydantic request/response types
  services/                    Inference and investigator services
  tests/                       Backend unit and model tests
datasets/
  ai4i2020.csv                 UCI AI4I dataset
models/
  saved/                       Locally trained model artifacts
  training/                    Data preparation and training scripts
frontend/
  src/                         React dashboard
reports/                       Project report and demo materials
requirements.txt                Python dependencies
```

## Run locally on Windows

Prerequisites: Windows, Python 3.13 for the configured CPU-only PyTorch wheel, and Node.js with npm. Use two VS Code PowerShell terminals. Start both services from the project root or the indicated folder.

### 1. Install Python dependencies

If creating the environment from scratch, run this once from the project root:

```powershell
python -m venv venv
```

Then activate it and install dependencies:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the environment's interpreter directly, for example `.\venv\Scripts\python.exe -m pip install -r requirements.txt`.

The requirements select CPU-only PyTorch for this Windows learning setup. In a new environment, train the models before starting the API:

```powershell
python -m models.training.prepare_data
python -m models.training.train_classifier
python -m models.training.train_autoencoder
```

### 2. Start the API

From the project root:

```powershell
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001
```

- Interactive API documentation: <http://127.0.0.1:8001/docs>
- Health check: <http://127.0.0.1:8001/health>

The API loads the classifier, Autoencoder, and SQLite schema at startup. If model artifacts are missing or invalid, startup reports the error; rerun the training commands.

### 3. Start the dashboard

In another terminal:

```powershell
cd .\frontend
npm.cmd install
npm.cmd run dev -- --host 127.0.0.1
```

Open <http://127.0.0.1:5173>. `npm.cmd` avoids PowerShell's script-policy block on `npm.ps1`. The Vite development proxy forwards `/api` requests to the backend at port 8001.

## API workflow

The default machine ID is `demo-machine-001`; callers can supply a different `machine_id`.

1. `POST /sensor-data` validates and stores a reading.
2. `POST /analyze` runs both models and the investigator, then stores the reading and analysis together.
3. `GET /machines` lists known machines; `GET /machines/{machine_id}/readings` returns recent sensor history.
4. `GET /machines/{machine_id}/analysis/latest` returns the latest full investigation, and `GET /failure-risk/{machine_id}` returns the latest model result.
5. `GET /maintenance/{machine_id}` retrieves suggested checks and recorded actions.
6. `POST /maintenance/{machine_id}/actions` records a human review with `action` and `reviewed_by`; `notes` is optional.

Example request body for `/analyze`:

```json
{
  "machine_id": "press-01",
  "product_type": "M",
  "air_temperature_k": 300.0,
  "process_temperature_k": 310.0,
  "rotational_speed_rpm": 1500,
  "torque_nm": 40.0,
  "tool_wear_min": 10
}
```

`POST /analyze` records its own reading. Calling `/sensor-data` first and then analyzing the same physical sample creates two reading records.

The local database file is `backend/database/failure_before_failure.db`. Treat it as local project data; back it up before replacing or deleting it.

## Deploy the dashboard to Vercel and the API to Render

The dashboard is hosted on Vercel and the FastAPI API on Render. The Vercel site cannot use `127.0.0.1` to contact your server. Both services need this project in a Git repository; the Render service also uses a persistent disk for SQLite.

### Deploy the API to Render

1. Push this project to GitHub, including the files in `models/saved/`.
2. In Render, choose **New > Blueprint**, select the GitHub repository, and apply the configuration in `render.yaml`.
3. Wait for the service to become healthy. Its API URL will look like `https://failure-before-failure-api.onrender.com`.
4. After deploying the dashboard, set the Render environment variable `ALLOWED_ORIGINS` to the exact Vercel website origin, without a trailing slash, then redeploy the API.

The Blueprint configures Python 3.13, the API start command, a `/health` check, and a 1 GB persistent disk. The database is stored at `/var/data/failure_before_failure.db` so records survive service restarts and deploys. Render persistent disks require a paid service plan.

For local use, the database remains at `backend/database/failure_before_failure.db`. `DATABASE_PATH` optionally overrides its location, which lets the deployed service use its persistent disk.

```text
ALLOWED_ORIGINS=https://your-project.vercel.app
```

For multiple allowed website origins, separate them with commas. Do not use `*` for a public deployment. The local development origins are allowed by default only when `ALLOWED_ORIGINS` is unset.

### Deploy the React dashboard

1. Import the GitHub repository into Vercel.
2. Set the Vercel **Root Directory** to `frontend`.
3. Use `npm run build` as the build command and `dist` as the output directory (Vercel usually detects these for Vite).
4. Add the environment variable `VITE_API_BASE_URL` with the public base URL of the Render API service, for example `https://failure-before-failure-api.onrender.com`. Do not add a trailing slash or `/api`.
5. Deploy or redeploy after setting the variable. Vite embeds `VITE_*` variables during the build.

Copy the Vercel site's final origin into the Render service's `ALLOWED_ORIGINS` setting, then redeploy the API. The frontend template is in `frontend/.env.example`; the backend template is in `backend/.env.example`. The Vite development proxy remains in use locally.

Deployment requires the project to be available through GitHub and requires access to Render and Vercel accounts. No account credentials or secrets belong in the source files. A public demo has no authentication in this prototype, so do not use it to store private or sensitive machine data.

## Data and model notes

The project uses the [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), a synthetic dataset with 10,000 rows and no reported missing values. It is licensed under CC BY 4.0; cite the UCI Machine Learning Repository when reusing or publishing the data.

Model inputs are product `Type` and five numeric sensor features. IDs and failure-mode labels are excluded from classifier inputs to avoid leakage. The classifier uses a stratified 80/20 split; scaling and categorical encoding are fit inside the training pipeline. The Autoencoder trains only on normal training rows and sets its threshold from normal validation errors, not the held-out test set.

### Recorded evaluation from this workspace

These are results from one run with the checked-in scripts and fixed random seed; they are not claims about live equipment.

| Model | Held-out result |
|---|---|
| Random Forest classifier | Accuracy 0.982; balanced accuracy 0.799; failure precision 0.820; failure recall 0.603; F1 0.695; average precision 0.762; ROC AUC 0.965 |
| Autoencoder | Threshold 0.165453; held-out normal false-alarm rate 6.21% (120/1932); failure recall 0.235 (16/68); anomaly precision 0.118; ROC AUC 0.757 |

The Random Forest confusion matrix was `[[1923, 9], [27, 41]]`: it missed 27 of 68 failure-labelled test rows. The Autoencoder missed 52 of 68. These results show why accuracy alone is insufficient and why an anomaly must not be presented as a predicted breakdown.

The investigator's HDF, PWF, OSF, and TWF patterns mirror AI4I's synthetic data rules. Those thresholds are not universal factory operating limits. The classifier score is not calibrated as a real-world probability, and the system has not been validated on live machinery.

## Tests and checks

From the project root:

```powershell
.\venv\Scripts\Activate.ps1
python -m unittest discover -s backend\tests -v
```

From `frontend`:

```powershell
npm.cmd run lint
npm.cmd run build
```

## Learning milestones and materials

- [Project report](reports/project-report.md)
- [Presentation outline](reports/presentation-outline.md)
- [Short demo script](reports/demo-script.md)

The project is intended for learning and portfolio demonstration, not for safety-critical or unsupervised maintenance decisions.
