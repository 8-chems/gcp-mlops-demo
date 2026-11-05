# GCP MLOps demo: Airflow + MLflow + XGBoost + Cloud Run

An end-to-end MLOps proof of concept on Google Cloud. The ML model (XGBoost on the UCI *Bank
Marketing* dataset) is only a vehicle: the goal is to **orchestrate, track, deploy, monitor
and reproduce** an ML workflow.

![Architecture](architecture.png)

Design decisions worth knowing:

* **Airflow orchestrates, it does not compute.** Each DAG task triggers the same Cloud Run Job
  with a different sub-command (`python -m src.cli <step>`), so Composer needs no ML dependencies.
* **Deploying code is not deploying a model.** GitHub Actions deploys application code. A new model
  goes live when the pipeline registers a better version and moves the MLflow alias
  `champion`; the DAG then calls `POST /reload` on the API.
* **Promotion guard.** The best of 3 runs (by ROC-AUC) must reach `MIN_ROC_AUC` (default 0.75)
  **and** beat the current champion to be promoted.
* **MLflow and the API are private Cloud Run services.** Callers send Google ID tokens
  (`src/utils/auth.py`); the proper service accounts get `roles/run.invoker`.
* **No long-lived keys.** GitHub Actions authenticates with Workload Identity Federation.

```
.
├── airflow/dags/xgboost_pipeline.py   # DAG xgboost_training_pipeline
├── src/                               # pipeline code (data, features, models, utils) + CLI
├── training/Dockerfile                # image of the Cloud Run Job
├── mlflow_server/                     # MLflow tracking server image (Cloud SQL + GCS)
├── api/                               # FastAPI prediction service
├── dashboard/                         # Streamlit app
├── infrastructure/terraform/          # APIs, GCS, BigQuery, AR, SAs/IAM, Cloud SQL, WIF
├── tests/  scripts/  .github/workflows/{ci,cd}.yml
```

---

## How to read this guide

All commands are for **Windows PowerShell**.

**Every value lives in one place: the file `env.ps1` created in Step 2.** All later commands use
variables (`$PROJECT_ID`, `$REGION`, `$BUCKET`, ...), so a typo can only happen once, in that file.

| Variable | Meaning | Value |
|---|---|---|
| `$PROJECT_ID` | GCP project ID | `mlops-demo-gcp` |
| `$REGION` | GCP region | `europe-west1` |
| `$GITHUB_REPO` | GitHub repository (`OWNER/REPO`) | `8-chems/gcp-mlops-demo` |
| `$BILLING` | Your billing account ID | the only value you must fill in yourself |
| `$MY_EMAIL` | Your Google account | detected automatically |
| `$BUCKET` | Data bucket created by Terraform | `mlops-demo-gcp-bucket` |
| `$REGISTRY` | Artifact Registry path | `europe-west1-docker.pkg.dev/mlops-demo-gcp/mlops` |
| `$SQL_CONN` | Cloud SQL connection name | `mlops-demo-gcp:europe-west1:mlflow-pg` |

Rules of thumb:

* Run every command from the **project root** (the folder containing `requirements.txt`) unless a step says otherwise.
* **New PowerShell window? Run `. .\env.ps1` first** (dot, space, dot-backslash). Variables vanish when a window closes. If a command fails with `expected one argument` or an empty value, this is almost always the reason. The helper `Check-Env` (defined in `env.ps1`) tells you which variables are missing.
* A backtick `` ` `` at the very end of a line continues the command. **Nothing, not even a space, may follow it.**
* In PowerShell, `curl` is an alias of another command: this guide uses `curl.exe`.
* Docker Desktop must be **running** for Step 6.

---

## 0. Prerequisites

| Tool | Check |
|---|---|
| A Google account with billing enabled | |
| Google Cloud CLI (`gcloud`, includes `bq`) | `gcloud --version` |
| Terraform ≥ 1.5 | `terraform -version` |
| Docker Desktop (WSL 2 backend) | `docker --version` |
| **Python 3.11** (3.13 is *not* supported by the pinned versions) | `py -3.11 --version` |
| Git | `git --version` |
| GitHub CLI | `gh --version` |

Install with winget. The `--source winget` option avoids the Microsoft Store source, which can fail with a certificate error:

```powershell
winget install --id Python.Python.3.11 --source winget
winget install --id Google.CloudSDK --source winget
winget install --id Hashicorp.Terraform --source winget
winget install --id Docker.DockerDesktop --source winget
winget install --id Git.Git --source winget
winget install --id GitHub.cli --source winget
```

Close and reopen PowerShell afterwards. Python 3.13 can stay installed: the `py` launcher lets you choose 3.11 per project (`py -0` lists both).

**Cost warning.** Cloud Composer (about $10+/day) and Cloud SQL (about $10/month) bill while they exist, even idle. Cloud Run scales to zero. Do Steps 1-10 first, create Composer only in Step 11, and run **Step 15 (cleanup)** when done. Set a budget alert in *Billing → Budgets & alerts*.

**Certificate errors?** If `pip`, `gcloud`, `terraform`, `git` or `docker build` fail with `unable to get local issuer certificate` / `CERTIFICATE_VERIFY_FAILED`, your antivirus or network is re-signing HTTPS traffic. See the first row of the Troubleshooting table before continuing.

---

## 1. Run everything locally first (free)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python --version          # must print 3.11.x
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and retry.

Make ruff treat `airflow` as an external library (otherwise it reports an import-order error because the project has an `airflow/` folder), then lint and test:

```powershell
Add-Content pyproject.toml "`n[tool.ruff.lint.isort]`nknown-third-party = [`"airflow`"]"
ruff check . --fix
ruff check .
pytest -q
```

`ruff check .` should print `All checks passed!` and `pytest -q` should report `13 passed`.

Optional: run the whole ML flow locally (clean, split, train 3 XGBoost runs, evaluate, register) on a throw-away SQLite MLflow:

```powershell
python scripts\download_data.py
$env:MIN_ROC_AUC = "0.5"
python scripts\local_smoke_test.py
```

The smoke test ends with `OK`. To browse the 3 runs, go **into the folder it created** and use a relative path (absolute SQLite paths are easy to get wrong on Windows):

```powershell
$SMOKE = Get-ChildItem $env:TEMP -Directory -Filter "mlops-smoke-*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
cd $SMOKE.FullName
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://localhost:5000, stop with Ctrl+C, then come back with `cd` to your project folder. Without `data\bank-additional-full.csv` the smoke test uses synthetic data.

Fix a line-ending trap now: shell scripts must use Unix line endings or the MLflow container will not start.

```powershell
Set-Content .gitattributes "*.sh text eol=lf" -Encoding ascii
(Get-Content mlflow_server\entrypoint.sh -Raw) -replace "`r`n","`n" | Set-Content -NoNewline mlflow_server\entrypoint.sh
```

---

## 2. Create `env.ps1` (all variables in one file)

First look up your billing account ID:

```powershell
gcloud billing accounts list
```

Create the file. **Replace `XXXXXX-XXXXXX-XXXXXX` by your billing account ID** (the only value to edit), then run the block:

```powershell
@'
# ---- edit only $BILLING ----
$BILLING      = "XXXXXX-XXXXXX-XXXXXX"

# ---- project settings ----
$PROJECT_ID   = "mlops-demo-gcp"
$REGION       = "europe-west1"
$GITHUB_REPO  = "8-chems/gcp-mlops-demo"

# ---- derived names (do not edit) ----
$MY_EMAIL     = gcloud config get-value account 2>$null
$BUCKET       = "$PROJECT_ID-bucket"
$REGISTRY     = "$REGION-docker.pkg.dev/$PROJECT_ID/mlops"
$SQL_CONN     = "${PROJECT_ID}:${REGION}:mlflow-pg"
$JOB          = "mlops-pipeline-job"
$COMPOSER_ENV = "mlops-demo"
$SA_MLFLOW    = "sa-mlflow@$PROJECT_ID.iam.gserviceaccount.com"
$SA_TRAINING  = "sa-training@$PROJECT_ID.iam.gserviceaccount.com"
$SA_API       = "sa-api@$PROJECT_ID.iam.gserviceaccount.com"
$SA_DASHBOARD = "sa-dashboard@$PROJECT_ID.iam.gserviceaccount.com"
$SA_AIRFLOW   = "sa-airflow@$PROJECT_ID.iam.gserviceaccount.com"
$CICD_SA      = "sa-cicd@$PROJECT_ID.iam.gserviceaccount.com"

function Check-Env([string[]]$Names) {
  $missing = $Names | Where-Object { -not (Get-Variable $_ -ValueOnly -ErrorAction SilentlyContinue) }
  if ($missing) { Write-Host "MISSING: $($missing -join ', ')  -> run: . .\env.ps1" -ForegroundColor Red }
  else { Write-Host "OK: all variables set" -ForegroundColor Green }
}
'@ | Set-Content env.ps1 -Encoding ascii

Add-Content .gitignore "env.ps1"
Add-Content .gitignore "payload.json"

. .\env.ps1
Check-Env PROJECT_ID,REGION,GITHUB_REPO,BILLING,MY_EMAIL,BUCKET,REGISTRY,SQL_CONN
$PROJECT_ID; $REGION; $GITHUB_REPO; $MY_EMAIL; $BUCKET; $REGISTRY; $SQL_CONN
```

`$MY_EMAIL` is only filled after `gcloud auth login` (Step 3); if it is empty now, run `. .\env.ps1` again after logging in.
Later steps append `WIF_PROVIDER`, `MLFLOW_URL` and `API_URL` to this same file.

## 3. Create the GCP project and link billing

```powershell
. .\env.ps1
gcloud auth login
gcloud auth application-default login
. .\env.ps1                      # refresh $MY_EMAIL now that you are logged in
gcloud projects create $PROJECT_ID
gcloud config set project $PROJECT_ID
gcloud billing projects link $PROJECT_ID --billing-account=$BILLING
gcloud auth application-default set-quota-project $PROJECT_ID
gcloud projects list
```

* If `projects create` says the project already exists (and it is yours), continue.
* If the ID is taken globally, change `$PROJECT_ID` once in `env.ps1` and run `. .\env.ps1`: every other name derives from it.
* If linking fails with a quota error, your billing account has too many linked projects. See *Troubleshooting → billing quota*.

## 4. Provision the infrastructure with Terraform

Creates: required APIs, GCS bucket, BigQuery dataset `mlops_demo` (+ `prediction_results`), Artifact Registry repo `mlops`, 6 service accounts with least-privilege IAM, Cloud SQL PostgreSQL for MLflow, the DB password in Secret Manager, and Workload Identity Federation for GitHub.

`terraform.tfvars` is generated from your variables, so it always matches `env.ps1`:

```powershell
. .\env.ps1
cd infrastructure\terraform
@"
project_id  = "$PROJECT_ID"
region      = "$REGION"
github_repo = "$GITHUB_REPO"
"@ | Set-Content terraform.tfvars -Encoding ascii
Get-Content terraform.tfvars

# The Terraform code names the bucket "<project_id>-mlops-demo". Rename it to "<project_id>-bucket":
(Get-Content main.tf -Raw).Replace('${var.project_id}-mlops-demo','${var.project_id}-bucket') | Set-Content main.tf -NoNewline -Encoding ascii
Select-String -Path main.tf -Pattern "bucket_name ="

terraform init
terraform apply
```

`Select-String` must show `bucket_name = "${var.project_id}-bucket"`.

> **Already ran `terraform apply` before this change?** Terraform will replace the old (empty) bucket `mlops-demo-gcp-mlops-demo` with `mlops-demo-gcp-bucket`. If you had already uploaded the dataset (Step 5), upload it again after the apply.

Type `yes` when asked (about 8-12 minutes; Cloud SQL is the slow part). If it fails with *API not enabled*, wait one minute and run `terraform apply` again.

> **Already ran `terraform apply` with a different repository name?** The GitHub repository allowed to deploy is stored in Google Cloud. Fix `$GITHUB_REPO` in `env.ps1`, rerun the block above and `terraform apply` again: it updates this setting in place (no resources are recreated).

Read the outputs and go back to the project root:

```powershell
$WIF_PROVIDER = terraform output -raw wif_provider
terraform output -raw bucket
terraform output -raw cloudsql_connection_name
cd ..\..
Add-Content env.ps1 "`$WIF_PROVIDER = '$WIF_PROVIDER'"
$WIF_PROVIDER
```

The bucket printed by Terraform must equal `$BUCKET`, and the connection name must equal `$SQL_CONN`. `$WIF_PROVIDER` must look like `projects/123456789/locations/global/workloadIdentityPools/github-pool/providers/github-provider`.

## 5. Upload the dataset to Cloud Storage

```powershell
. .\env.ps1
python scripts\download_data.py
gcloud storage cp data\bank-additional-full.csv "gs://$BUCKET/raw/bank-additional-full.csv"
gcloud storage ls "gs://$BUCKET/raw/"
```

(Skip the first line if `data\bank-additional-full.csv` already exists.) The pipeline samples `SAMPLE_ROWS=15000` of the ~41k rows (fixed seed 42).

## 6. Build and push the Docker images

Start **Docker Desktop** and wait until it reports that the engine is running. Check with `docker info` (it must print server information, not an error).

```powershell
. .\env.ps1
Check-Env REGISTRY,REGION
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet

docker build --platform linux/amd64 -t "$REGISTRY/mlflow:latest" mlflow_server
docker build --platform linux/amd64 -t "$REGISTRY/training:latest" -f training/Dockerfile .
docker build --platform linux/amd64 -t "$REGISTRY/api:latest" -f api/Dockerfile .
docker build --platform linux/amd64 -t "$REGISTRY/dashboard:latest" -f dashboard/Dockerfile .

foreach ($i in "mlflow","training","api","dashboard") {
  docker push "$REGISTRY/${i}:latest"
}

gcloud artifacts docker images list $REGISTRY
```

The last command must list the four images.

## 7. Deploy the MLflow tracking server (Cloud Run + Cloud SQL + GCS)

```powershell
. .\env.ps1
Check-Env PROJECT_ID,REGION,REGISTRY,BUCKET,SQL_CONN,SA_MLFLOW

$MLFLOW_ENV = "DB_USER=mlflow,DB_NAME=mlflow,CLOUDSQL_CONNECTION_NAME=${SQL_CONN},ARTIFACT_ROOT=gs://${BUCKET}/mlflow-artifacts"

gcloud run deploy mlflow --region $REGION --image "$REGISTRY/mlflow:latest" `
  --service-account $SA_MLFLOW `
  --no-allow-unauthenticated `
  --add-cloudsql-instances $SQL_CONN `
  --set-env-vars $MLFLOW_ENV `
  --set-secrets "DB_PASSWORD=mlflow-db-password:latest" `
  --memory 1Gi --cpu 1 --min-instances 0 --max-instances 2
```

Then read the service URL (only after the deploy succeeded) and save it:

```powershell
$MLFLOW_URL = gcloud run services describe mlflow --region $REGION --format "value(status.url)"
Add-Content env.ps1 "`$MLFLOW_URL = '$MLFLOW_URL'"
$MLFLOW_URL
```

The service is private. Give your account permission, then use the authenticated proxy and open http://localhost:5000 (Ctrl+C to stop):

```powershell
gcloud run services add-iam-policy-binding mlflow --region $REGION --member "user:$MY_EMAIL" --role roles/run.invoker
gcloud run services proxy mlflow --region $REGION --port 5000
```

## 8. Create the pipeline Cloud Run Job and wire permissions

```powershell
. .\env.ps1
Check-Env PROJECT_ID,REGION,REGISTRY,BUCKET,MLFLOW_URL,SA_TRAINING,JOB

$JOB_ENV = "GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION},GCS_BUCKET=${BUCKET},BQ_DATASET=mlops_demo,MLFLOW_TRACKING_URI=${MLFLOW_URL},MIN_ROC_AUC=0.75,SAMPLE_ROWS=15000"

gcloud run jobs create $JOB --region $REGION `
  --image "$REGISTRY/training:latest" `
  --service-account $SA_TRAINING `
  --set-env-vars $JOB_ENV `
  --memory 2Gi --cpu 1 --max-retries 0 --task-timeout 1800
```

Allow the service accounts to call the private MLflow service:

```powershell
foreach ($sa in $SA_TRAINING,$SA_API,$SA_DASHBOARD) {
  gcloud run services add-iam-policy-binding mlflow --region $REGION `
    --member "serviceAccount:$sa" --role roles/run.invoker
}
```

## 9. Milestone 1: run the pipeline by hand (no Airflow yet)

Each step runs as a Cloud Run Job execution; this validates GCS, BigQuery, MLflow and permissions.

```powershell
. .\env.ps1
$RUN_ID = "manual1"
foreach ($s in "extract","validate","preprocess","load","train","evaluate","register") {
  Write-Host "=== $s ==="
  gcloud run jobs execute $JOB --region $REGION "--args=${s},--run-id,${RUN_ID}" --wait
  if ($LASTEXITCODE -ne 0) { Write-Host "Step $s failed, stopping." -ForegroundColor Red; break }
}
```

Expected result: `processed/clean.csv` in the bucket, table `mlops_demo.training_data` in BigQuery, 3 runs in the MLflow experiment `bank-marketing-xgboost`, and model `bank-marketing-xgb` version 1 with alias `champion`. Verify:

```powershell
bq query --use_legacy_sql=false "SELECT split, COUNT(*) AS n, AVG(target) AS rate FROM mlops_demo.training_data GROUP BY split"
```

If a step fails, read the job logs (Step 13): the failing step logs `pipeline_failure` with the error. A failing `evaluate` (`ThresholdNotMet`) is a *feature*: the model is below `MIN_ROC_AUC` and is not registered.

## 10. Milestone 2: deploy the API and the dashboard

```powershell
. .\env.ps1
Check-Env PROJECT_ID,REGION,REGISTRY,MLFLOW_URL,SA_API,SA_DASHBOARD,SA_AIRFLOW

$API_ENV = "GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION},MLFLOW_TRACKING_URI=${MLFLOW_URL},LOG_PREDICTIONS_TO_BQ=true"

gcloud run deploy api --region $REGION --image "$REGISTRY/api:latest" `
  --service-account $SA_API `
  --no-allow-unauthenticated --min-instances 0 --memory 1Gi `
  --set-env-vars $API_ENV

$API_URL = gcloud run services describe api --region $REGION --format "value(status.url)"
Add-Content env.ps1 "`$API_URL = '$API_URL'"
$API_URL
```

The dashboard, Airflow and you must be allowed to call the private API:

```powershell
foreach ($sa in $SA_DASHBOARD,$SA_AIRFLOW) {
  gcloud run services add-iam-policy-binding api --region $REGION `
    --member "serviceAccount:$sa" --role roles/run.invoker
}
gcloud run services add-iam-policy-binding api --region $REGION --member "user:$MY_EMAIL" --role roles/run.invoker
```

Deploy the dashboard:

```powershell
$DASH_ENV = "GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION},MLFLOW_TRACKING_URI=${MLFLOW_URL},API_URL=${API_URL}"

gcloud run deploy dashboard --region $REGION --image "$REGISTRY/dashboard:latest" `
  --service-account $SA_DASHBOARD `
  --allow-unauthenticated --min-instances 0 --memory 1Gi `
  --set-env-vars $DASH_ENV

gcloud run services describe dashboard --region $REGION --format "value(status.url)"
```

Test the API. First get an identity token, tell the API to load the champion model, and check its health:

```powershell
$TOKEN = gcloud auth print-identity-token
curl.exe -s -X POST "$API_URL/reload" -H "Authorization: Bearer $TOKEN"
curl.exe -s "$API_URL/health" -H "Authorization: Bearer $TOKEN"
```

Create the request file and send a prediction:

```powershell
@'
{
  "age": 40, "job": "admin.", "marital": "married", "education": "high.school",
  "default": "no", "housing": "yes", "loan": "no", "contact": "cellular", "month": "may",
  "day_of_week": "mon", "campaign": 2, "pdays": 999, "previous": 0, "poutcome": "nonexistent",
  "emp_var_rate": 1.1, "cons_price_idx": 93.9, "cons_conf_idx": -36.4, "euribor3m": 4.8,
  "nr_employed": 5191
}
'@ | Set-Content payload.json -Encoding ascii

curl.exe -s -X POST "$API_URL/predict" -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" --data "@payload.json"
```

The token lasts about an hour; rerun `$TOKEN = gcloud auth print-identity-token` to refresh it.

> **Security note.** The dashboard is public to keep the demo simple. To restrict it, deploy it with `--no-allow-unauthenticated` (or put Identity-Aware Proxy in front). The API and MLflow stay private.

Open the dashboard URL: *Data overview* (BigQuery), *Model experiments* (MLflow), *Model performance* (champion artifacts) and *Predictions* (calls the API).

## 11. Airflow on Cloud Composer (temporary environment)

```powershell
. .\env.ps1
Check-Env PROJECT_ID,REGION,API_URL,JOB,SA_AIRFLOW,COMPOSER_ENV

$COMPOSER_VARS = "GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION},PIPELINE_JOB_NAME=${JOB},API_URL=${API_URL}"

gcloud composer environments create $COMPOSER_ENV --location $REGION `
  --environment-size small `
  --service-account $SA_AIRFLOW `
  --env-variables $COMPOSER_VARS
```

This takes **20-30 minutes**. Then deploy the DAG and run it:

```powershell
gcloud composer environments storage dags import --environment $COMPOSER_ENV --location $REGION --source airflow\dags\xgboost_pipeline.py
Start-Sleep -Seconds 120
gcloud composer environments run $COMPOSER_ENV --location $REGION dags list
gcloud composer environments run $COMPOSER_ENV --location $REGION dags unpause "--" xgboost_training_pipeline
gcloud composer environments run $COMPOSER_ENV --location $REGION dags trigger "--" xgboost_training_pipeline
gcloud composer environments describe $COMPOSER_ENV --location $REGION --format "value(config.airflowUri)"
```

Open the Airflow UI URL to watch the 9 tasks: `extract_data → validate_data → preprocess_data → load_bigquery → trigger_training → evaluate_model → register_model → reload_api → pipeline_report`. Trigger it a second time: the new model is only promoted if its ROC-AUC beats the champion.

(The quoted `"--"` is passed to `gcloud` untouched; PowerShell would otherwise interpret it.)

## 12. CI/CD with GitHub Actions

Variables and secrets are created **before** the first push, because a push to `main` triggers the CD workflow, which needs them.

**a) Log in and create the private repository (without pushing yet)**

```powershell
. .\env.ps1
Check-Env GITHUB_REPO,PROJECT_ID,REGION,WIF_PROVIDER,CICD_SA
gh auth login
git init
git add .
git commit -m "Initial MLOps demo"
git branch -M main
gh repo create $GITHUB_REPO --private --source=. --remote=origin
```

If the repository already exists, use `git remote add origin "https://github.com/$GITHUB_REPO.git"` instead of `gh repo create`.

**b) Add the variables and secrets with `gh`**

```powershell
gh variable set GCP_PROJECT_ID --repo $GITHUB_REPO --body $PROJECT_ID
gh variable set GCP_REGION --repo $GITHUB_REPO --body $REGION
gh secret set WIF_PROVIDER --repo $GITHUB_REPO --body $WIF_PROVIDER
gh secret set WIF_SERVICE_ACCOUNT --repo $GITHUB_REPO --body $CICD_SA

gh variable list --repo $GITHUB_REPO
gh secret list --repo $GITHUB_REPO
```

Prefer the web UI? Same values under *Settings → Secrets and variables → Actions*: **Variables** `GCP_PROJECT_ID`, `GCP_REGION`; **Secrets** `WIF_PROVIDER`, `WIF_SERVICE_ACCOUNT`.

**c) Push, which triggers the first CD run**

```powershell
git push -u origin main
gh run list --repo $GITHUB_REPO --limit 5
gh run watch --repo $GITHUB_REPO
```

**d) Try the full loop (optional)**

```powershell
git checkout -b test-ci
Add-Content README.md "# test"
git commit -am "Test CI"
git push -u origin test-ci
gh pr create --repo $GITHUB_REPO --fill
gh pr checks --repo $GITHUB_REPO --watch
```

**CI** runs ruff, pytest (including a tiny XGBoost training and the API) and builds the 4 Docker images. Merging (`gh pr merge --squash --repo $GITHUB_REPO`) triggers **CD**: it builds and pushes images tagged with the commit SHA, updates the Cloud Run Job, deploys the API and dashboard and runs health checks. CD works because Steps 6-10 already created the services it updates. No JSON key is stored in GitHub: only the WIF provider name and the service-account e-mail.

## 13. Observability with Cloud Logging

All components print one JSON object per event; Cloud Logging indexes `jsonPayload.event`. In *Logs Explorer* (Google Cloud console → Logging) try:

```text
-- failed pipeline steps
jsonPayload.event="pipeline_failure"
-- all errors
severity>=ERROR
-- training runs and their metrics
jsonPayload.event="model_training_completed"
-- predictions served by the API
resource.type="cloud_run_revision" jsonPayload.event="model_prediction"
-- cloud run job logs only
resource.type="cloud_run_job" resource.labels.job_name="mlops-pipeline-job"
-- Airflow task logs
resource.type="cloud_composer_environment"
```

From PowerShell (the `:` operator means "contains" and avoids inner double quotes, which PowerShell mangles when calling `gcloud`):

```powershell
gcloud logging read "jsonPayload.event:pipeline_failure" --limit 10 --freshness 1d
gcloud run services logs read mlflow --region $REGION --limit 50
gcloud run jobs executions list --job $JOB --region $REGION
```

Events emitted: `pipeline_step_started/completed/failure`, `data_extracted`, `data_validated`, `data_preprocessed`, `data_loaded_bigquery`, `model_training_started/completed`, `model_evaluation_passed/failed`, `model_registered`, `model_loaded`, `model_prediction`, `api_connection_failure`, `dashboard_error`, `pipeline_report`, `api_reloaded`.

Optional alert on pipeline failures (log-based metric, then an alert policy in *Monitoring → Alerting*):

```powershell
gcloud logging metrics create pipeline_failures --description "Failed pipeline steps" --log-filter "jsonPayload.event:pipeline_failure"
```

## 14. Final end-to-end check (definition of success)

1. Push a change: CI runs the tests. 2. Merge: CD deploys. 3. Trigger the DAG in Airflow. 4. The job trains 3 XGBoost models and MLflow records parameters, metrics and artifacts. 5. The best model is registered and promoted to `champion`. 6. `reload_api` makes the API serve it (`GET /model-info`). 7. The dashboard shows experiments and answers predictions. 8. Logs Explorer shows the whole run.

## 15. Cleanup (do not skip: Composer and Cloud SQL bill continuously)

```powershell
. .\env.ps1
gcloud composer environments delete $COMPOSER_ENV --location $REGION --quiet
foreach ($s in "dashboard","api","mlflow") { gcloud run services delete $s --region $REGION --quiet }
gcloud run jobs delete $JOB --region $REGION --quiet
gcloud logging metrics delete pipeline_failures --quiet
cd infrastructure\terraform
terraform destroy
cd ..\..
```

Composer also creates a bucket named like `europe-west1-mlops-demo-*`: delete it if it remains (`gcloud storage rm -r gs://BUCKET_NAME`). To delete everything at once: `gcloud projects delete $PROJECT_ID`.

To pause only the expensive parts between sessions: delete Composer (recreate in about 25 minutes) and stop Cloud SQL (`gcloud sql instances patch mlflow-pg --activation-policy=NEVER`, back with `--activation-policy=ALWAYS`).

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `unable to get local issuer certificate` / `CERTIFICATE_VERIFY_FAILED` (pip, gcloud, terraform, git, docker build) | Antivirus HTTPS scanning, proxy or VPN re-signs traffic. Test on a phone hotspot, or turn off HTTPS scanning. For `gcloud`: export the proxy's root certificate to `C:\certs\corp-ca.pem` and run `gcloud config set core/custom_ca_certs_file C:\certs\corp-ca.pem`. For other tools set `$env:SSL_CERT_FILE`, `git config --global http.sslCAInfo C:\certs\corp-ca.pem`, `pip config set global.cert C:\certs\corp-ca.pem`. |
| `pip` fails building `numpy` | You are not on Python 3.11. Recreate the venv with `py -3.11 -m venv .venv` (Step 1). |
| `--region: expected one argument`, empty values, `Cannot bind argument` | Variables are not loaded (new window). Run `. .\env.ps1`, then `Check-Env PROJECT_ID,REGION,BUCKET,REGISTRY`. |
| `MLFLOW_URL` / `API_URL` / `WIF_PROVIDER` empty | The step that sets them did not succeed or was not run yet. Redo Step 7, 10 or 4, and check `env.ps1` contains the line. |
| Terraform or `gcloud` says a bucket / repo is not found | Names differ from `env.ps1`. `terraform output -raw bucket` is the source of truth for the bucket. |
| `gcloud billing projects link` quota error | The billing account has too many linked projects. List them with `gcloud billing projects list --billing-account=$BILLING` and unlink unused ones with `gcloud billing projects unlink PROJECT_ID` (unlinked projects lose paid services). |
| `docker`: `failed to connect to the docker API` | Docker Desktop is not running. Start it and wait until `docker info` works. If it will not start: `wsl --install`, `wsl --update`, enable virtualization in BIOS. |
| MLflow revision fails with `exec /entrypoint.sh: no such file or directory` | Windows line endings in `mlflow_server\entrypoint.sh`. Convert with the command in Step 1, rebuild and push the image, redeploy. |
| `mlflow ui` says `unable to open database file` | Wrong SQLite path. `cd` into the smoke-test folder and use `sqlite:///mlflow.db` (Step 1). |
| `403 Forbidden` calling MLflow/API | Missing `roles/run.invoker` for the caller (Steps 7, 8, 10), or an expired ID token (refresh `$TOKEN`). |
| MLflow revision fails to start | `gcloud run services logs read mlflow --region $REGION`: check the Cloud SQL instance is `RUNNABLE`, the secret exists, and `--add-cloudsql-instances` is set. |
| `evaluate` fails with `ThresholdNotMet` | Model below `MIN_ROC_AUC`. Lower it: `gcloud run jobs update $JOB --region $REGION --update-env-vars MIN_ROC_AUC=0.7`. |
| API returns `503 No model loaded` | No champion yet: run the pipeline, then `POST /reload`. |
| DAG not shown in Airflow | Wait 2-3 min; check *DAG import errors* in the UI; ensure the Composer env variables exist. |
| `CloudRunExecuteJobOperator` 403 | `sa-airflow` needs `roles/run.developer` and `serviceAccountUser` on `sa-training` (Terraform grants both). |
| CD cannot authenticate | Check the secrets (`gh secret list`) and that `github_repo` in `terraform.tfvars` equals `$GITHUB_REPO` exactly; if not, fix it and run `terraform apply` again. |
| BigQuery `Access Denied` | Wait about 1 minute after `terraform apply`; IAM changes propagate with a delay. |

## Tests

`pytest -q` covers: data validation, preprocessing (leakage column removed, reproducible stratified split), XGBoost training + MLflow logging, threshold gate and champion promotion, and the API (health, predict, input validation, no-model case). `ruff check .` enforces code quality.