locals {
  bucket_name = "${var.project_id}-bucket"
  apis = [
    "run.googleapis.com", "sqladmin.googleapis.com", "bigquery.googleapis.com",
    "storage.googleapis.com", "artifactregistry.googleapis.com", "composer.googleapis.com",
    "secretmanager.googleapis.com", "iam.googleapis.com", "iamcredentials.googleapis.com",
    "logging.googleapis.com", "monitoring.googleapis.com", "cloudresourcemanager.googleapis.com",
    "sts.googleapis.com", "compute.googleapis.com",
  ]
  # Service accounts that run workloads (CI/CD must be able to "act as" them when deploying)
  runtime_sas = ["sa-mlflow", "sa-training", "sa-api", "sa-dashboard", "sa-airflow"]
  all_sas     = concat(local.runtime_sas, ["sa-cicd"])
}

resource "google_project_service" "apis" {
  for_each           = toset(local.apis)
  service            = each.value
  disable_on_destroy = false
}

# ---------- Cloud Storage ----------
resource "google_storage_bucket" "data" {
  name                        = local.bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true # POC: allows `terraform destroy` with content
  depends_on                  = [google_project_service.apis]
}

# ---------- BigQuery ----------
resource "google_bigquery_dataset" "mlops" {
  dataset_id                 = "mlops_demo"
  location                   = var.bq_location
  delete_contents_on_destroy = true
  depends_on                 = [google_project_service.apis]
}

resource "google_bigquery_table" "prediction_results" {
  dataset_id          = google_bigquery_dataset.mlops.dataset_id
  table_id            = "prediction_results"
  deletion_protection = false
  schema = jsonencode([
    { name = "prediction_id", type = "STRING", mode = "REQUIRED" },
    { name = "timestamp", type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "model_name", type = "STRING", mode = "NULLABLE" },
    { name = "model_version", type = "STRING", mode = "NULLABLE" },
    { name = "prediction", type = "INTEGER", mode = "NULLABLE" },
    { name = "probability", type = "FLOAT", mode = "NULLABLE" },
    { name = "features", type = "STRING", mode = "NULLABLE" },
  ])
}
# NB: `training_data` is created by the pipeline (load_bigquery step).

# ---------- Artifact Registry ----------
resource "google_artifact_registry_repository" "docker" {
  repository_id = "mlops"
  location      = var.region
  format        = "DOCKER"
  depends_on    = [google_project_service.apis]
}

# ---------- Service accounts ----------
resource "google_service_account" "sa" {
  for_each     = toset(local.all_sas)
  account_id   = each.value
  display_name = each.value
  depends_on   = [google_project_service.apis]
}

locals {
  sa_email = { for k, v in google_service_account.sa : k => v.email }
}

# MLflow: Cloud SQL + artifacts bucket + DB password secret
resource "google_project_iam_member" "mlflow_sql" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${local.sa_email["sa-mlflow"]}"
}

resource "google_storage_bucket_iam_member" "bucket_rw" {
  for_each = toset(["sa-mlflow", "sa-training"])
  bucket   = google_storage_bucket.data.name
  role     = "roles/storage.objectAdmin"
  member   = "serviceAccount:${local.sa_email[each.value]}"
}

# Training job: BigQuery read/write + jobs
resource "google_bigquery_dataset_iam_member" "training_editor" {
  dataset_id = google_bigquery_dataset.mlops.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${local.sa_email["sa-training"]}"
}

resource "google_bigquery_dataset_iam_member" "api_editor" {
  dataset_id = google_bigquery_dataset.mlops.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${local.sa_email["sa-api"]}"
}

resource "google_bigquery_dataset_iam_member" "dashboard_viewer" {
  dataset_id = google_bigquery_dataset.mlops.dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = "serviceAccount:${local.sa_email["sa-dashboard"]}"
}

resource "google_project_iam_member" "bq_jobs" {
  for_each = toset(["sa-training", "sa-dashboard"])
  project  = var.project_id
  role     = "roles/bigquery.jobUser"
  member   = "serviceAccount:${local.sa_email[each.value]}"
}

# Airflow (Composer worker) can run Cloud Run Jobs and act as the training SA
resource "google_project_iam_member" "airflow_roles" {
  for_each = toset(["roles/composer.worker", "roles/run.developer"])
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${local.sa_email["sa-airflow"]}"
}

resource "google_service_account_iam_member" "airflow_acts_as_training" {
  service_account_id = google_service_account.sa["sa-training"].name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${local.sa_email["sa-airflow"]}"
}

# CI/CD: push images, deploy Cloud Run, act as runtime SAs
resource "google_project_iam_member" "cicd_roles" {
  for_each = toset(["roles/artifactregistry.writer", "roles/run.admin"])
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${local.sa_email["sa-cicd"]}"
}

resource "google_service_account_iam_member" "cicd_acts_as" {
  for_each           = toset(local.runtime_sas)
  service_account_id = google_service_account.sa[each.value].name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${local.sa_email["sa-cicd"]}"
}

# ---------- Cloud SQL (MLflow backend store) ----------
resource "random_password" "db" {
  length  = 24
  special = false # keeps the SQLAlchemy URL simple (no escaping needed)
}

resource "google_sql_database_instance" "mlflow" {
  name                = "mlflow-pg"
  region              = var.region
  database_version    = "POSTGRES_15"
  deletion_protection = false
  settings {
    tier              = var.sql_tier
    edition           = "ENTERPRISE"
    availability_type = "ZONAL"
    disk_size         = 10
    backup_configuration { enabled = false }
  }
  depends_on = [google_project_service.apis]
}

resource "google_sql_database" "mlflow" {
  name     = "mlflow"
  instance = google_sql_database_instance.mlflow.name
}

resource "google_sql_user" "mlflow" {
  name     = "mlflow"
  instance = google_sql_database_instance.mlflow.name
  password = random_password.db.result
}

resource "google_secret_manager_secret" "db_password" {
  secret_id = "mlflow-db-password"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "db_password" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = random_password.db.result
}

resource "google_secret_manager_secret_iam_member" "mlflow_reads_secret" {
  secret_id = google_secret_manager_secret.db_password.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${local.sa_email["sa-mlflow"]}"
}

# ---------- Workload Identity Federation for GitHub Actions ----------
resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github-pool"
  display_name              = "GitHub Actions"
  depends_on                = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-provider"
  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
  }
  attribute_condition = "assertion.repository == '${var.github_repo}'"
  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account_iam_member" "github_wif" {
  service_account_id = google_service_account.sa["sa-cicd"].name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repo}"
}
