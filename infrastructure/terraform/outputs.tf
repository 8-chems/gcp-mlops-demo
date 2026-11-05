output "bucket" {
  value = google_storage_bucket.data.name
}

output "cloudsql_connection_name" {
  value = google_sql_database_instance.mlflow.connection_name
}

output "wif_provider" {
  value = google_iam_workload_identity_pool_provider.github.name
}

output "cicd_service_account" {
  value = google_service_account.sa["sa-cicd"].email
}

output "artifact_registry" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}"
}
