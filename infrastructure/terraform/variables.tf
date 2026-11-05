variable "project_id" {
  type        = string
  description = "GCP project ID (dedicated project recommended)"
}

variable "region" {
  type    = string
  default = "europe-west1"
}

variable "bq_location" {
  type    = string
  default = "EU"
}

variable "github_repo" {
  type        = string
  description = "GitHub repository allowed to deploy, format OWNER/REPO"
}

variable "sql_tier" {
  type    = string
  default = "db-f1-micro"
}
