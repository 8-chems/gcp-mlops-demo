<#
.SYNOPSIS
    Creates the Streamlit auth secret in GCP Secret Manager for the dashboard service.

.DESCRIPTION
    - Generates a cryptographically random cookie secret
    - Builds secrets.toml with Google OAuth credentials
    - Uploads the file to Secret Manager
    - Deletes the local file immediately (never committed)
    - Grants the dashboard service account read access

.PARAMETER GoogleClientId
    Google OAuth 2.0 client ID  (from GCP Console → APIs & Services → Credentials)

.PARAMETER GoogleClientSecret
    Google OAuth 2.0 client secret

.PARAMETER DashUrl
    Public URL of the deployed dashboard, e.g. https://dashboard-xyz-uc.a.run.app
    The redirect URI will be set to "$DashUrl/oauth2callback"

.PARAMETER SaDashboard
    Service account email used by the dashboard Cloud Run service,
    e.g. dashboard-sa@my-project.iam.gserviceaccount.com

.PARAMETER SecretName
    Name of the Secret Manager secret  (default: dashboard-auth-secrets)

.EXAMPLE
    .\scripts\setup_auth_secret.ps1 `
        -GoogleClientId     "123456.apps.googleusercontent.com" `
        -GoogleClientSecret "GOCSPX-..." `
        -DashUrl            "https://dashboard-abc123-uc.a.run.app" `
        -SaDashboard        "dashboard-sa@my-project.iam.gserviceaccount.com"
#>
[CmdletBinding()]
param (
    [Parameter(Mandatory)] [string] $GoogleClientId,
    [Parameter(Mandatory)] [string] $GoogleClientSecret,
    [Parameter(Mandatory)] [string] $DashUrl,
    [Parameter(Mandatory)] [string] $SaDashboard,
    [string] $SecretName = "dashboard-auth-secrets"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── 1. Generate a 32-byte random cookie secret ────────────────────────────────
$bytes = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
$cookieSecret = [Convert]::ToBase64String($bytes)

# ── 2. Build secrets.toml (in memory → temp file) ────────────────────────────
$tempFile = [System.IO.Path]::GetTempFileName()   # random path outside repo
try {
    @"
[auth]
redirect_uri              = "$($DashUrl.TrimEnd('/'))/oauth2callback"
cookie_secret             = "$cookieSecret"
client_id                 = "$GoogleClientId"
client_secret             = "$GoogleClientSecret"
server_metadata_url       = "https://accounts.google.com/.well-known/openid-configuration"
"@ | Set-Content $tempFile -Encoding ascii

    Write-Host "[1/3] secrets.toml written to temp file"

    # ── 3. Create (or update) the Secret Manager secret ──────────────────────
    $existingSecrets = gcloud secrets list --filter="name:$SecretName" --format="value(name)" 2>$null
    if ($existingSecrets) {
        Write-Host "[2/3] Secret '$SecretName' exists — adding a new version"
        gcloud secrets versions add $SecretName --data-file=$tempFile
    } else {
        Write-Host "[2/3] Creating secret '$SecretName'"
        gcloud secrets create $SecretName --data-file=$tempFile --replication-policy="automatic"
    }

} finally {
    # ── 4. Delete temp file regardless of success/failure ────────────────────
    Remove-Item $tempFile -Force -ErrorAction SilentlyContinue
    Write-Host "[3/3] Temp file deleted — credentials never persisted locally"
}

# ── 5. Grant dashboard SA read access ────────────────────────────────────────
Write-Host "Granting secretAccessor to $SaDashboard ..."
gcloud secrets add-iam-policy-binding $SecretName `
    --member "serviceAccount:$SaDashboard" `
    --role   "roles/secretmanager.secretAccessor"

Write-Host ""
Write-Host "Done. Remember to add '$($DashUrl.TrimEnd('/'))/oauth2callback'"
Write-Host "as an Authorised redirect URI in the Google Cloud OAuth 2.0 client."
