$WIF_PROVIDER = 'projects/88374104629/locations/global/workloadIdentityPools/github-pool/providers/github-provider'
$WIF_PROVIDER = 'projects/88374104629/locations/global/workloadIdentityPools/github-pool/providers/github-provider'
$MLFLOW_URL = 'https://mlflow-qikperbo7q-ew.a.run.app'
$API_URL = 'https://api-qikperbo7q-ew.a.run.app'
. .\env.ps1
$DASH_URL = gcloud run services describe dashboard --region $REGION --format "value(status.url)"
"$DASH_URL/oauth2callback"