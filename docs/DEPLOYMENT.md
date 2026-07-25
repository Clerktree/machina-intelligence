# Deployment

## Docker

The reference deployment includes the API, MCP-compatible Python package,
fault model, and RUL model (including their scikit-learn runtime). SQLite state is persisted in the `machina-data`
volume.

```bash
docker compose up --build -d
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/capabilities
```

Set `MACHINA_API_KEY` in the deployment environment to protect every route
except `/health`. Send it as `X-Machina-API-Key`; never put it in the image,
source tree, or a committed Compose file.

For production, put the API behind TLS and authentication, replace SQLite
with a managed database, restrict network access to telemetry producers, and
configure backups. The container must not be used as a direct safety-control
loop without site certification and an independent safety system.

## Lab deployment

The lab node already has Docker and enough root filesystem space for the
reference image. Keep the build context and Docker data root off the nearly
full home filesystem. Store secrets outside the image and use SSH keys rather
than passwords for operational access.
