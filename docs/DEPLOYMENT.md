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

The Compose deployment requires an API key and uses the enhanced bearing
classifier by default. Set the key outside the repository before starting:

```bash
export MACHINA_API_KEY="replace-with-a-long-random-secret"
docker compose up --build -d
curl http://127.0.0.1:8000/ready \
  -H "X-Machina-API-Key: $MACHINA_API_KEY"
```

The service returns `ready` only when all configured model plugins are present.
Every response includes an `X-Request-ID`; retain it with the model version,
artifact hash, input source, and operator decision in an external audit store.
The runtime pins the scikit-learn major/minor line used to serialize the
published baseline artifacts; do not upgrade model-serving dependencies without
re-running the release audit and inference tests.

Set `MACHINA_API_KEY` in the deployment environment to protect every route
except `/health`. Send it as `X-Machina-API-Key`; never put it in the image,
source tree, or a committed Compose file.

The container is read-only except for the persistent `/data` volume, drops Linux
capabilities, and enables `no-new-privileges`. Treat this as a secure default,
not a substitute for an OT network threat model or an IEC 62443 assessment.

For production, put the API behind TLS and authentication, replace SQLite
with a managed database, restrict network access to telemetry producers, and
configure backups. The container must not be used as a direct safety-control
loop without site certification and an independent safety system.

## Lab deployment

The lab node already has Docker and enough root filesystem space for the
reference image. Keep the build context and Docker data root off the nearly
full home filesystem. Store secrets outside the image and use SSH keys rather
than passwords for operational access.
