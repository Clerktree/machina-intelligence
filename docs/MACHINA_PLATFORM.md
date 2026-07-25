# Machina platform definition

Machina is the product. Models are replaceable capability plugins inside
Machina.

The reference API can persist assets, telemetry, maintenance events, and model
registrations in SQLite by setting `MACHINA_DB_PATH=/path/to/machina.db`. The
default in-memory mode remains useful for tests and ephemeral experiments;
production deployments should use a managed database and backups.

## Intelligence layers

1. **Perception** — ingest vibration, acoustics, temperature, pressure,
   current, torque, PLC tags, images, logs, and maintenance events.
2. **Machine state** — infer operating mode, health, degradation, anomaly,
   fault, efficiency, and production context.
3. **Prediction** — estimate faults, remaining useful life, energy use,
   quality outcomes, and risk.
4. **Reasoning** — explain evidence, compare likely causes, retrieve manuals,
   and propose inspection or maintenance actions.
5. **Action** — create alerts, work-order drafts, reports, and feedback loops;
   never silently control safety-critical equipment.

## Model families

- Time-series models for telemetry and degradation
- Signal models for vibration and acoustic data
- Vision models for inspection imagery
- Tabular models for process and quality prediction
- Retrieval and language models for manuals, logs, and engineering dialogue

The CWRU bearing classifier is the first `fault_diagnosis` plugin. It should
not be marketed as the complete Machina intelligence system.
