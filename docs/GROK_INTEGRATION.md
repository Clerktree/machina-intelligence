# Grok Build integration

Grok Build supports project-scoped MCP servers. The checked-in
`.grok/config.toml` exposes the following tools when Grok is launched from
this repository:

- `machina__analyze_machine_window`: analyze one timestamp-ordered sensor
  window and return status, anomaly score, contributing sensors, and a
  recommendation.
- `machina__machine_harness_health`: confirm the active model version.
- `machina__list_machine_intelligence_capabilities`: discover the broader
  Machina capability registry.
- `machina__register_machine_asset`: register an asset.
- `machina__record_machine_telemetry`: record telemetry tied to an asset.
- `machina__record_maintenance_event`: create maintenance history.
- `machina__machina_platform_snapshot`: inspect current platform state.
- `machina__list_machine_assets`: discover registered assets.
- `machina__list_registered_models`: discover registered model plugins.
- `machina__index_maintenance_document`: index a manual, SOP, or maintenance note.
- `machina__search_machine_knowledge`: retrieve relevant maintenance knowledge.
- `machina__analyze_machine_energy`: compare energy per output against history.
- `machina__predict_process_quality`: predict a manufacturing failure mode.
- `machina__prepare_maintenance_brief`: combine asset context, indexed evidence, and model plugins for grounded reasoning.

Install the optional adapter dependency first:

```bash
pip install -e '.[mcp]'
grok
```

The adapter uses the same inference function as the HTTP API. Later, the
implementation can call a deployed model endpoint instead of the baseline,
without changing Grok's tool contract.
