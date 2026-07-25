# Machina industry-readiness gate

This document defines what Machina must demonstrate before a customer may use
it in production advisory workflows. It is an engineering readiness plan, not
a certification or declaration of conformity.

## Current status

| Area | Current state | Release gate |
|---|---|---|
| Fault benchmark | CWRU controlled benchmark; grouped and leave-one-RPM-out evaluation | Held-out customer machines and operating regimes |
| Inference safety | Input validation, confidence threshold, abstention, human-review flag | Site-specific thresholds validated against missed-fault and false-alarm costs |
| Traceability | Model version, SHA-256, request ID, and audit-safe request log | Durable tamper-evident audit retention owned by deployment operator |
| Deployment security | API key, read-only container, dropped capabilities, readiness endpoint | OT threat model, segmentation, certificate management, penetration test |
| Condition monitoring workflow | API and platform contracts | Map site workflow to ISO 17359 and diagnostic data flow to ISO 13374 |
| Safety | Decision support only; no automatic control loop | Separate safety function, risk assessment, and validation under applicable machinery standards |
| MLOps | Versioned artifacts and release audit | Drift monitoring, rollback, approval, canary release, incident response |

## Non-negotiable production evidence

Before a production advisory deployment, collect a signed evaluation bundle with:

- machine-level train/validation/test separation;
- per-machine precision, recall, false alarms per operating hour, missed-fault rate,
  lead time, calibration, and abstention rate;
- sensor placement, sampling rate, operating mode, load, and maintenance history;
- known-fault and unknown-fault tests;
- latency, CPU/RAM, disk, power, offline behavior, and recovery tests at the edge;
- model, code, configuration, and dataset hashes;
- human-oversight procedure and an explicit prohibition on safety control;
- security review covering identity, network segmentation, secrets, updates, and logs.

## Standards mapping

ISO 17359:2018 provides general guidelines for establishing a machine condition-
monitoring programme. ISO 13374-1/-2 cover condition-monitoring data processing,
communication, presentation, and reference processing models. ISA/IEC 62443
provides the industrial automation cybersecurity lifecycle and component/system
security framework. ISO 13849-1:2023 is relevant if any Machina output becomes
part of a safety-related control function; Machina must remain outside that
function unless separately engineered and assessed.

Machina must not be marketed as certified against these standards merely because
its software has an API or a passing benchmark. Certification and conformity
assessment require the applicable product scope, evidence, and competent
external review.
