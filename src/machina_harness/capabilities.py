from typing import Literal

from pydantic import BaseModel


CapabilityStatus = Literal["available", "planned"]


class Capability(BaseModel):
    id: str
    name: str
    description: str
    status: CapabilityStatus
    required_inputs: list[str]


MACHINA_CAPABILITIES = [
    Capability(
        id="fault_diagnosis",
        name="Fault diagnosis",
        description="Classify likely machine faults from sensor windows.",
        status="available",
        required_inputs=["vibration or process sensors"],
    ),
    Capability(
        id="anomaly_detection",
        name="Anomaly detection",
        description="Detect unusual behavior and rank contributing signals.",
        status="available",
        required_inputs=["timestamped sensor window"],
    ),
    Capability(
        id="remaining_useful_life",
        name="Remaining useful life",
        description="Estimate time or cycles before a defined degradation threshold.",
        status="available",
        required_inputs=["historical degradation series", "operating context"],
    ),
    Capability(
        id="energy_intelligence",
        name="Energy intelligence",
        description="Find abnormal consumption, load inefficiency, and operating drift.",
        status="available",
        required_inputs=["power, load, and production context"],
    ),
    Capability(
        id="quality_prediction",
        name="Quality prediction",
        description="Connect machine conditions to process or product quality outcomes.",
        status="available",
        required_inputs=["process sensors", "quality labels"],
    ),
    Capability(
        id="maintenance_copilot",
        name="Maintenance copilot",
        description="Ground explanations and work instructions in manuals and maintenance history.",
        status="planned",
        required_inputs=["machine findings", "manuals", "maintenance records"],
    ),
    Capability(
        id="machine_knowledge",
        name="Machine knowledge",
        description="Maintain an asset graph of machines, components, sensors, events, and interventions.",
        status="planned",
        required_inputs=["asset registry", "sensor topology", "event history"],
    ),
]
