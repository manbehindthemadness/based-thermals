"""
Represent restored camera display settings as editable hardware pipeline nodes.
"""

from __future__ import annotations

from typing import Any

from ..pipeline import node


def restored_hardware_nodes(hardware: Any, profile_id: str) -> list[dict[str, Any]]:
    """
    Build nodes from the restored baseline without issuing additional camera commands.
    """
    state = hardware.state()
    if profile_id != "duo":
        return [
            node("hardware", "device_control", control=name, value=state[name]["value"])
            for name, spec in hardware.specs.items()
            if spec.supported and spec.scope == "device" and spec.effect == "preview"
            and state.get(name, {}).get("available", False)
        ]
    if not hardware.original:
        return []
    result = [
        node("hardware", "preset", value=hardware.processing_preset),
        node("hardware", "boost", value=hardware.boost),
    ]
    groups = (
        ("camera_colors", {"palette": "palette"}),
        ("detail", {"enabled": "detail_enabled", "amount": "detail"}),
        ("noise", {name: name for name in (
            "noise_mode", "noise_general", "noise_spatial", "noise_temporal",
        )}),
        ("brightness", {"value": "brightness"}),
        ("contrast", {"value": "contrast"}),
    )
    for kind, fields in groups:
        if not all(state.get(name, {}).get("available", False) for name in fields.values()):
            continue
        params = {key: state[name]["value"] for key, name in fields.items()}
        if kind == "detail":
            params["enabled"] = bool(params["enabled"])
            params["fixed"] = hardware.fixed_range
        result.append(node("hardware", kind, **params))
    result.append(node("hardware", "gamma", value=hardware.gamma))
    return result
