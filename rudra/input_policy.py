"""Compatibility guard for the Python renderer, which still uses SDR buffers."""


def require_legacy_input_policy(params: dict) -> None:
    """Reject headroom requests before decode, inference, staging or publication."""
    if "rudra_project" in params:
        raise ValueError("A RUDRA project cannot be read as legacy render parameters")
    for settings in (params, params.get("decode", {})):
        if not isinstance(settings, dict):
            raise ValueError("Invalid decode settings")
        if "headroom_policy" in settings and settings["headroom_policy"] != "disabled":
            raise ValueError("This Python rendering path does not support headroom_policy")
