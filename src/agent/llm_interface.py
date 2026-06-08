def build_llm_agent_payload(
    active_model,
    pk_model_name,
    discovery_top_results,
    mechanism_hint=None,
    mechanism_confirmation=None,
):
    """
    Lightweight handoff payload for future LLM orchestration layers.
    This function is intentionally schema-stable and side-effect free.
    """
    return {
        "task_type": "pkpd_discovery_handoff",
        "active_model": active_model,
        "pk_model_name": pk_model_name,
        "top_results": discovery_top_results,
        "mechanism_hint": mechanism_hint,
        "mechanism_confirmation": mechanism_confirmation,
    }
