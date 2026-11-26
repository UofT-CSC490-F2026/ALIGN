"""Person 1 compatible synchronous interface for a warm Modal arbiter.

Use inside ``with arbiter_training.modal_app.app.run():`` for an ephemeral app.
No endpoint or deployment is created by importing this module.
"""

from .contract import messages_for_observation, parse_decision
from .modal_app import ArbiterModel, _decision_text


class ModalArbiter:
    def __init__(self, model: str = "qwen", adapter_path: str = "",
                 invalid_policy: str = "raise", remote_model=None):
        if model not in {"qwen", "gpt-oss"}:
            raise ValueError("model must be qwen or gpt-oss")
        if adapter_path and model != "qwen":
            raise ValueError("The current adapter is available only for Qwen")
        if invalid_policy not in {"query", "raise"}:
            raise ValueError("invalid_policy must be query or raise")
        self.model = model
        self.invalid_policy = invalid_policy
        self.remote_model = remote_model or ArbiterModel(
            model_name=model, adapter_path=adapter_path)

    def decide(self, observation: dict) -> dict:
        return self.decide_many([observation])[0]

    def decide_many(self, observations: list[dict]) -> list[dict]:
        if not observations:
            return []
        prompts = [messages_for_observation(item) for item in observations]
        outputs = self.remote_model.decide_batch.remote(prompts)
        if len(outputs) != len(observations):
            raise RuntimeError("Modal returned a different number of decisions")
        decisions = []
        for observation, result in zip(observations, outputs):
            try:
                decisions.append(parse_decision(_decision_text(
                    self.model, result["text"]), observation=observation,
                    enforceable=True))
            except ValueError:
                if self.invalid_policy == "raise":
                    raise
                decisions.append({
                    "action": "query",
                    "reason": "The arbiter response was invalid; inspect the proposal before acting.",
                    "affected_agents": [observation["actor_id"]],
                    "evidence": {"proposal_sequence": observation["proposal_sequence"],
                                 "error": "invalid_model_output"},
                })
        return decisions
