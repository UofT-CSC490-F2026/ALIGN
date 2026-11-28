"""Exact configured-secret redaction, not a general secret scanner."""
import os

DEFAULT_SECRET_ENVS = ("ALIGN_AGENT_API_KEY", "ALIGN_ARBITER_API_KEY", "ALIGN_DATABASE_URL")


class Redactor:
    def __init__(self, values=()):
        self.values = tuple(sorted({v for v in values if v}, key=len, reverse=True))

    @classmethod
    def from_env(cls):
        extra = tuple(x.strip() for x in os.environ.get("ALIGN_TRACE_SECRET_ENV", "").split(",") if x.strip())
        return cls(os.environ.get(k, "") for k in (*DEFAULT_SECRET_ENVS, *extra))

    def clean(self, value):
        if isinstance(value, str):
            for secret in self.values:
                value = value.replace(secret, "[REDACTED]")
            return value
        if isinstance(value, list):
            return [self.clean(v) for v in value]
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                key = self.clean(k)
                if key in result:
                    raise ValueError("Redacted dictionary key collision")
                result[key] = self.clean(v)
            return result
        return value
