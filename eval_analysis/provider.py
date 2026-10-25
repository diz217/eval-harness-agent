"""Small Responses API adapter plus strict local replay of recorded model turns."""
from __future__ import annotations

import copy
import hashlib
import json
import os

import requests
import truststore

DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"


def request_hash(instructions, messages, tools):
    payload = json.dumps({"instructions": instructions, "input": messages, "tools": tools}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class ProviderError(RuntimeError):
    pass


class OpenAIProvider:
    kind = "openai_responses"

    def __init__(self, model=DEFAULT_MODEL):
        self.model = model
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise ProviderError("OPENAI_API_KEY is missing. Set it in the environment; never put it in source files.")
        truststore.inject_into_ssl()

    def respond(self, instructions, messages, tools, max_output_tokens, timeout):
        try:
            response = requests.post("https://api.openai.com/v1/responses",
                headers={"Authorization": "Bearer " + self.key},
                json={"model": self.model, "instructions": instructions, "input": messages,
                      "tools": tools, "parallel_tool_calls": False, "tool_choice": "required",
                      "max_output_tokens": max_output_tokens, "store": False}, timeout=timeout)
        except requests.RequestException:
            raise ProviderError("Model request failed at the transport layer; credentials and response bodies are not logged") from None
        try:
            result = response.json()
        except ValueError:
            raise ProviderError(f"Model returned a non-JSON response (HTTP {response.status_code})") from None
        if not response.ok:
            code = (result.get("error") or {}).get("code")
            # Never print provider error messages: some include credential fragments.
            safe_codes = {"invalid_api_key", "insufficient_quota", "rate_limit_exceeded", "model_not_found"}
            raise ProviderError(f"Model request failed: HTTP {response.status_code}, code={code if code in safe_codes else 'provider_error'}")
        return {"output": result.get("output", []), "usage": result.get("usage") or {},
                "model": result.get("model", self.model), "status": result.get("status"),
                "request_id": response.headers.get("x-request-id")}


class ReplayProvider:
    kind = "recorded_replay"

    def __init__(self, path):
        events = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()]
        self.turns = [e for e in events if e["event"] == "model_response"]
        if not self.turns:
            raise ValueError("Trace contains no recorded model responses")
        self.model = self.turns[0]["result"]["model"]
        self.index = 0

    def respond(self, instructions, messages, tools, max_output_tokens, timeout):
        if self.index >= len(self.turns):
            raise ProviderError("Replay exhausted before report completion")
        event = self.turns[self.index]
        if request_hash(instructions, messages, tools) != event["request_sha256"]:
            raise ProviderError("Replay input differs from recorded model request; replay refused")
        self.index += 1
        return copy.deepcopy(event["result"])
