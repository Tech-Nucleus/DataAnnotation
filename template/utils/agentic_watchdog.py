"""Agentic Watchdog and Cost-Control Subsystem for Bittensor Subnet 498.

Implements:
1. Deterministic interruptible sleep state with live CLI tuning.
2. Token & call circuit breaker tracking daily API consumption in .agentic_state/budget_ledger.json.
3. Lightweight Gemini integration (google-genai SDK with REST fallback) with strict token budgets:
   - Validator: max_output_tokens=120 (forces 1-2 line engineering diagnostics)
   - Miner: max_output_tokens=1024
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import bittensor as bt


class AgenticWatchdog:
    """Manages sleep cycles, daily API call/token budgets, and Gemini calls."""

    def __init__(
        self,
        role: str = "validator",
        sleep_seconds: Optional[float] = None,
        max_daily_calls: Optional[int] = None,
        state_dir: Optional[Path | str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        strict_live_llm: bool = False,
    ):
        self.role = role.lower()
        self.state_dir = Path(state_dir or "./.agentic_state").resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.state_dir / "budget_ledger.json"
        self.override_path = self.state_dir / "sleep_override.json"
        self.strict_live_llm = strict_live_llm

        # Defaults
        env_sleep = float(os.getenv("AGENTIC_SLEEP_SECONDS", "300"))
        self.default_sleep_seconds = float(sleep_seconds if sleep_seconds is not None else env_sleep)

        env_daily = int(os.getenv("AGENTIC_MAX_DAILY_CALLS", "48" if self.role == "miner" else "200"))
        self.max_daily_calls = int(max_daily_calls if max_daily_calls is not None else env_daily)

        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model or os.getenv("AGENTIC_LLM_MODEL", "gemini-2.5-flash")

        self._lock = threading.Lock()
        self._wake_event = threading.Event()
        self.current_state = "IDLE"
        self.call_logs: list[Dict[str, Any]] = []

    def get_sleep_seconds(self) -> float:
        """Fetch current sleep interval respecting dynamic CLI overrides."""
        try:
            if self.override_path.is_file():
                data = json.loads(self.override_path.read_text(encoding="utf-8"))
                val = data.get("sleep_seconds")
                if val is not None and float(val) > 0:
                    return float(val)
        except Exception:
            pass
        return self.default_sleep_seconds

    def set_sleep_seconds(self, seconds: float) -> None:
        """Update sleep duration dynamically for live operator tuning."""
        if seconds <= 0:
            raise ValueError("sleep_seconds must be positive")
        self.default_sleep_seconds = float(seconds)
        try:
            self.override_path.write_text(
                json.dumps({"sleep_seconds": float(seconds), "updated_at": datetime.now(timezone.utc).isoformat()}),
                encoding="utf-8",
            )
        except Exception as e:
            bt.logging.warning(f"Failed to persist sleep override: {e}")

    def wake(self) -> None:
        """Interrupt an active sleep cycle (manual step advance)."""
        self._wake_event.set()

    def sleep(self, seconds: Optional[float] = None) -> None:
        """Sleep deterministically while keeping the thread responsive to wake signals."""
        duration = seconds if seconds is not None else self.get_sleep_seconds()
        self.current_state = "SLEEP"
        self._wake_event.clear()
        bt.logging.info(f"[{self.role.upper()} WATCHDOG] Entering SLEEP for {duration:.1f}s")
        self._wake_event.wait(timeout=duration)
        self.current_state = "ACTIVE"
        self._wake_event.clear()

    # -------------------------------------------------------------------------
    # Budget Ledger & Circuit Breaker
    # -------------------------------------------------------------------------

    def _get_utc_today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _load_ledger(self) -> Dict[str, Any]:
        with self._lock:
            today = self._get_utc_today()
            if self.ledger_path.is_file():
                try:
                    data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
                    if data.get("date") == today:
                        return data
                except Exception:
                    pass
            # Initialize fresh ledger for the day
            fresh = {
                "date": today,
                "total_calls": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "calls_by_role": {
                    "validator": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
                    "miner": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
                },
                "validator_llm_calls": [],
                "miner_llm_calls": [],
                "circuit_breaker_tripped": False,
            }
            try:
                self.ledger_path.write_text(json.dumps(fresh, indent=2), encoding="utf-8")
            except Exception:
                pass
            return fresh

    def _save_ledger(self, data: Dict[str, Any]) -> None:
        with self._lock:
            try:
                self.ledger_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception as e:
                bt.logging.warning(f"Failed to write budget ledger: {e}")

    def check_budget(self) -> bool:
        """Check if role is within its daily API call allocation."""
        ledger = self._load_ledger()
        role_stats = ledger.get("calls_by_role", {}).get(self.role, {})
        calls = role_stats.get("calls", 0)
        if calls >= self.max_daily_calls:
            if not ledger.get("circuit_breaker_tripped"):
                ledger["circuit_breaker_tripped"] = True
                self._save_ledger(ledger)
                bt.logging.warning(
                    f"[{self.role.upper()} WATCHDOG] Daily budget limit reached ({calls}/{self.max_daily_calls}). "
                    "Circuit breaker TRIPPED! Falling back to heuristic mode."
                )
            return False
        return True

    def record_call(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record an API invocation in the ledger."""
        ledger = self._load_ledger()
        ledger["total_calls"] += 1
        ledger["total_prompt_tokens"] += prompt_tokens
        ledger["total_completion_tokens"] += completion_tokens
        role_stats = ledger.setdefault("calls_by_role", {}).setdefault(
            self.role, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        )
        role_stats["calls"] += 1
        role_stats["prompt_tokens"] += prompt_tokens
        role_stats["completion_tokens"] += completion_tokens
        self._save_ledger(ledger)

    def record_llm_proof(self, role: str, proof: Dict[str, Any]) -> None:
        """Log raw LLM proof metadata to budget ledger for audit verification."""
        ledger = self._load_ledger()
        key = "validator_llm_calls" if role.lower() == "validator" else "miner_llm_calls"
        calls_list = ledger.setdefault(key, [])
        calls_list.append(proof)
        self._save_ledger(ledger)

    def get_ledger_summary(self) -> Dict[str, Any]:
        """Return the current day's token and call metrics."""
        return self._load_ledger()

    # -------------------------------------------------------------------------
    # Gemini LLM Integration
    # -------------------------------------------------------------------------

    def call_gemini(
        self,
        system_prompt: str,
        user_content: str,
        image_bytes: Optional[bytes] = None,
        mime_type: str = "image/jpeg",
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.2,
        strict_live_llm: Optional[bool] = None,
        round_num: Optional[int] = None,
    ) -> Optional[str]:
        """Invoke Gemini 2.5 Flash within token budget and return raw text."""
        strict = self.strict_live_llm if strict_live_llm is None else strict_live_llm

        if not self.check_budget():
            if strict:
                raise RuntimeError(f"[{self.role.upper()}] Daily API call budget exceeded under strict_live_llm=True!")
            return None

        if not self.api_key:
            err = f"[{self.role.upper()}] No GEMINI_API_KEY available."
            bt.logging.error(err)
            if strict:
                raise RuntimeError(err)
            return None

        max_tokens = max_output_tokens or (300 if self.role == "validator" else 1024)
        # Compute image token estimate (768x1152 is 2 tiles of 258 tokens = 516 tokens)
        image_tokens = 516 if image_bytes else 0

        t0 = time.perf_counter()
        sdk_err: Optional[Exception] = None

        # 1. Try google-genai SDK
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            contents = []
            if image_bytes:
                contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
            contents.append(user_content)

            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
                temperature=temperature,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )

            resp = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

            latency_ms = int((time.perf_counter() - t0) * 1000)
            p_tokens = 0
            c_tokens = 0
            if hasattr(resp, "usage_metadata") and resp.usage_metadata is not None:
                p_tokens = getattr(resp.usage_metadata, "prompt_token_count", 0) or 0
                c_tokens = getattr(resp.usage_metadata, "candidates_token_count", 0) or 0
            if p_tokens == 0:
                p_tokens = len(user_content) // 4 + image_tokens
            if c_tokens == 0 and resp.text:
                c_tokens = len(resp.text) // 4

            self.record_call(p_tokens, c_tokens)

            # Record call in logs
            call_record = {
                "round_num": round_num or 0,
                "role": self.role,
                "model": self.model,
                "prompt_tokens": p_tokens,
                "image_tokens": image_tokens,
                "candidate_tokens": c_tokens,
                "raw_request_text": user_content,
                "raw_response_text": resp.text or "",
                "latency_ms": latency_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.call_logs.append(call_record)
            return resp.text
        except Exception as exc:
            sdk_err = exc
            bt.logging.warning(f"[{self.role.upper()}] google-genai SDK call failed: {exc}; attempting REST fallback.")

        # 2. REST Fallback
        rest_err: Optional[Exception] = None
        try:
            import urllib.request
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            parts = []
            if image_bytes:
                b64_data = base64.b64encode(image_bytes).decode("utf-8")
                parts.append({"inline_data": {"mime_type": mime_type, "data": b64_data}})
            parts.append({"text": user_content})

            payload = {
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                },
            }
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                latency_ms = int((time.perf_counter() - t0) * 1000)
                candidates = result.get("candidates", [])
                if candidates:
                    parts_out = candidates[0].get("content", {}).get("parts", [])
                    out_text = "".join(p.get("text", "") for p in parts_out)
                    p_tok = result.get("usageMetadata", {}).get("promptTokenCount", len(req_data) // 4 + image_tokens)
                    c_tok = result.get("usageMetadata", {}).get("candidatesTokenCount", len(out_text) // 4)
                    self.record_call(p_tok, c_tok)

                    call_record = {
                        "round_num": round_num or 0,
                        "role": self.role,
                        "model": self.model,
                        "prompt_tokens": p_tok,
                        "image_tokens": image_tokens,
                        "candidate_tokens": c_tok,
                        "raw_request_text": user_content,
                        "raw_response_text": out_text,
                        "latency_ms": latency_ms,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    self.call_logs.append(call_record)
                    return out_text
        except Exception as exc:
            rest_err = exc
            bt.logging.error(f"[{self.role.upper()}] REST fallback also failed: {exc}")

        if strict:
            raise RuntimeError(
                f"[{self.role.upper()}] Live Gemini call failed under strict_live_llm=True! "
                f"SDK Error: {sdk_err}; REST Error: {rest_err}"
            )

        return None
