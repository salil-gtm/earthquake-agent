"""
Earthquake Intelligence Agent (CLI)
===================================

An agent that answers natural-language questions about real seismic
activity by chaining four tools across multiple Gemini calls and
threading the full conversation through every turn.

Tools:
  - geocode_location          (OpenStreetMap Nominatim, no API key)
  - fetch_recent_earthquakes  (USGS FDSN, no API key)
  - haversine_distance        (pure math, no network)
  - summarize_seismic_history (USGS aggregated stats over years)

Each call to the LLM rebuilds the prompt from the entire `messages`
history, so the model always sees prior tool calls and their results.
The loop continues until the model returns an `answer` instead of
another tool call.

Setup:
  pip install -r requirements.txt
  Create a .env file next to this script:
    GEMINI_API_KEY=your-key-here
    GEMINI_MODEL=gemini-2.5-flash-lite

Run:
  python earthquake_agent.py                  # interactive REPL
  python earthquake_agent.py demo             # canned queries
  python earthquake_agent.py "your question"  # one-shot

When run from the command line, every line of output is also appended
to `agent_log.txt` next to this file, delimited by a `===== run @ ...`
header. Importing this module as a library does NOT touch stdout.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import datetime as dt

import requests
from dotenv import load_dotenv

# ============================================================
# Configuration
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
THROTTLE_SECONDS = float(os.getenv("THROTTLE_SECONDS", "6"))  # free-tier RPM friendly
LOG_PATH = os.path.join(os.path.dirname(__file__), "agent_log.txt")

# ============================================================
# Logging — when run as a script, tee stdout into agent_log.txt.
# This is OFF by default when this file is imported as a module so
# library users never get their stdout silently redirected.
# ============================================================

class _Tee:
    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "a", encoding="utf-8")
        self._f.write(f"\n\n===== run @ {dt.datetime.now().isoformat()} =====\n")
        self._f.flush()

    def write(self, msg: str) -> None:
        sys.__stdout__.write(msg)
        self._f.write(msg)
        self._f.flush()

    def flush(self) -> None:
        sys.__stdout__.flush()
        self._f.flush()


def _enable_run_log() -> None:
    """Tee all subsequent stdout writes to agent_log.txt."""
    sys.stdout = _Tee(LOG_PATH)

# ============================================================
# LLM client — Gemini via REST (no SDK dependency)
# ============================================================

GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)


RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def call_llm(prompt: str, max_retries: int = 4) -> str:
    """Send a prompt to Gemini's REST endpoint and return the text response.

    Sleeps THROTTLE_SECONDS before the first attempt to stay under the
    free-tier rate limit. Retries transient HTTP errors (429, 5xx) and
    network errors with exponential backoff. Other 4xx responses (bad
    key, malformed request, safety block) raise immediately so the
    caller sees them on attempt 1 instead of after four wasted retries.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Create a .env in this folder with:\n"
            "  GEMINI_API_KEY=your-key-here\n"
            "  GEMINI_MODEL=gemini-2.5-flash-lite"
        )

    print(f"  [waiting {THROTTLE_SECONDS}s to respect rate limits...]", flush=True)
    time.sleep(THROTTLE_SECONDS)

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(
                GEMINI_ENDPOINT,
                params={"key": GEMINI_API_KEY},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
        except requests.RequestException as e:
            last_exc = e
            if attempt == max_retries:
                break
            backoff = 2 ** attempt
            print(
                f"  [network error: {e}; retrying in {backoff}s "
                f"({attempt}/{max_retries})]",
                flush=True,
            )
            time.sleep(backoff)
            continue

        if r.status_code in RETRYABLE_STATUS and attempt < max_retries:
            backoff = 2 ** attempt
            print(
                f"  [Gemini returned HTTP {r.status_code}; retrying in "
                f"{backoff}s ({attempt}/{max_retries})]",
                flush=True,
            )
            time.sleep(backoff)
            continue

        r.raise_for_status()
        data = r.json()
        # Standard shape: candidates[0].content.parts[0].text
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            # Bubble up the raw payload so the caller can see what came back
            # (rate-limit JSON, safety-blocked response, etc.).
            return json.dumps(data)

    # Exhausted retries
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Gemini call failed after {max_retries} retries.")


# ============================================================
# System prompt — turns a chat model into an agent
# ============================================================

SYSTEM_PROMPT = """You are an Earthquake Intelligence Agent. You answer questions about real seismic activity by reasoning step-by-step and calling tools.

You have access to FOUR tools. Use them aggressively — never guess coordinates, never estimate distances, never invent recent earthquakes.

1. geocode_location(place: str) -> {"lat": float, "lon": float, "display_name": str}
   Convert a place name to coordinates. Always call this first when the user mentions a city/region.
   Example: geocode_location("Bangalore, India")

2. fetch_recent_earthquakes(lat: float, lon: float, radius_km: float, days: int = 7, min_magnitude: float = 2.5) -> list of quakes
   Live data from USGS. Returns recent earthquakes within radius_km of (lat, lon) over the last `days` days at or above `min_magnitude`.
   Each quake has: time (ISO UTC), magnitude, place, lat, lon, depth_km, url.
   Example: fetch_recent_earthquakes(lat=12.97, lon=77.59, radius_km=500, days=7, min_magnitude=2.5)

3. haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> {"km": float}
   Great-circle distance in kilometers between two points. Use to tell the user how far away a quake was.

4. summarize_seismic_history(lat: float, lon: float, radius_km: float, years: int = 5, min_magnitude: float = 4.0) -> stats
   Aggregated stats from USGS for the past `years` years within radius_km. Returns count, average magnitude, max magnitude, and yearly breakdown. Use this for "compared to normal" questions.

You must respond in ONE of these two JSON formats and NOTHING else:

To call a tool:
{"thought": "<one short sentence about what you're doing and why>", "tool_name": "<name>", "tool_arguments": {"<arg>": <value>, ...}}

To finalize:
{"thought": "<one short sentence summarizing>", "answer": "<your final answer in plain English, citing magnitudes/distances/dates>"}

CONCRETE EXAMPLES:

User: Any earthquakes near Bangalore in the last week?
Response: {"thought": "Need coordinates for Bangalore first.", "tool_name": "geocode_location", "tool_arguments": {"place": "Bangalore, India"}}

Tool Result: {"lat": 12.97, "lon": 77.59, "display_name": "Bengaluru, Karnataka, India"}
Response: {"thought": "Now query USGS for recent quakes in a 500km radius.", "tool_name": "fetch_recent_earthquakes", "tool_arguments": {"lat": 12.97, "lon": 77.59, "radius_km": 500, "days": 7, "min_magnitude": 2.5}}

Tool Result: {"count": 0, "quakes": []}
Response: {"thought": "No quakes — answer the user.", "answer": "No earthquakes of magnitude 2.5+ were recorded within 500 km of Bangalore in the last 7 days, according to USGS."}

RULES:
- Respond with ONLY the JSON. No prose, no markdown fences.
- Always geocode_location before fetch_recent_earthquakes when a place name is given.
- Always include a brief "thought" so your reasoning is visible.
- Cite magnitudes and approximate distances/times in the final answer.
- If a tool returns an error, try a different approach (different radius, different magnitude floor, etc.) before giving up.
"""


# ============================================================
# Tools
# ============================================================

USGS_FDSN = "https://earthquake.usgs.gov/fdsnws/event/1/query"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
# Public APIs (USGS, OSM Nominatim) want a polite, identifying User-Agent.
# Override via the AGENT_USER_AGENT env var when forking.
USER_AGENT = os.getenv(
    "AGENT_USER_AGENT",
    "earthquake-agent/1.0 (https://github.com/your-username/earthquake-agent)",
)


def geocode_location(place: str) -> str:
    """Resolve a place name to (lat, lon) via OpenStreetMap Nominatim."""
    try:
        r = requests.get(
            NOMINATIM,
            params={"q": place, "format": "json", "limit": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return json.dumps({"error": f"No location found for '{place}'."})
        hit = data[0]
        return json.dumps({
            "lat": float(hit["lat"]),
            "lon": float(hit["lon"]),
            "display_name": hit.get("display_name", place),
        })
    except requests.RequestException as e:
        return json.dumps({"error": f"Geocoding failed: {e}"})


def fetch_recent_earthquakes(
    lat: float,
    lon: float,
    radius_km: float,
    days: int = 7,
    min_magnitude: float = 2.5,
) -> str:
    """Fetch recent earthquakes from USGS within a radius."""
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=int(days))
    try:
        r = requests.get(
            USGS_FDSN,
            params={
                "format": "geojson",
                "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
                "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
                "latitude": lat,
                "longitude": lon,
                "maxradiuskm": radius_km,
                "minmagnitude": min_magnitude,
                "orderby": "time",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        r.raise_for_status()
        feats = r.json().get("features", [])
        quakes = []
        for f in feats[:25]:  # cap to keep prompt small
            p = f.get("properties", {}) or {}
            g = (f.get("geometry") or {}).get("coordinates") or [None, None, None]
            t_ms = p.get("time")
            iso = (
                dt.datetime.fromtimestamp(t_ms / 1000, dt.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
                if t_ms else None
            )
            quakes.append({
                "time": iso,
                "magnitude": p.get("mag"),
                "place": p.get("place"),
                "lat": g[1],
                "lon": g[0],
                "depth_km": g[2],
                "url": p.get("url"),
            })
        return json.dumps({
            "count": len(quakes),
            "window_days": int(days),
            "min_magnitude": min_magnitude,
            "radius_km": radius_km,
            "quakes": quakes,
        })
    except requests.RequestException as e:
        return json.dumps({"error": f"USGS request failed: {e}"})


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    """Great-circle distance between two points in km."""
    try:
        r_earth = 6371.0088
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return json.dumps({"km": round(r_earth * c, 2)})
    except Exception as e:
        return json.dumps({"error": f"Distance calc failed: {e}"})


def summarize_seismic_history(
    lat: float,
    lon: float,
    radius_km: float,
    years: int = 5,
    min_magnitude: float = 4.0,
) -> str:
    """Aggregate USGS stats over the past `years` years for the area."""
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=int(years) * 365)
    try:
        r = requests.get(
            USGS_FDSN,
            params={
                "format": "geojson",
                "starttime": start.strftime("%Y-%m-%d"),
                "endtime": end.strftime("%Y-%m-%d"),
                "latitude": lat,
                "longitude": lon,
                "maxradiuskm": radius_km,
                "minmagnitude": min_magnitude,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        r.raise_for_status()
        feats = r.json().get("features", [])
        per_year: dict[int, list[float]] = {}
        max_mag = 0.0
        max_event = None
        for f in feats:
            p = f.get("properties", {}) or {}
            t_ms = p.get("time")
            mag = p.get("mag")
            if t_ms is None or mag is None:
                continue
            event_dt = dt.datetime.fromtimestamp(t_ms / 1000, dt.timezone.utc)
            per_year.setdefault(event_dt.year, []).append(mag)
            if mag > max_mag:
                max_mag = mag
                max_event = {
                    "magnitude": mag,
                    "place": p.get("place"),
                    "time": event_dt.isoformat().replace("+00:00", "Z"),
                    "url": p.get("url"),
                }
        total = sum(len(v) for v in per_year.values())
        avg = round(sum(m for v in per_year.values() for m in v) / total, 2) if total else None
        yearly = {
            str(y): {"count": len(v), "avg_magnitude": round(sum(v) / len(v), 2)}
            for y, v in sorted(per_year.items())
        }
        return json.dumps({
            "years_window": int(years),
            "radius_km": radius_km,
            "min_magnitude": min_magnitude,
            "total_count": total,
            "avg_per_year": round(total / max(int(years), 1), 2),
            "average_magnitude": avg,
            "max_event": max_event,
            "yearly_breakdown": yearly,
        })
    except requests.RequestException as e:
        return json.dumps({"error": f"USGS history request failed: {e}"})


tools = {
    "geocode_location": geocode_location,
    "fetch_recent_earthquakes": fetch_recent_earthquakes,
    "haversine_distance": haversine_distance,
    "summarize_seismic_history": summarize_seismic_history,
}


# ============================================================
# Response parser — defensive against fences, prose wrappers, and key-name drift
# ============================================================

def parse_llm_response(text: str) -> dict:
    """Pull a JSON object out of whatever the LLM emitted."""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse LLM response: {text[:200]}")


def coerce_args(parsed: dict) -> dict:
    """Normalize 'tool_arguments' to a dict regardless of small LLM mistakes."""
    raw = parsed.get("tool_arguments")
    if raw is None:
        for k in ("tool_args", "arguments", "args", "parameters", "params"):
            if k in parsed:
                raw = parsed[k]
                break
    if isinstance(raw, dict):
        return raw
    return {}


# ============================================================
# Agent loop
# ============================================================

def run_agent(user_query: str, max_iterations: int = 6, verbose: bool = True):
    """User query -> LLM -> [tool -> result -> LLM]* -> final answer.

    The full conversation history is rebuilt into the prompt every
    iteration. That's how the LLM "remembers" prior tool results.
    """
    if verbose:
        print(f"\n{'=' * 64}")
        print(f"  USER: {user_query}")
        print(f"{'=' * 64}")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\n--- Iteration {iteration} ---")

        # Rebuild the prompt from the full message history every iteration.
        # Each LLM call therefore sees every prior tool call and result —
        # that's how the model "remembers" what's already been tried.
        prompt_parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(content)
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "tool":
                prompt_parts.append(f"Tool Result: {content}")
        prompt = "\n\n".join(prompt_parts)

        # ---- LLM call ----
        response_text = call_llm(prompt)
        if verbose:
            print(f"LLM raw: {response_text.strip()}")

        # ---- Parse ----
        try:
            parsed = parse_llm_response(response_text)
        except ValueError as e:
            if verbose:
                print(f"  Parse error: {e}. Asking LLM to retry with valid JSON.")
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": "Your last response was not valid JSON. Reply with ONLY a JSON object as specified."
            })
            continue

        # Surface the agent's own thought (reasoning chain)
        thought = parsed.get("thought")
        if thought and verbose:
            print(f"  thought: {thought}")

        # ---- Final answer? ----
        if "answer" in parsed:
            if verbose:
                print(f"\n{'=' * 64}")
                print(f"  AGENT ANSWER: {parsed['answer']}")
                print(f"{'=' * 64}\n")
            return parsed["answer"]

        # ---- Tool call ----
        tool_name = parsed.get("tool_name")
        if not tool_name:
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": "Response missing both 'answer' and 'tool_name'. Pick one."
            })
            continue

        args = coerce_args(parsed)
        if verbose:
            print(f"  -> tool call: {tool_name}({args})")

        if tool_name not in tools:
            err = json.dumps({
                "error": f"Unknown tool: {tool_name}. Available: {list(tools)}"
            })
            if verbose:
                print(f"  -> error: {err}")
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "tool", "content": err})
            continue

        try:
            result = tools[tool_name](**args)
        except TypeError as e:
            err = json.dumps({"error": f"Bad arguments for {tool_name}: {e}"})
            if verbose:
                print(f"  -> error: {err}")
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "tool", "content": err})
            continue

        if verbose:
            preview = result if len(result) <= 600 else result[:600] + "...(truncated)"
            print(f"  -> result: {preview}")

        messages.append({"role": "assistant", "content": response_text})
        messages.append({"role": "tool", "content": result})

    print("\nMax iterations reached without a final answer.")
    return None


# ============================================================
# Entry points
# ============================================================

DEMO_QUERIES = [
    "Any earthquakes near Bangalore in the last 7 days?",
    "What was the strongest earthquake within 1000 km of Tokyo in the last 30 days, and how far was it from the city center?",
    "Compare seismic activity around San Francisco over the last 5 years to the last 30 days. Anything unusual?",
]


def repl():
    print("=" * 64)
    print("  Earthquake Intelligence Agent — interactive mode")
    print(f"  Model: {GEMINI_MODEL}")
    print("  Tools: geocode_location, fetch_recent_earthquakes,")
    print("         haversine_distance, summarize_seismic_history")
    print("  Type 'exit' to quit.")
    print("=" * 64)
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            continue
        if q.lower() in {"exit", "quit", "bye"}:
            print("Bye.")
            return
        run_agent(q)


def demo():
    print("=" * 64)
    print("  Earthquake Intelligence Agent — demo mode")
    print(f"  Model: {GEMINI_MODEL}")
    print("=" * 64)
    for q in DEMO_QUERIES:
        run_agent(q)


if __name__ == "__main__":
    _enable_run_log()
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            demo()
        else:
            run_agent(" ".join(sys.argv[1:]))
    else:
        repl()
