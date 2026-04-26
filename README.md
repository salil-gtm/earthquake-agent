# Earthquake Intelligence Agent

A small CLI agent that answers natural-language questions about real seismic activity. It uses Google Gemini as the reasoning core and chains four tools across multiple LLM turns: live USGS earthquake data, OpenStreetMap geocoding, haversine distance, and multi-year USGS aggregates.

Sample interaction:

```
You: What was the strongest quake within 1000 km of Tokyo in the last month, and how far was it from the city?

  thought: Need coordinates for Tokyo.
  -> tool call: geocode_location({'place': 'Tokyo, Japan'})
  -> result: {"lat": 35.68, "lon": 139.76, "display_name": "..."}

  thought: Pull recent USGS data for that area.
  -> tool call: fetch_recent_earthquakes({'lat': 35.68, 'lon': 139.76, ...})
  -> result: {"count": 14, "quakes": [...]}

  thought: Compute distance from the strongest one to the city center.
  -> tool call: haversine_distance({...})
  -> result: {"km": 412.7}

  AGENT ANSWER: The strongest quake was M6.1 off the eastern coast of Honshu
  on 2025-09-14, about 413 km from central Tokyo.
```

## Why an agent and not a one-shot prompt

Each tool does something a language model genuinely cannot do alone:

| Tool | Why the LLM can't do it itself |
|---|---|
| `geocode_location` | LLMs hallucinate coordinates |
| `fetch_recent_earthquakes` | Real-time data; LLM training cutoff is months stale |
| `haversine_distance` | LLMs are unreliable at trigonometry |
| `summarize_seismic_history` | Requires aggregating thousands of records from a remote API |

The agent loop calls Gemini repeatedly, threading the full conversation (prior tool calls and their results) into every prompt, until the model returns an `answer` instead of another tool call.

## Install

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste a free Gemini key from https://aistudio.google.com/app/apikey
```

## Run

```bash
python earthquake_agent.py                  # interactive REPL
python earthquake_agent.py demo             # 3 canned queries, end-to-end
python earthquake_agent.py "your question"  # one-shot
```

Each run is also appended to `agent_log.txt` (delimited by a `===== run @ <timestamp> =====` line) so you can audit prior runs without re-running them.

## Architecture in 30 seconds

`run_agent(query)` does the following loop:

1. Build a prompt by serializing the entire `messages` list (system → user → assistant → tool → assistant → tool → …).
2. Send to Gemini.
3. Parse the response, which must be one of:
   - `{"thought": ..., "tool_name": ..., "tool_arguments": {...}}` — execute the tool, append result to `messages`, loop.
   - `{"thought": ..., "answer": ...}` — done.
4. Cap at `max_iterations` to avoid runaway loops.

The parser is forgiving: it strips markdown fences, pulls JSON out of prose, and accepts a few common aliases for `tool_arguments` (`arguments`, `args`, `parameters`).

## Configuration

All via environment variables (loaded from `.env`):

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Your Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | Any Gemini text model |
| `THROTTLE_SECONDS` | `6` | Sleep before each LLM call. Bump to 10 if you hit 429s. Free-tier Flash-Lite is 15 RPM. |

## Notes

- USGS FDSN and OpenStreetMap Nominatim are both free and require no key, but they ask for a polite User-Agent and modest request rates. The defaults stay well within their public terms.
- `fetch_recent_earthquakes` caps results at 25 per call to keep prompts compact.
- The agent is a single file (`earthquake_agent.py`) — easy to read end-to-end, easy to fork.

## License

MIT — see `LICENSE`.
