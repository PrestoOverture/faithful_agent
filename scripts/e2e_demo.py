#!/usr/bin/env python3
import json
import os
import sys


def load_dotenv(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip("\"")
            if key and key not in os.environ:
                os.environ[key] = value


if not os.environ.get("OPENAI_API_KEY"):
    load_dotenv(".env")

if not os.environ.get("OPENAI_API_KEY"):
    print(
        "OPENAI_API_KEY is required. Set it in .env or run: OPENAI_API_KEY=sk-... python scripts/e2e_demo.py",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print(
        "openai package not installed. Install it with: uv add openai",
        file=sys.stderr,
    )
    sys.exit(1)

from faithfulagent import verify
from faithfulagent.reporter.formatters import to_json, to_summary

client = OpenAI()

system_prompt = "Summarize the flight information naturally."
user_prompt = (
    "Find flight information for Shanghai to Beijing on 2025-08-15. "
    "Use the tool to get the data, then summarize it."
)

tool_response = {
    "airline": "China Eastern",
    "price": 553,
    "currency": "CNY",
    "duration_minutes": 270,
    "stops": 1,
    "departure": "2025-08-15T08:30:00",
    "arrival": "2025-08-15T13:00:00",
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_flight_info",
            "description": "Get flight details for a route and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    }
]

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

first_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
)

assistant_message = first_response.choices[0].message
tool_calls = assistant_message.tool_calls

if not tool_calls:
    print("Model did not call the tool. Exiting.", file=sys.stderr)
    sys.exit(1)

raw_tool_calls = []
openai_tool_calls = []

for tc in tool_calls:
    tc_dict = {
        "id": tc.id,
        "type": tc.type,
        "function": {
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        },
    }
    raw_tool_calls.append(tc_dict)
    openai_tool_calls.append({**tc_dict, "response": tool_response})

messages.append(
    {
        "role": "assistant",
        "content": assistant_message.content,
        "tool_calls": raw_tool_calls,
    }
)

for tc in tool_calls:
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(tool_response),
        }
    )

final_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)

final_text = final_response.choices[0].message.content or ""

print("Agent output:")
print(final_text)

heuristic_report = verify(
    agent_output=final_text,
    tool_calls=openai_tool_calls,
    mode="heuristic",
    source_format="openai",
)

print("\nHeuristic report (JSON):")
print(to_json(heuristic_report))

refs = []
field_paths = {
    "airline": "$.airline",
    "price": "$.price",
    "currency": "$.currency",
    "duration_minutes": "$.duration_minutes",
    "stops": "$.stops",
    "departure": "$.departure",
    "arrival": "$.arrival",
}

call_id = openai_tool_calls[0]["id"]

for field, path in field_paths.items():
    value_str = str(tool_response[field])
    start = final_text.find(value_str)
    if start == -1:
        print(
            f"Strict mode: value for '{field}' not found in output; skipping ref.",
            file=sys.stderr,
        )
        continue
    end = start + len(value_str)
    refs.append(
        {
            "span": [start, end],
            "call_id": call_id,
            "path": path,
            "transform": "none",
        }
    )

strict_input = {"text": final_text, "refs": refs}

strict_report = verify(
    agent_output=strict_input,
    tool_calls=openai_tool_calls,
    mode="strict",
    source_format="openai",
)

print("\nStrict report (JSON):")
print(to_json(strict_report))

print("\nSummaries:")
print("Heuristic:", to_summary(heuristic_report))
print("Strict:", to_summary(strict_report))
