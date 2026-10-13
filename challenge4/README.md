# Challenge 4: An Answer Workflow that Verifies and Refines

Built on the Challenge 3 notebook. Every real question is answered in three fixed steps by an ADK
`SequentialAgent`, with a Greeter agent in front.

Notebook: `challenge4_answer_workflow.ipynb` (Colab Enterprise). Package: `weather_answer_team/` for `adk run` / `adk web`.

| Agent | Step | Does | Writes state key |
|---|---|---|---|
| `greeter` | root | Greets; hands every real question to `answer_team` with `transfer_to_agent` | - |
| `researcher` | 1. Search | Weather tools + Google Search agent (as `AgentTool`); writes a first draft | `draft` |
| `critic` | 2. Critique | Reads `{draft}`; lists 3-5 specific improvements, no rewriting | `critique` |
| `refiner` | 3. Refine | Reads `{draft}` and `{critique}`; returns the final answer | `final_answer` |

How the steps talk to each other: `output_key` saves each step's reply into the shared session
state, and `{placeholder}` syntax in the next step's instruction reads it back. The `SequentialAgent`
fixes the order, so no agent decides who runs next.

## Evidence in the notebook

The chat helper prints every event with its author plus a preview of each step's text, so one
question shows: the Greeter's hand-off, the researcher's tool calls, the draft, the critique, and the
refined final answer. A follow-up cell prints `draft`, `critique` and `final_answer` straight from
session state. Section 5 checks three requests: a weather-and-events question runs all three steps
in order using both tool kinds, a weather-only question still gets critiqued and refined, and small
talk is answered by the Greeter alone. Section 5b shows the Challenge 2 validation still blocking
non-U.S. and malicious input at the Greeter.

## Running it

Same as before: upload to Colab Enterprise, connect, run top to bottom (Enter to skip the OpenAI
prompt). A full three-step answer takes 30-60 seconds because it makes four or five model calls.
Section 7 writes the `weather_answer_team/` package; in the `adk web` dev UI the Events panel shows
the steps firing in order and the State tab shows the three keys.

Alternative noted on the slides: the same Critique and Refine agents could sit inside a `LoopAgent`
with `max_iterations` and an "approve" tool that sets `escalate` to stop early. The Sequential
version was chosen because the requirement is a single verify-and-refine pass.
