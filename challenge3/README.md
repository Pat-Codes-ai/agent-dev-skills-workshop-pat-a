# Challenge 3: Multi-Agent Weather and Search Team

The Challenge 2 chatbot becomes a team of three agents. Same tools, same callbacks, same chat helper.

Notebook: `challenge3_multi_agent.ipynb` (Colab Enterprise). Package: `weather_team/` for `adk run` / `adk web`.

| Agent | Role | Wiring |
|---|---|---|
| `root_agent` | Receives every request and delegates | entry point; input validation callback runs here |
| `weather_agent` | Challenge 1/2 weather bot (`get_lat_lon`, `get_current_weather`) | `sub_agents=[weather_agent]`: root hands off with `transfer_to_agent` |
| `search_agent` | ADK built-in `google_search` tool | `tools=[AgentTool(agent=search_agent)]`: root calls it like a function |

Why two mechanisms: the built-in Google Search tool must be the only tool in its agent, so the
documented pattern (and the one on the workshop slides) is to wrap that agent as an `AgentTool`.
The weather agent has ordinary function tools, so it can be a plain sub-agent.

## Evidence of delegation

The chat helper prints each event with its author, so the notebook output shows lines like:

```
You: What's the weather like in Miami right now?
   [root_agent] hands off to transfer_to_agent({'agent_name': 'weather_agent'})
   [weather_agent] calls get_lat_lon({'place': 'Miami, FL'})
   [weather_agent] calls get_current_weather({...})
Bot (weather_agent): It's a warm, sunny day in Miami ...
```

Section 5 runs four requests (weather, news, both, small talk) and checks which agents were used.
Section 5b shows the Challenge 2 validation still blocking non-U.S. and malicious input at the root.

## Running it

Same as before: upload to Colab Enterprise, connect, run top to bottom. Section 6 asks for an
OpenAI key (Enter to skip); it moves only the root coordinator to GPT, since Google Search is
Gemini-only. Section 7 writes the `weather_team/` package for the terminal or the `adk web` dev UI,
whose Events panel draws the hand-offs between agents.
