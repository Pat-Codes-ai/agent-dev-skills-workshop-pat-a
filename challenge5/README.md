# Challenge 5 (Bonus): Deploying an Agent to Agent Platform

Deploys a Google Search agent with the Challenge 2 guardrail callbacks to **Agent Runtime** (Agent
Platform's managed runtime, formerly Agent Engine) and tests it, following the four steps on the
workshop slide and the Google Cloud quickstart in this folder.

Notebook: `challenge5_deploy_agent.ipynb` (Colab Enterprise, **fresh runtime**).

| Slide step | Notebook |
|---|---|
| 1. Initialize Agent Platform with project and location | `vertexai.Client(project, location)` + staging bucket |
| 2. Define your agent using the ADK | `Agent(model="gemini-3.8-flash", tools=[google_search], before/after_model_callback=...)` |
| 3. Test locally | `agent_engines.AdkApp(agent=...)` and `async_stream_query` (in-memory sessions) |
| 4. `agent_engines.create()` and use it | `client.agent_engines.create(agent=app, config={...})`, then remote `async_stream_query` |

## Decisions worth knowing

- **ADK version.** `google-cloud-aiplatform[agent_engines,adk]` pins google-adk 1.14.x. That is what
  runs in the container and what the install leaves on the Colab runtime, hence the fresh-runtime note.
  The agent and callbacks use only the 1.x API surface.
- **Global model endpoint.** The Agent Engine resource lives in `us-central1`; Gemini 3.x is served from
  `global`. The deploy config sets `env_vars={"GOOGLE_CLOUD_LOCATION": "global"}` (as in the instructor's
  snippet) so the container's model calls go to the right place.
- **`identity_type=AGENT_IDENTITY` is omitted.** It creates a per-agent principal with no permissions
  until IAM roles are granted, which a lab student account cannot do. The default identity, the
  Reasoning Engine service agent, already has what a deployed agent needs.
- **Guardrails ship with the agent.** The callbacks are defined in the notebook, so `cloudpickle`
  packages them with the `AdkApp`. The remote test sends a malicious prompt and gets the callback's
  canned refusal back from the deployment itself, proving the guardrail runs in Agent Runtime.
- **Clean-up is guarded.** `DELETE_AGENT = False` keeps the deployment visible to the instructor.

## Reconnecting later

```python
client = vertexai.Client(project=PROJECT_ID, location="us-central1")
remote = client.agent_engines.get(name="projects/.../locations/us-central1/reasoningEngines/<id>")
async for event in remote.async_stream_query(user_id="u", message="..."):
    ...
```
