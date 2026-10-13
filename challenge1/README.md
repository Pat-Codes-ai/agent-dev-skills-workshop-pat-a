# Challenge 1: Weather Chatbot with the Google ADK

A friendly chatbot. Ask *"Hey, what is the weather like in Denver?"* and it answers with the current
conditions.

Notebook: `challenge1_weather_alerts_agent.ipynb` (Colab Enterprise).

## How it works

| Challenge requirement | Where it is met |
|---|---|
| NWS function (lat/lon -> weather), PEP 8 type hints and docstring | `get_current_weather` in Section 2 |
| Google Maps Geocoding function (place -> lat/lon) | `get_lat_lon` in Section 2 |
| Functions added as ADK tools, agent given instructions | `make_weather_agent` in Section 3 |
| Gemini and a third-party model | Gemini 3.8 Flash in Section 3, GPT-4o via LiteLLM in Section 6 |
| Test code for multiple U.S. cities | Loop over five cities in Section 5 |

## Running it

1. Google Cloud console > Vertex AI > Colab Enterprise > Notebooks > Upload the `.ipynb`, region `us-central1`.
2. Connect to the default runtime and run the cells top to bottom.
3. Section 6 asks for an OpenAI key. Press Enter to skip if you do not have one.

No Google Maps API key is needed to get the bot online. `get_lat_lon` calls the Google Maps
Geocoding API (v4) with an API key if `GOOGLE_MAPS_API_KEY` is set, otherwise with the lab project's
own OAuth credentials. If Google Maps refuses the request (student accounts cannot create API keys),
it falls back to Open-Meteo's free geocoder so the chatbot still answers. The tool output shows
which source was used.
