"""Weather chatbot agent for `adk run weather_bot` / `adk web`.

Same tools and instructions as the Challenge 1 notebook, packaged as an ADK agent module.
ADK looks for a module-level `root_agent`.
"""
import os
from typing import Any, Dict
from urllib.parse import quote

import google.auth
import google.auth.transport.requests
import requests
from google.adk.agents import LlmAgent

try:
    _, _project = google.auth.default()
except Exception:  # no credentials (e.g. editing locally) - fall back to the env var
    _project = None
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or _project or "qwiklabs-gcp-03-9b118d33ce6b"
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", PROJECT_ID)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")      # Gemini 3.x is served from the global endpoint
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")

MODEL_GEMINI = "gemini-3.8-flash"


def get_lat_lon(place: str) -> Dict[str, Any]:
    """Convert a place name to latitude/longitude using the Google Maps Geocoding API.

    Args:
        place (str): A city, address or landmark, e.g. "Denver, CO".

    Returns:
        Dict[str, Any]: ``{"status": "success", "place": str, "lat": float, "lon": float}``
        or ``{"status": "error", "error_message": str}``.
    """
    # 1) Google Maps Geocoding API (v4). Uses an API key if you have one, otherwise the
    #    project's own credentials (student accounts cannot create API keys).
    try:
        if os.environ.get("GOOGLE_MAPS_API_KEY"):
            headers = {"X-Goog-Api-Key": os.environ["GOOGLE_MAPS_API_KEY"]}
        else:
            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            creds.refresh(google.auth.transport.requests.Request())
            headers = {"Authorization": f"Bearer {creds.token}", "X-Goog-User-Project": PROJECT_ID}
        resp = requests.get(f"https://geocode.googleapis.com/v4/geocode/address/{quote(place)}", headers=headers, timeout=15)
        results = resp.json().get("results", [])
        if results:
            best = results[0]
            return {"status": "success", "source": "Google Maps", "place": best["formattedAddress"],
                    "lat": best["location"]["latitude"], "lon": best["location"]["longitude"]}
    except Exception:
        pass

    # 2) Fallback: Open-Meteo's free geocoder (no key). Looks up the city name, U.S. only.
    try:
        city = place.split(",")[0].strip()
        resp = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                            params={"name": city, "count": 1, "countryCode": "US"}, timeout=15)
        results = resp.json().get("results", [])
        if not results:
            return {"status": "error", "error_message": f"Could not find a U.S. place called '{place}'."}
        best = results[0]
        return {"status": "success", "source": "Open-Meteo", "place": f"{best['name']}, {best.get('admin1', '')}, USA",
                "lat": best["latitude"], "lon": best["longitude"]}
    except Exception as exc:
        return {"status": "error", "error_message": f"Geocoding failed: {exc}"}


def get_current_weather(lat: float, lon: float) -> Dict[str, Any]:
    """Get the current weather conditions for a U.S. location from the National Weather Service API.

    Args:
        lat (float): Latitude in decimal degrees, e.g. 39.7392.
        lon (float): Longitude in decimal degrees, e.g. -104.9903.

    Returns:
        Dict[str, Any]: ``{"status": "success", "location": str, "period": str, "temperature": str,
        "wind": str, "conditions": str, "details": str}`` or ``{"status": "error", "error_message": str}``.
        Only United States locations are supported.
    """
    headers = {"User-Agent": "(adk-weather-workshop, student@example.com)"}  # NWS requires a User-Agent
    try:
        points = requests.get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}", headers=headers, timeout=15)
        if points.status_code != 200:
            return {"status": "error", "error_message": "The National Weather Service only covers U.S. locations."}
        props = points.json()["properties"]

        forecast = requests.get(props["forecast"], headers=headers, timeout=15).json()
        now = forecast["properties"]["periods"][0]
        where = props["relativeLocation"]["properties"]
        return {
            "status": "success",
            "location": f"{where['city']}, {where['state']}",
            "period": now["name"],
            "temperature": f"{now['temperature']}°{now['temperatureUnit']}",
            "wind": f"{now['windSpeed']} {now['windDirection']}",
            "conditions": now["shortForecast"],
            "details": now["detailedForecast"],
        }
    except Exception as exc:
        return {"status": "error", "error_message": f"Weather lookup failed: {exc}"}


INSTRUCTIONS = """You are a friendly weather chatbot for U.S. locations.

When someone asks about the weather somewhere:
1. Call get_lat_lon with the place name to get its coordinates.
2. Call get_current_weather with those coordinates.
3. Reply in a warm, conversational tone (2-4 sentences) with the temperature, conditions and wind.
   Add a practical tip (umbrella, sunscreen, jacket...) when it fits.

If a tool returns an error, explain it simply, e.g. that you only cover U.S. locations.
Only answer weather questions."""

root_agent = LlmAgent(
    name="weather_bot",
    model=MODEL_GEMINI,
    description="Friendly chatbot that reports current U.S. weather conditions.",
    instruction=INSTRUCTIONS,
    tools=[get_lat_lon, get_current_weather],
)
