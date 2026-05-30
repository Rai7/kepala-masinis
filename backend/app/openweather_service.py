import httpx
from time import perf_counter
from typing import Any, Dict, Optional, Tuple
from .config import settings
from .mongodb import get_collection
from .llm_client import run_llm_extraction
from .tavily_service import search_web_context

# In-memory cache for fast lookup
_coords_cache: Dict[str, Tuple[float, float]] = {}

def get_openweather_client() -> httpx.Client | None:
    if not settings.enable_open_weather or not settings.open_weather_api_key:
        return None
    return httpx.Client(
        base_url="https://api.openweathermap.org/data/3.0",
        timeout=httpx.Timeout(5.0, connect=2.0),
    )

def resolve_station_coordinates(station_name: str) -> Tuple[Optional[float], Optional[float]]:
    # Normalize station name for cache key
    key = station_name.strip().lower()
    
    # 1. Check in-memory cache
    if key in _coords_cache:
        return _coords_cache[key]
        
    # 2. Check MongoDB cache if enabled
    if settings.enable_mongo_cache:
        coll = get_collection("station_coordinates_cache")
        if coll is not None:
            try:
                doc = coll.find_one({"station_name": key})
                if doc and "lat" in doc and "lon" in doc:
                    _coords_cache[key] = (doc["lat"], doc["lon"])
                    return doc["lat"], doc["lon"]
            except Exception:
                pass

    # 3. Resolve using Tavily if enabled
    if settings.enable_tavily_search and settings.tavily_api_key:
        query = f"Latitude and Longitude of {station_name} train station Indonesia"
        result = search_web_context(query, options={"max_results": 2})
        if result.get("results"):
            snippets = " ".join([r.get("snippet", "") for r in result["results"]])
            
            # Use LLM to extract lat/lon
            prompt = f"Extract the latitude and longitude from the following text for {station_name} train station. Respond ONLY with a JSON object like {{\"lat\": -6.176, \"lon\": 106.827}}. If not found, return {{\"lat\": null, \"lon\": null}}.\n\nText: {snippets}"
            
            try:
                import json
                llm_response = run_llm_extraction(
                    system_prompt="You are a precise data extraction bot. Output only valid JSON.",
                    user_prompt=prompt
                )
                
                if llm_response:
                    # Parse JSON
                    start_idx = llm_response.find("{")
                    end_idx = llm_response.rfind("}")
                    if start_idx != -1 and end_idx != -1:
                        json_str = llm_response[start_idx:end_idx+1]
                        data = json.loads(json_str)
                        lat = data.get("lat")
                        lon = data.get("lon")
                        
                        if lat is not None and lon is not None:
                            _coords_cache[key] = (float(lat), float(lon))
                            if settings.enable_mongo_cache:
                                try:
                                    coll = get_collection("station_coordinates_cache")
                                    if coll is not None:
                                        coll.update_one(
                                            {"station_name": key},
                                            {"$set": {"lat": float(lat), "lon": float(lon)}},
                                            upsert=True
                                        )
                                except Exception:
                                    pass
                            return float(lat), float(lon)
            except Exception:
                pass
                
    return None, None

def get_weather(lat: float, lon: float) -> dict[str, Any]:
    client = get_openweather_client()
    if client is None:
        return {"used": False, "error": "OpenWeather disabled or API key missing"}
        
    started = perf_counter()
    try:
        response = client.get(
            "/onecall",
            params={
                "lat": lat,
                "lon": lon,
                "exclude": "minutely,hourly,daily",
                "units": "metric",
                "lang": "id",
                "appid": settings.open_weather_api_key
            }
        )
        response.raise_for_status()
        data = response.json()
        current = data.get("current", {})
        weather = current.get("weather", [{}])[0]
        
        return {
            "used": True,
            "temp": current.get("temp"),
            "feels_like": current.get("feels_like"),
            "humidity": current.get("humidity"),
            "description": weather.get("description"),
            "main": weather.get("main"),
            "latency_ms": round((perf_counter() - started) * 1000, 2),
            "error": None
        }
    except httpx.HTTPStatusError as exc:
        # Fallback to 2.5/weather if OneCall 3.0 fails (e.g. not subscribed)
        if exc.response.status_code in (401, 403):
            try:
                client.base_url = httpx.URL("https://api.openweathermap.org/data/2.5")
                resp = client.get(
                    "/weather",
                    params={
                        "lat": lat,
                        "lon": lon,
                        "units": "metric",
                        "lang": "id",
                        "appid": settings.open_weather_api_key
                    }
                )
                resp.raise_for_status()
                data = resp.json()
                main = data.get("main", {})
                weather = data.get("weather", [{}])[0]
                return {
                    "used": True,
                    "temp": main.get("temp"),
                    "feels_like": main.get("feels_like"),
                    "humidity": main.get("humidity"),
                    "description": weather.get("description"),
                    "main": weather.get("main"),
                    "latency_ms": round((perf_counter() - started) * 1000, 2),
                    "error": None,
                    "fallback_to_2_5": True
                }
            except Exception as inner_exc:
                return {"used": False, "error": f"OneCall failed, 2.5 fallback also failed: {str(inner_exc)}"}
        return {"used": False, "error": str(exc)}
    except Exception as exc:
        return {"used": False, "error": str(exc)}
    finally:
        client.close()
