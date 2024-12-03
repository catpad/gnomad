from dataclasses import dataclass
import requests
from typing import Optional


@dataclass
class WeatherInfo:
    temperature: float
    description: str
    humidity: int
    wind_speed: float
    feels_like: float


class WeatherService:
    def __init__(self, api_key: str):
        """Initialize weather service with OpenWeatherMap API key."""
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"

    def get_weather(self, lat: float, lng: float) -> Optional[WeatherInfo]:
        """Get current weather for given coordinates."""
        try:
            params = {
                'lat': lat,
                'lon': lng,
                'appid': self.api_key,
                'units': 'metric'  # Use Celsius
            }

            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            return WeatherInfo(
                temperature=data['main']['temp'],
                description=data['weather'][0]['description'],
                humidity=data['main']['humidity'],
                wind_speed=data['wind']['speed'],
                feels_like=data['main']['feels_like']
            )
        except Exception as e:
            print(f"Error getting weather data: {e}")
            return None

