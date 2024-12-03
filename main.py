import googlemaps
import random
import math
import time
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import requests
from geopy.distance import geodesic
from pathlib import Path
import wikipedia
from typing import Optional, Dict

from blogger import WORDPRESS_URL
from create_post import create_daily_blog_post, WordPressBlogGenerator
from defs import GOOGLE_API_KEY, WORDPRESS_USERNAME, WORDPRESS_PASSWORD, OPENAI_API_KEY
from postcard import GnomeImageGenerator
from selfie import GnomeSelfieGenerator


@dataclass
class City:
    name: str
    lat: float
    lng: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'City':
        return cls(**data)


@dataclass
class WikiInfo:
    title: str
    summary: str
    url: str
    full_content: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'WikiInfo':
        return cls(**data)


@dataclass
class PlaceOfInterest:
    name: str
    lat: float
    lng: float
    place_id: str
    types: List[str]
    rating: Optional[float] = None
    photo_references: List[str] = None
    address: Optional[str] = None
    wiki_info: Optional[WikiInfo] = None  # Added wiki_info field

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.wiki_info:
            data['wiki_info'] = self.wiki_info.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'PlaceOfInterest':
        wiki_info_data = data.pop('wiki_info', None)
        poi = cls(**data)
        if wiki_info_data:
            poi.wiki_info = WikiInfo.from_dict(wiki_info_data)
        return poi


class JourneyState:
    def __init__(self):
        self.current_position: Optional[City] = None
        self.previous_position: Optional[City] = None
        self.current_waypoint_index: int = 0
        self.total_distance: float = 0
        self.journey: List[Tuple[int, Tuple[float, float], str]] = []
        self.current_day: int = 0
        self.last_update: str = ""

    def to_dict(self) -> dict:
        return {
            'current_position': self.current_position.to_dict() if self.current_position else None,
            'previous_position': self.previous_position.to_dict() if self.previous_position else None,
            'current_waypoint_index': self.current_waypoint_index,
            'total_distance': self.total_distance,
            'journey': self.journey,
            'current_day': self.current_day,
            'last_update': self.last_update
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'JourneyState':
        state = cls()
        state.current_position = City.from_dict(data['current_position']) if data['current_position'] else None
        state.previous_position = City.from_dict(data['previous_position']) if data['previous_position'] else None
        state.current_waypoint_index = data['current_waypoint_index']
        state.total_distance = data['total_distance']
        state.journey = data['journey']
        state.current_day = data['current_day']
        state.last_update = data['last_update']
        return state


class TravelingGnomeBlog:
    def __init__(self, api_key: str, base_path: str = "gnome_journey"):
        """Initialize the traveling gnome blog with Google Maps API key."""
        self.gmaps = googlemaps.Client(key=api_key)
        self.base_path = Path(base_path)
        self.state = JourneyState()
        self.state_file = self.base_path / "journey_state.json"
        self.initialize_directories()

        # Set Wikipedia language to English
        wikipedia.set_lang('en')

    def get_wiki_info(self, query: str, search: bool = True) -> Optional[WikiInfo]:
        """
        Get Wikipedia information for a given query.
        If search is True, it will try to find the most relevant Wikipedia page.
        """
        try:
            if search:
                # Search for the most relevant page
                search_results = wikipedia.search(query, results=3)
                if not search_results:
                    return None

                # Try each result until we find a good match
                for result in search_results:
                    try:
                        page = wikipedia.page(result, auto_suggest=False)
                        # Basic relevance check - if the query appears in the title
                        if any(word.lower() in page.title.lower()
                               for word in query.lower().split()):
                            return WikiInfo(
                                title=page.title,
                                summary=page.summary,
                                url=page.url,
                                full_content=page.content
                            )
                    except (wikipedia.exceptions.DisambiguationError,
                            wikipedia.exceptions.PageError):
                        continue

                # If no good match found, try the first result
                try:
                    page = wikipedia.page(search_results[0], auto_suggest=False)
                    return WikiInfo(
                        title=page.title,
                        summary=page.summary,
                        url=page.url,
                        full_content=page.content
                    )
                except:
                    return None
            else:
                # Direct page lookup
                page = wikipedia.page(query, auto_suggest=False)
                return WikiInfo(
                    title=page.title,
                    summary=page.summary,
                    url=page.url,
                    full_content=page.content
                )

        except wikipedia.exceptions.DisambiguationError as e:
            # Try the first suggestion
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                return WikiInfo(
                    title=page.title,
                    summary=page.summary,
                    url=page.url,
                    full_content=page.content
                )
            except:
                return None
        except Exception as e:
            print(f"Error getting Wikipedia info for {query}: {str(e)}")
            return None

    def initialize_directories(self):
        """Create necessary directories for the blog."""
        self.base_path.mkdir(exist_ok=True)
        (self.base_path / "maps").mkdir(exist_ok=True)
        (self.base_path / "places").mkdir(exist_ok=True)
        (self.base_path / "street_view").mkdir(exist_ok=True)

    def geocode_city(self, city_name: str) -> City:
        """Convert city name to coordinates using Google Geocoding API."""
        try:
            result = self.gmaps.geocode(city_name)[0]
            location = result['geometry']['location']
            return City(
                name=city_name,
                lat=location['lat'],
                lng=location['lng']
            )
        except Exception as e:
            print(f"Error geocoding {city_name}: {str(e)}")
            raise

    def find_random_nearby_city(self, lat: float, lng: float, radius_km: float) -> tuple[City, bool]:
        """Find a random city within specified radius using Google Places API."""
        radius_meters = radius_km * 1000
        try:
            places_result = self.gmaps.places_nearby(
                location=(lat, lng),
                radius=radius_meters,
                type='locality'
            )

            if places_result.get('results'):
                place = random.choice(places_result['results'])
                location = place['geometry']['location']
                return City(
                    name=place['name'],
                    lat=location['lat'],
                    lng=location['lng']
                ), False
            else:
                return City(
                    name=f"Location near {lat:.2f}, {lng:.2f}",
                    lat=lat,
                    lng=lng
                ), True
        except Exception as e:
            print(f"Error finding nearby city: {str(e)}")
            raise

    def handle_ocean_crossing(self, start: City, target: City) -> City:
        """Handle crossing large bodies of water."""
        try:
            # Jump about 75% of the way to the target
            new_lat = start.lat + (target.lat - start.lat) * 0.75
            new_lng = start.lng + (target.lng - start.lng) * 0.75

            # Find nearest city to landing point
            landing_city, is_water = self.find_random_nearby_city(new_lat, new_lng, 100)
            if not is_water:
                return landing_city

            # If still over water, try closer to target
            new_lat = start.lat + (target.lat - start.lat) * 0.9
            new_lng = start.lng + (target.lng - start.lng) * 0.9
            landing_city, _ = self.find_random_nearby_city(new_lat, new_lng, 150)
            return landing_city

        except Exception as e:
            print(f"Error in ocean crossing: {str(e)}")
            # Emergency fallback - jump halfway
            new_lat = start.lat + (target.lat - start.lat) * 0.5
            new_lng = start.lng + (target.lng - start.lng) * 0.5
            landing_city, _ = self.find_random_nearby_city(new_lat, new_lng, 200)
            return landing_city

    def calculate_distance(self, point1: City, point2: City) -> float:
        """Calculate distance between two points in kilometers."""
        return geodesic(
            (point1.lat, point1.lng),
            (point2.lat, point2.lng)
        ).kilometers

    def get_next_point(self, start: City, target: City, distance_km: float) -> tuple:
        """Calculate next point given a start, target, and distance."""
        # Calculate bearing
        lat1 = math.radians(start.lat)
        lon1 = math.radians(start.lng)
        lat2 = math.radians(target.lat)
        lon2 = math.radians(target.lng)

        d_lon = lon2 - lon1

        y = math.sin(d_lon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)

        bearing = math.atan2(y, x)

        # Add some randomness to the bearing (±30 degrees)
        bearing += math.radians(random.uniform(-30, 30))

        # Calculate new point
        R = 6371.0  # Earth's radius in km

        lat2 = math.asin(
            math.sin(lat1) * math.cos(distance_km / R) +
            math.cos(lat1) * math.sin(distance_km / R) * math.cos(bearing)
        )

        lon2 = lon1 + math.atan2(
            math.sin(bearing) * math.sin(distance_km / R) * math.cos(lat1),
            math.cos(distance_km / R) - math.sin(lat1) * math.sin(lat2)
        )

        return math.degrees(lat2), math.degrees(lon2)

    def save_state(self):
        """Save current journey state to file."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def load_state(self) -> bool:
        """Load journey state from file. Returns True if successful."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                self.state = JourneyState.from_dict(data)
                return True
            return False
        except Exception as e:
            print(f"Error loading state: {e}")
            return False

    def create_daily_directory(self) -> Path:
        """Create and return path to directory for current day."""
        day_dir = self.base_path / f"day_{self.state.current_day}"
        day_dir.mkdir(exist_ok=True)
        return day_dir

    def save_map_image(self, url: str, filename: str):
        """Save map image from Google Static Maps API."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(f"Error saving map: {e}")

    def generate_overview_map(self) -> str:
        """Generate overall journey map with country view and prominent path."""
        if not self.state.journey:
            return ""

        # Get last location coordinates
        last_lat, last_lon = self.state.journey[-1][1]

        # Get country bounds using Geocoding API
        try:
            result = self.gmaps.reverse_geocode((last_lat, last_lon))
            country = next((
                component
                for component in result[0]['address_components']
                if 'country' in component['types']
            ), None)

            if country:
                # Get country boundaries
                geocode_result = self.gmaps.geocode(country['long_name'])
                viewport = geocode_result[0]['geometry']['viewport']

                # Extract country bounds
                country_bounds = (
                    f"{viewport['southwest']['lat']},{viewport['southwest']['lng']}|"
                    f"{viewport['northeast']['lat']},{viewport['northeast']['lng']}"
                )
            else:
                # Fallback to journey bounds if country lookup fails
                lats = [loc[1][0] for loc in self.state.journey]
                lons = [loc[1][1] for loc in self.state.journey]
                country_bounds = f"{min(lats)},{min(lons)}|{max(lats)},{max(lons)}"

        except Exception as e:
            print(f"Error getting country bounds: {e}")
            # Fallback to journey bounds
            lats = [loc[1][0] for loc in self.state.journey]
            lons = [loc[1][1] for loc in self.state.journey]
            country_bounds = f"{min(lats)},{min(lons)}|{max(lats)},{max(lons)}"

        # Create path with more prominent styling
        path_points = [f"{loc[1][0]},{loc[1][1]}" for loc in self.state.journey]

        # Create a more prominent path
        path = (
            # Main path - thicker red line
                f"path=color:0xFF0000|weight:8|" + "|".join(path_points) +
                # Glow effect - wider transparent line
                f"&path=color:0xFF000088|weight:12|" + "|".join(path_points)
        )

        # Add markers for start and end points
        markers = [
            # Start point - green marker
            f"markers=color:green|label:S|{path_points[0]}",
            # End point - red marker
            f"markers=color:red|label:E|{path_points[-1]}"
        ]

        base_url = "https://maps.googleapis.com/maps/api/staticmap?"
        params = {
            'size': '800x600',
            'maptype': 'terrain',
            'key': self.gmaps.key,
            'bounds': country_bounds,
            'style': 'feature:administrative.country|element:geometry.stroke|color:0x000000|weight:1'
        }

        # Combine URL parts
        url_parts = [
            base_url,
            "&".join(f"{k}={v}" for k, v in params.items()),
            path,
            "&".join(markers)
        ]

        return "&".join(part for part in url_parts if part)

    def generate_recent_map(self, days: int = 7) -> str:
        """Generate detailed map of recent journey with pins for each location visited."""
        recent_journey = self.state.journey[-days:] if len(self.state.journey) > days else self.state.journey
        if not recent_journey:
            return ""

        # Get coordinates for map bounds
        lats = [loc[1][0] for loc in recent_journey]
        lons = [loc[1][1] for loc in recent_journey]

        # Create path string for the route line
        path_points = [f"{loc[1][0]},{loc[1][1]}" for loc in recent_journey]
        path = f"color:0x0000FF|weight:5|" + "|".join(path_points)

        # Create markers for each location
        # Using different colors for start (green), intermediate (blue), and current (red) locations
        markers = []
        for i, loc in enumerate(recent_journey):
            if i == 0:  # Starting point
                markers.append(f"color:green|label:S|{loc[1][0]},{loc[1][1]}")
            elif i == len(recent_journey) - 1:  # Current location
                markers.append(f"color:red|label:C|{loc[1][0]},{loc[1][1]}")
            else:  # Intermediate stops
                markers.append(f"color:blue|{loc[1][0]},{loc[1][1]}")

        base_url = "https://maps.googleapis.com/maps/api/staticmap?"
        params = {
            'size': '800x600',
            'maptype': 'roadmap',
            'path': path,
            'key': self.gmaps.key,
            'markers': markers,  # Google Maps API will handle multiple marker strings
            'bounds': f"{min(lats)},{min(lons)}|{max(lats)},{max(lons)}"
        }

        # Construct URL with multiple markers
        url_parts = []
        for key, value in params.items():
            if isinstance(value, list):
                for v in value:
                    url_parts.append(f"markers={v}")
            else:
                url_parts.append(f"{key}={value}")

        return base_url + "&".join(url_parts)

    def find_places_of_interest(self, city: City) -> List[PlaceOfInterest]:
        """Find top 3 interesting places in current city and get their Wikipedia info."""
        try:
            places_result = self.gmaps.places_nearby(
                location=(city.lat, city.lng),
                radius=5000,  # 5km radius
                type=['tourist_attraction', 'museum', 'park', 'church', 'landmark']
            )

            places = []
            for place in places_result.get('results', []):
                photo_refs = []
                if 'photos' in place:
                    photo_refs = [photo['photo_reference'] for photo in place['photos']]

                poi = PlaceOfInterest(
                    name=place['name'],
                    lat=place['geometry']['location']['lat'],
                    lng=place['geometry']['location']['lng'],
                    place_id=place['place_id'],
                    types=place['types'],
                    rating=place.get('rating', 0),
                    photo_references=photo_refs,
                    address=place.get('vicinity')
                )
                places.append(poi)

            # Sort by rating and take top 3
            places.sort(key=lambda x: x.rating or 0, reverse=True)
            top_places = places[:3]

            # Get Wikipedia information for each place
            print("\nGathering Wikipedia information for places of interest:")
            for place in top_places:
                print(f"- Searching for {place.name}...")
                # Try to find Wikipedia info using the place name and city name
                wiki_info = self.get_wiki_info(f"{place.name} {city.name}")
                if not wiki_info:
                    # Try without city name
                    wiki_info = self.get_wiki_info(place.name)
                place.wiki_info = wiki_info
                if wiki_info:
                    print(f"  Found article: {wiki_info.title}")
                else:
                    print("  No Wikipedia article found")

            return top_places

        except Exception as e:
            print(f"Error finding places of interest: {e}")
            return []

    def save_wiki_info(self, city: City, places: List[PlaceOfInterest], day_dir: Path):
        """Save Wikipedia information for city and places to files."""
        wiki_dir = day_dir / "wiki_info"
        wiki_dir.mkdir(exist_ok=True)

        # Get and save city Wikipedia info
        print(f"\nGathering Wikipedia information for {city.name}...")
        city_wiki = self.get_wiki_info(city.name)
        if city_wiki:
            print(f"Found article: {city_wiki.title}")

            city_wiki_path = wiki_dir / "city_info.json"
            with open(city_wiki_path, 'w', encoding='utf-8') as f:
                json.dump(city_wiki.to_dict(), f, indent=2, ensure_ascii=False)

            # Save full content as markdown
            city_wiki_md = wiki_dir / "city_info.md"
            with open(city_wiki_md, 'w', encoding='utf-8') as f:
                f.write(f"# {city_wiki.title}\n\n")
                f.write(city_wiki.full_content)
        else:
            print("No Wikipedia article found")

        # Save places' Wikipedia info
        for i, place in enumerate(places, 1):
            if place.wiki_info:
                place_wiki_path = wiki_dir / f"place_{i}_{place.name.replace('/', '_')}_info.json"
                with open(place_wiki_path, 'w', encoding='utf-8') as f:
                    json.dump(place.wiki_info.to_dict(), f, indent=2, ensure_ascii=False)

                # Save full content as markdown
                place_wiki_md = wiki_dir / f"place_{i}_{place.name.replace('/', '_')}_info.md"
                with open(place_wiki_md, 'w', encoding='utf-8') as f:
                    f.write(f"# {place.wiki_info.title}\n\n")
                    f.write(place.wiki_info.full_content)

    def save_place_photos(self, place: PlaceOfInterest, day_dir: Path):
        """Save photos for a place of interest."""
        if not place.photo_references:
            return

        place_dir = day_dir / "places" / place.name.replace('/', '_')
        place_dir.mkdir(parents=True, exist_ok=True)

        for i, photo_ref in enumerate(place.photo_references):
            try:
                photo_data = self.gmaps.places_photo(
                    photo_ref,
                    max_width=800
                )
                photo_path = place_dir / f"photo_{i + 1}.jpg"
                with open(photo_path, 'wb') as f:
                    for chunk in photo_data:
                        f.write(chunk)
            except Exception as e:
                print(f"Error saving photo for {place.name}: {e}")

    def save_street_views(self, city: City, day_dir: Path) -> List[Path]:
        """Save three street view images for the current location."""
        street_view_paths = []
        headings = [0, 120, 240]  # Evenly spaced angles for variety

        try:
            street_view_dir = day_dir / "street_view"
            street_view_dir.mkdir(exist_ok=True)

            for i, heading in enumerate(headings, 1):
                # Construct Street View Static API URL
                base_url = "https://maps.googleapis.com/maps/api/streetview"
                params = {
                    'size': '800x600',
                    'location': f"{city.lat},{city.lng}",
                    'heading': str(heading),
                    'pitch': '10',
                    'fov': '90',
                    'source': 'outdoor',
                    'key': self.gmaps.key
                }

                url = f"{base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"

                # Get the image
                response = requests.get(url)

                if response.status_code == 200:
                    content_type = response.headers.get('content-type')
                    if content_type and 'image' in content_type:
                        if len(response.content) > 1000:
                            filepath = street_view_dir / f"view_{i}.jpg"
                            with open(filepath, 'wb') as f:
                                f.write(response.content)
                            print(f"Saved street view image {i} at heading {heading}°")
                            street_view_paths.append(filepath)

            return street_view_paths

        except Exception as e:
            print(f"Error capturing street views: {e}")
            return street_view_paths

    def move_to_next_position(self, waypoints: List[str]):
        """Calculate and move to next position."""
        if not self.state.current_position:
            return

        waypoint_cities = [self.geocode_city(city) for city in waypoints]
        next_waypoint = waypoint_cities[(self.state.current_waypoint_index + 1) % len(waypoints)]

        # Calculate next position
        distance_km = random.uniform(10, 50)
        next_lat, next_lng = self.get_next_point(
            self.state.current_position,
            next_waypoint,
            distance_km
        )

        # Find nearest city to this point and check if we're over water
        next_city, is_water = self.find_random_nearby_city(next_lat, next_lng, 20)

        # If we're over water, handle ocean crossing
        if is_water:
            print("\nWater detected! Crossing to next continent...")
            next_city = self.handle_ocean_crossing(self.state.current_position, next_waypoint)
            print("Landed on next continent!")

        # Update state
        self.state.previous_position = self.state.current_position
        self.state.current_position = next_city

        # Calculate and update total distance
        actual_distance = self.calculate_distance(self.state.previous_position, self.state.current_position)
        self.state.total_distance += actual_distance

        # Update journey log
        self.state.journey.append((
            self.state.current_day,
            (next_city.lat, next_city.lng),
            next_city.name
        ))

        # Check if we've reached next waypoint (within 5km)
        if self.calculate_distance(self.state.current_position, next_waypoint) < 5:
            self.state.current_waypoint_index += 1
            print(f"\nReached waypoint: {next_waypoint.name}")

    def daily_update(self, waypoints: List[str]) -> bool:
        """Execute one day's journey update. Returns True if completed."""
        try:
            # Check if state exists, if not initialize it
            if not self.load_state():
                print("Starting new journey...")
                self.state.current_position = self.geocode_city(waypoints[0])
                self.state.current_day = 0
                self.state.journey = [(0, (self.state.current_position.lat,
                                           self.state.current_position.lng),
                                       self.state.current_position.name)]

            # Check if we've already updated today
            today = datetime.now().strftime("%Y-%m-%d")
            if self.state.last_update == today:
                print("Already updated today")
                return False

            print(f"\nDay {self.state.current_day + 1} Update")
            print("=" * 40)

            # Create directory for today's update
            day_dir = self.create_daily_directory()

            # Save current position information
            current_position_info = {
                'day': self.state.current_day,
                'current_city': self.state.current_position.to_dict(),
                'total_distance': self.state.total_distance,
                'date': today
            }

            with open(day_dir / 'position_info.json', 'w') as f:
                json.dump(current_position_info, f, indent=2)

            # Generate and save maps
            print("Generating maps...")
            overview_map_url = self.generate_overview_map()
            recent_map_url = self.generate_recent_map()

            if overview_map_url:
                self.save_map_image(overview_map_url, day_dir / "overview_map.png")
            if recent_map_url:
                self.save_map_image(recent_map_url, day_dir / "recent_map.png")

            print("Capturing street view images...")
            self.save_street_views(self.state.current_position, day_dir)

            # Find and save information about interesting places
            print("Finding places of interest...")
            places = self.find_places_of_interest(self.state.current_position)

            # Save places information
            places_info = [place.to_dict() for place in places]
            with open(day_dir / 'places_of_interest.json', 'w') as f:
                json.dump(places_info, f, indent=2)

            # Save photos for each place
            print(f"Saving photos for {len(places)} places...")
            for place in places:
                self.save_place_photos(place, day_dir)

            # Save street view images
            # print("Capturing street view images...")
            # self.save_street_view(self.state.current_position, day_dir)

            # Move to next position
            print("Moving to next position...")
            self.move_to_next_position(waypoints)

            # Update state
            self.state.last_update = today
            self.state.current_day += 1
            self.save_state()

            print("\nDaily update completed successfully!")
            print(f"Current Location: {self.state.current_position.name}")
            print(f"Total Distance: {self.state.total_distance:.2f} km")
            return True

        except Exception as e:
            print(f"Error during daily update: {e}")
            raise


def main():
    # Define waypoints for the journey
    waypoints = [
        "Cabo da Roca", 
    ]

    image_generator = GnomeImageGenerator(OPENAI_API_KEY)
    selfie_generator = GnomeSelfieGenerator(Path("selfies"))  # Directory containing selfie_*.png files

    # Create the gnome blog instance
    gnome_blog = TravelingGnomeBlog(GOOGLE_API_KEY)

    # Create WordPress blog generator
    wp_generator = WordPressBlogGenerator(
        wordpress_url=WORDPRESS_URL,
        wordpress_user=WORDPRESS_USERNAME,
        wordpress_pass=WORDPRESS_PASSWORD,
        openai_key=OPENAI_API_KEY,
    )

    try:
        # Execute daily update
        if gnome_blog.daily_update(waypoints):
            print("\nDaily update completed successfully!")

            # Get the current day directory
            current_day = gnome_blog.state.current_day
            day_dir = gnome_blog.base_path / f"day_{current_day}"

            # Create and publish blog post
            if create_daily_blog_post(wp_generator, day_dir,
                                      image_generator=image_generator,
                                      selfie_generator=selfie_generator):
                print("Blog post published successfully!")
            else:
                print("Failed to publish blog post")
        else:
            print("\nNo update needed for today.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()


