import time

import googlemaps
from PIL import Image as PILImage
from wordpress_xmlrpc import Client, WordPressPost
from wordpress_xmlrpc.methods.posts import NewPost
from wordpress_xmlrpc.methods.media import UploadFile
from openai import OpenAI
import random
from typing import List, Dict, Optional
from pathlib import Path
import json
from datetime import datetime
import base64
from dataclasses import dataclass
import xmlrpc.client as xmlrpc_client

from defs import GOOGLE_API_KEY, WEATHER_API_KEY
from postcard import GnomeImageGenerator
from selfie import GnomeSelfieGenerator
from weather import WeatherInfo, WeatherService


@dataclass
class BlogContent:
    journey_description: str
    city_description: str
    cultural_insight: str
    poi_descriptions: List[str]
    literary_quote: Optional[dict] = None


class WordPressBlogGenerator:
    def __init__(self, wordpress_url: str, wordpress_user: str, wordpress_pass: str, openai_key: str):
        """Initialize WordPress and OpenAI clients."""
        self.wp_client = Client(wordpress_url, wordpress_user, wordpress_pass)
        self.openai_client = OpenAI(api_key=openai_key)
        self.gmaps = googlemaps.Client(key=GOOGLE_API_KEY)
        self.weather_service = WeatherService(WEATHER_API_KEY)
        self.selected_selfie = None

        # Track journey stats
        self.total_distance = 0
        self.days_on_road = 0
        self.visited_waypoints = []

        self.cultural_topics = [
            "local_cuisine",
            "traditions",
            "arts_and_culture",
            "daily_life",
            "history",
            "culinary",
            "language_lesson"  # Added new topic
        ]

        self.topic_prompts = {
            "local_cuisine": "What's unique about {city}'s food scene?",
            "traditions": "What's an interesting tradition in {city}?",
            "arts_and_culture": "What defines {city}'s cultural scene?",
            "daily_life": "What's daily life like in {city}?",
            "history": "What's a fascinating piece of {city}'s history?",
            "culinary": "Provide a detailed local recipe of {city}",
            "language_lesson": "Create a fun, tiny language lesson about {country}'s language. Include 3-4 useful "
                               "phrases with pronunciation and cultural context."
        }

    def update_journey_context(self, total_distance: float, days: int, waypoints: List[str]):
        """Update journey statistics for context."""
        self.total_distance = total_distance
        self.days_on_road = days
        self.visited_waypoints = waypoints

    def _get_journey_context(self) -> str:
        """Generate context string including journey statistics."""
        waypoints_str = ", ".join(self.visited_waypoints[-5:])  # Show last 5 waypoints
        return f"""I am a traveling garden gnome Oliver documenting my journey around the world on foot. 
        I'm knowledgeable, grumpy, curious, and always excited to learn about new places and cultures.
        I'm extremely friendly, very funny, sometimes grumpy but always optimistic.
        I like to pepper my posts with jokes, verses, quotes and philosophical musings.
        I never just describe a place: I describe my own visit and participation!

        Journey Stats:
        - I've been on the road for {self.days_on_road} days
        - Traveled {self.total_distance:.1f} kilometers so far
        - Recent stops include: {waypoints_str}
        """

    def get_openai_completion(self, prompt: str, max_tokens: int = 1150) -> str:
        """Get completion from OpenAI API with context."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self._get_journey_context()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting OpenAI completion: {e}")
            return "Content generation failed."

    def generate_weather_description(self, weather: WeatherInfo, city_name: str) -> str:
        """Generate a gnome's description of the weather."""
        prompt = f"""As a traveling gnome in {city_name}, describe today's weather:
        Temperature: {weather.temperature}°C (feels like {weather.feels_like}°C)
        Description: {weather.description}
        Humidity: {weather.humidity}%
        Wind Speed: {weather.wind_speed} m/s

        Make it funny and personal - how does this weather affect a garden gnome?
        Keep it under 100 words."""

        return self.get_openai_completion(prompt, max_tokens=150)

    def generate_journey_description(self, distance: float, prev_city: str, prev_country: str,
                                     current_city: str, current_country: str) -> str:
        """Generate a short journey update."""
        prompt = f"""Briefly describe my {distance:.1f}km journey from {prev_city}, {prev_country} to {current_city}, {current_country}. 
                    Focus on one interesting observation.
                    Greet readers in local language of {current_country}.
                    Eventually mention overall distance traveled, the cities you visited and the days spend on the road.
                    Insert phrases and sentences in the local language with translation.
                    If I crossed a border, note the language changes.
                    Pepper my posts with local songs, verses, quotes and philosophical musings.
                    Mark important words, names and locations with HTML bold <b>.
                    Mark quotes with HTML italic <i>."""
        return self.get_openai_completion(prompt, max_tokens=1000)

    def enhance_city_description(self, wiki_content: str, city_name: str, country: str) -> str:
        """Create a brief city description."""
        prompt = f"""Based on this info about {city_name}: {wiki_content[:500]}... 
                    Share two fascinating things about this city.
                    Keep it under 175 words.
                    When possible insert phrases and sentences in {country}'s language with translation.
                    Insert cultural references to {country}'s artists, poets, composers, etc.
                    Mark important words, names and locations with HTML bold <b>.
                    Mark quotes with HTML italic <i>."""
        return self.get_openai_completion(prompt, max_tokens=1000)

    def generate_cultural_insight(self, city_name: str, country: str, wiki_content: str) -> tuple[str, str]:
        """Generate a brief cultural insight."""
        topic = random.choice(self.cultural_topics)
        prompt_template = self.topic_prompts[topic]

        base_prompt = f"Using this background: {wiki_content[:300]}... {prompt_template.format(city=city_name, country=country)} "
        if topic == "language_lesson":
            prompt = f"""{base_prompt}
                        Include common phrases in {country}'s language with pronunciation.
                        Explain any unique aspects of the local language or dialect in {city_name}.
                        Keep it under 150 words."""
        else:
            prompt = f"""{base_prompt}
                        Use local language phrases from {country} with translations.
                        Keep it under 150 words.
                        Mark important words, names and locations with HTML bold <b>.
                        Mark quotes with HTML italic <i>."""

        return topic, self.get_openai_completion(prompt, max_tokens=1000)

    def generate_basic_poi_description(self, poi_name: str) -> str:
        """Generate a very brief POI description without Wiki info."""
        prompt = f"What might make {poi_name} interesting to visit? One or two sentences only describing YOUR OWN visit and adventures. " \
                 f"Make it funny." \
                 f"Mark important words, names and locations with HTML bold <b>. Mark quotes with HTML italic <i>."
        return self.get_openai_completion(prompt, max_tokens=1000)

    def generate_blog_content(self, day_dir: Path) -> Optional[BlogContent]:
        """Generate all blog content using OpenAI."""
        try:
            current_day_num = int(day_dir.name.split('_')[1])

            with open(day_dir / 'position_info.json', 'r') as f:
                position_info = json.load(f)

            # Get country for current location
            current_result = self.gmaps.reverse_geocode(
                (position_info['current_city']['lat'],
                 position_info['current_city']['lng'])
            )
            current_country = next((
                component['long_name']
                for component in current_result[0]['address_components']
                if 'country' in component['types']
            ), "Unknown Country")

            # Get previous day's info and country
            prev_day_dir = day_dir.parent / f"day_{current_day_num - 1}"
            prev_position = None
            prev_country = "Unknown Country"

            if prev_day_dir.exists():
                try:
                    with open(prev_day_dir / 'position_info.json', 'r') as f:
                        prev_position = json.load(f)
                    prev_result = self.gmaps.reverse_geocode(
                        (prev_position['current_city']['lat'],
                         prev_position['current_city']['lng'])
                    )
                    prev_country = next((
                        component['long_name']
                        for component in prev_result[0]['address_components']
                        if 'country' in component['types']
                    ), "Unknown Country")
                except FileNotFoundError:
                    prev_position = position_info
                    prev_country = current_country
            else:
                prev_position = position_info
                prev_country = current_country

            # Calculate the actual distance traveled
            distance = position_info['total_distance'] - prev_position['total_distance']

            # Get city wiki info
            city_wiki_content = "A city with rich culture and history."  # Default
            city_wiki_path = day_dir / 'wiki_info/city_info.json'
            if city_wiki_path.exists():
                with open(city_wiki_path, 'r') as f:
                    city_wiki = json.load(f)
                    city_wiki_content = city_wiki.get('full_content', city_wiki_content)

            # Generate content with country context
            journey_description = self.generate_journey_description(
                distance,
                prev_position['current_city']['name'],
                prev_country,
                position_info['current_city']['name'],
                current_country
            )

            city_description = self.enhance_city_description(
                city_wiki_content,
                position_info['current_city']['name'],
                current_country
            )

            _, cultural_insight = self.generate_cultural_insight(
                position_info['current_city']['name'],
                current_country,
                city_wiki_content
            )

            # Handle POI descriptions
            poi_descriptions = []
            poi_path = day_dir / 'places_of_interest.json'
            if poi_path.exists():
                with open(poi_path, 'r') as f:
                    pois = json.load(f)

                for poi in pois[:3]:
                    poi_description = self.generate_basic_poi_description(poi['name'])
                    poi_descriptions.append(poi_description)

            return BlogContent(
                journey_description=journey_description,
                city_description=city_description,
                cultural_insight=cultural_insight,
                poi_descriptions=poi_descriptions
            )

        except Exception as e:
            print(f"Error generating blog content: {e}")
            print(f"Current directory being processed: {day_dir}")
            return None

    def _get_poi_photos(self, day_dir: Path) -> List[List[str]]:
        """Get lists of photo paths for each POI."""
        try:
            with open(day_dir / 'places_of_interest.json', 'r') as f:
                pois = json.load(f)

            photo_lists = []
            for poi in pois[:3]:  # Limit to 3 POIs
                poi_dir = day_dir / "places" / poi['name'].replace('/', '_')
                if poi_dir.exists():
                    photos = list(poi_dir.glob("photo_*.jpg"))
                    photo_lists.append([str(p) for p in photos[:1]])  # Only take first photo
                else:
                    photo_lists.append([])

            return photo_lists
        except Exception as e:
            print(f"Error getting POI photos: {e}")
            return []

    def generate_funny_title(self, section_type: str, city_name: str, content: str) -> str:
        """Generate a funny, contextual title for a section."""
        prompts = {
            'journey': f"""Given this travel update: "{content}"
                Create a short (2-5 words), hilarious title for this journey section. 
                Use puns, local context, or travel humor. Make it catchy and gnome-related if possible.""",

            'city': f"""Given this city description: "{content}"
                Create a short (2-5 words), funny title about {city_name}. 
                Use local stereotypes, cultural quirks, or city-specific jokes. Keep it light and playful.""",

            'culture': f"""Based on this cultural insight: "{content}"
                Create a short (2-5 words), witty title that plays with cultural elements mentioned.
                Use clever wordplay or humorous observations about local customs.""",

            'places': f"""Create a short (2-5 words), playful title for a section about local attractions in {city_name}.
                Make it gnome-related if possible. Be clever but not cheesy."""
        }

        prompt = prompts.get(section_type, "Create a funny short title")
        title = self.get_openai_completion(prompt, max_tokens=20).strip()

        # Remove quotes if present
        title = title.strip('"\'')
        return title

    def generate_map_titles(self, distance: float, country: str, total_distance: float, content: str) -> tuple[
        str, str]:
        """Generate funny titles for the recent and overview maps."""
        recent_prompt = f"""Given a {distance:.1f}km journey through {country}, create a funny, short (2-5 words) title 
        for a map showing recent travels. Use humor, puns, or wordplay related to the distance, country, or travel. 
        Make it gnome-related if possible."""

        overview_prompt = f"""Given a total journey of {total_distance:.1f}km with current location in {country}, 
        create a funny, short (2-5 words) title for an overview map. Consider the scale of the journey and make 
        it gnome-related if possible. Use humor or puns about world travel, exploration, or adventure."""

        recent_title = self.get_openai_completion(recent_prompt, max_tokens=20).strip().strip('"\'')
        overview_title = self.get_openai_completion(overview_prompt, max_tokens=20).strip().strip('"\'')

        return recent_title, overview_title

    def create_html_post(self, day_dir: Path, content: BlogContent) -> str:
        """Create beautifully formatted HTML blog post with postcard and funny titles."""
        try:
            with open(day_dir / 'position_info.json', 'r') as f:
                position_info = json.load(f)

            # Get the day number from the directory name
            current_day_num = int(day_dir.name.split('_')[1])

            city_name = position_info['current_city']['name']
            date = datetime.now().strftime("%B %d, %Y")

            # Get country for postcard generation
            result = self.gmaps.reverse_geocode(
                (position_info['current_city']['lat'],
                 position_info['current_city']['lng'])
            )
            country = next((
                component['long_name']
                for component in result[0]['address_components']
                if 'country' in component['types']
            ), None)

            # Store current valid images for POI section
            self.current_images = {}
            places_dir = day_dir / "places"
            if places_dir.exists():
                for i in range(1, 4):  # Check first 3 POIs
                    for poi_dir in places_dir.iterdir():
                        photo_path = poi_dir / f"selfie_photo_{i}.png"  # Check for selfie version first
                        if not photo_path.exists():
                            photo_path = poi_dir / f"photo_{i}.jpg"  # Fallback to original photo
                        if photo_path.exists():
                            self.current_images[f'poi_{i}_photo_1.jpg'] = photo_path

            # Generate map titles
            recent_map_title = "Last Week on the Road"
            overview_map_title = "My Epic Journey So Far"

            # Generate funny titles for sections
            journey_title = self.generate_funny_title('journey', city_name, content.journey_description)
            city_title = self.generate_funny_title('city', city_name, content.city_description)
            culture_title = self.generate_funny_title('culture', city_name, content.cultural_insight)
            places_title = self.generate_funny_title('places', city_name, " ".join(content.poi_descriptions))

            html = f"""
            <article style="font-family: Georgia, serif; max-width: 1200px; margin: 0 auto; line-height: 1.6; color: #2c3e50;">
                <header style="text-align: center; margin-bottom: 2em; padding: 2em 0; border-bottom: 2px solid #eaeaea;">
                    <h1 style="font-size: 2.5em; color: #2c3e50; margin-bottom: 0.5em;">{city_name}</h1>
                    <div style="color: #7f8c8d; font-size: 1.1em;">{date}</div>
                </header>

                <!-- Postcard Section -->
                <div style="text-align: center; margin: 2em 0;">
                    <img src="postcard.png" alt="Gnome's postcard from {country}" 
                         style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"/>
                    <div style="font-style: italic; color: #7f8c8d; margin-top: 1em;">Greetings from {city_name}!</div>
                </div>

                <hr style="border: none; height: 2px; background: #eaeaea; margin: 2em 0;"/>

                <!-- Maps Section -->
                <div style="display: flex; justify-content: space-between; gap: 20px; margin: 2em 0;">
                    <div style="flex: 1; text-align: center;">
                        <h3 style="color: #2c3e50; margin-bottom: 1em;">{recent_map_title}</h3>
                        <img src="recent_map.png" alt="Recent journey map" 
                             style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"/>
                    </div>

                    <div style="flex: 1; text-align: center;">
                        <h3 style="color: #2c3e50; margin-bottom: 1em;">{overview_map_title}</h3>
                        <img src="overview_map.png" alt="Overall journey map" 
                             style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"/>
                    </div>
                </div>

                <hr style="border: none; height: 2px; background: #eaeaea; margin: 2em 0;"/>

                <!-- Journey Stats Box -->
                <div style="background: #f8f9fa; border-radius: 8px; padding: 1.5em; margin: 2em 0; text-align: center; border: 1px solid #e9ecef;">
                    <div style="display: flex; justify-content: center; gap: 4em;">
                        <div>
                            <div style="font-size: 2.5em; color: #2c3e50; font-weight: bold;">{current_day_num}</div>
                            <div style="color: #6c757d; text-transform: uppercase; font-size: 0.9em;">Days on the Road</div>
                        </div>
                        <div>
                            <div style="font-size: 2.5em; color: #2c3e50; font-weight: bold;">{position_info['total_distance']:.0f}</div>
                            <div style="color: #6c757d; text-transform: uppercase; font-size: 0.9em;">Kilometers Traveled</div>
                        </div>
                    </div>
                </div>
            """

            # Get weather information
            weather = self.weather_service.get_weather(
                position_info['current_city']['lat'],
                position_info['current_city']['lng']
            )

            if weather:
                weather_description = self.generate_weather_description(
                    weather,
                    position_info['current_city']['name']
                )

                # Add weather section
                html += f"""
                <section style="margin: 2.5em 0; padding: 1.5em;">
                    <h2 style="color: #2c3e50; font-size: 1.8em; margin-bottom: 1em; 
                        text-align: center;
                        padding-bottom: 0.5em; border-bottom: 2px solid #3498db;">
                        Today's Weather Report
                    </h2>
                    <div style="display: flex; align-items: center; gap: 2em; 
                        background: #f5f6fa; padding: 1.5em; border-radius: 8px;">
                        <div style="flex: 1;">
                            <p style="font-size: 1.1em; margin-bottom: 1em;">
                                {weather_description}
                            </p>
                            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1em;">
                                <div>🌡️ Temperature: {weather.temperature}°C</div>
                                <div>💨 Wind: {weather.wind_speed} m/s</div>
                                <div>💧 Humidity: {weather.humidity}%</div>
                                <div>🤔 Feels like: {weather.feels_like}°C</div>
                            </div>
                        </div>
                    </div>
                </section>
                """

            # Add Journey Section
            html += f"""
                <section style="margin: 2.5em 0; padding: 1.5em;">
                    <h2 style="color: #2c3e50; font-size: 1.8em; margin-bottom: 1em; padding-bottom: 0.5em; border-bottom: 2px solid #e67e22;">{journey_title}</h2>
                    <div style="text-align: justify; padding: 0 1em;">
                        {content.journey_description}
                    </div>
                </section>

                <hr style="border: none; height: 2px; background: #eaeaea; margin: 2em 0;"/>
            """

            # Add Street Views Section if images exist
            street_view_dir = day_dir / "street_view"
            if street_view_dir.exists() and any(street_view_dir.glob("view_*.jpg")):
                html += f"""
                <section style="margin: 2.5em 0;">
                    <h2 style="color: #2c3e50; font-size: 1.8em; margin-bottom: 1em; text-align: center;">Wandering {city_name}'s Streets</h2>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5em; margin: 2em 0;">
                """

                for i in range(1, 4):
                    if (street_view_dir / f"view_{i}.jpg").exists():
                        html += f"""
                        <div class="view-card" style="background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                            <img src="street_view_{i}.jpg" alt="Street view {i}" style="width: 100%; aspect-ratio: 4/3; object-fit: cover;"/>
                            <div style="padding: 1em; text-align: center; color: #666;">Looking {['North', 'Southeast', 'Southwest'][i - 1]}</div>
                        </div>
                        """

                html += """
                    </div>
                </section>
                <hr style="border: none; height: 2px; background: #eaeaea; margin: 2em 0;"/>
                """

            # Add City Section
            html += f"""
                <section style="margin: 2.5em 0; padding: 1.5em;">
                    <h2 style="color: #2c3e50; font-size: 1.8em; margin-bottom: 1em; padding-bottom: 0.5em; border-bottom: 2px solid #e67e22;">{city_title}</h2>
                    <div style="text-align: justify; padding: 0 1em;">
                        {content.city_description}
                    </div>
                </section>

                <hr style="border: none; height: 2px; background: #eaeaea; margin: 2em 0;"/>

                <section style="margin: 2.5em 0; padding: 1.5em;">
                    <h2 style="color: #2c3e50; font-size: 1.8em; margin-bottom: 1em; padding-bottom: 0.5em; border-bottom: 2px solid #e67e22;">{culture_title}</h2>
                    <blockquote style="border-left: 4px solid #e67e22; margin: 1.5em 0; padding: 1em 2em; background: #f9f9f9; font-style: italic;">
                        {content.cultural_insight}
                    </blockquote>
                </section>
            """

            # Add POI Section only if there are valid POIs with images
            if self.current_images:
                html += self._create_poi_section(content, places_title)

            # Add Random Selfie Section
            if self.selected_selfie:
                html += f"""
                <hr style="border: none; height: 2px; background: #eaeaea; margin: 2em 0;"/>

                <section style="margin: 2.5em 0; padding: 1.5em; text-align: center;">
                    <h2 style="color: #2c3e50; font-size: 1.8em; margin-bottom: 1em; padding-bottom: 0.5em; border-bottom: 2px solid #e67e22;">See you tomorrow!</h2>
                    <div style="max-width: 500px; margin: 0 auto;">
                        <img src="{self.selected_selfie.name}" alt="Gnome selfie" 
                             style="width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"/>
                    </div>
                </section>
                """

            html += "</article>"
            return html

        except Exception as e:
            print(f"Error creating HTML post: {e}")
            return ""

    def _create_selfie_section(self, day_dir: Path) -> str:
        """Create the selfie section HTML."""
        try:
            if self.selected_selfie:  # Use the stored selfie
                return f"""
                <hr style="border: none; height: 2px; background: #eaeaea; margin: 2em 0;"/>

                <section style="margin: 2.5em 0; padding: 1.5em; text-align: center;">
                    <h2 style="color: #2c3e50; font-size: 1.8em; margin-bottom: 1em; padding-bottom: 0.5em; border-bottom: 2px solid #e67e22;">See you tomorrow!</h2>
                    <div style="max-width: 500px; margin: 0 auto;">
                        <img src="{self.selected_selfie.name}" alt="Gnome selfie" 
                             style="width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"/>
                    </div>
                </section>
                """
            return ""  # Return empty string if no selfie was selected
        except Exception as e:
            print(f"Error creating selfie section: {e}")
            return ""

    def _create_poi_section(self, content: BlogContent, places_title: str) -> str:
        """Create the HTML for the POIs section, skipping POIs without images."""
        if not content.poi_descriptions:
            return ""

        poi_html = f"""
        <hr style="border: none; height: 2px; background: #eaeaea; margin: 2em 0;"/>
        <section style="margin: 2.5em 0; padding: 1.5em;">
            <h2 style="color: #2c3e50; font-size: 1.8em; margin-bottom: 1em; padding-bottom: 0.5em; border-bottom: 2px solid #e67e22;">{places_title}</h2>
        """

        valid_pois = 0
        for i, desc in enumerate(content.poi_descriptions, 1):
            # Check if image exists for this POI
            img_key = f'poi_{i}_photo_1.jpg'
            if img_key not in self.current_images:
                print(f"Skipping POI {i} - no image available")
                continue

            valid_pois += 1
            poi_html += f"""
            <div style="margin-bottom: 3em;">
                <img src="{img_key}" alt="Gnome at POI {valid_pois}" 
                     style="width: 100%; max-height: 800px; object-fit: cover; border-radius: 8px; margin-bottom: 1em;"/>
                <div style="padding: 1em; text-align: justify; background: #fff; border-radius: 8px;">
                    {desc}
                </div>
            </div>
            """

        if valid_pois == 0:
            return ""  # Return empty string if no valid POIs

        poi_html += "</section>"
        return poi_html

    def publish_post(self, title: str, content: str, images: Dict[str, str], city_name: str) -> bool:
        """Publish post to WordPress with images."""

        def upload_image_with_retry(img_path: str, max_retries: int = 5, chunk_size: int = 1024 * 1024) -> Optional[
            dict]:
            """Helper function to upload image with retries and chunked reading."""
            for attempt in range(max_retries):
                try:
                    # Read and possibly resize image
                    with PILImage.open(img_path) as img:
                        # Always convert to RGB to avoid transparency issues
                        if img.mode in ('RGBA', 'P'):
                            img = img.convert('RGB')

                        # If image is larger than 1MB, resize it
                        if Path(img_path).stat().st_size > 1_000_000:
                            ratio = min(1920 / img.width, 1080 / img.height)
                            new_size = (int(img.width * ratio), int(img.height * ratio))
                            img = img.resize(new_size, PILImage.Resampling.LANCZOS)

                            # Save to temporary file with lower quality
                            temp_path = str(Path(img_path).parent / f"temp_{Path(img_path).name}")
                            img.save(temp_path, 'JPEG', quality=80, optimize=True)
                            img_path = temp_path

                    # Read file in chunks
                    with open(img_path, 'rb') as f:
                        image_data = f.read()

                    # Prepare upload data
                    filename = Path(img_path).name
                    data = {
                        'name': f'gnome_journey/{filename}',
                        'type': 'image/jpeg' if filename.lower().endswith('.jpg') else 'image/png',
                        'bits': xmlrpc_client.Binary(image_data)
                    }

                    # Upload to WordPress
                    response = self.wp_client.call(UploadFile(data))

                    # Clean up temporary file if it exists
                    if 'temp_' in img_path:
                        Path(img_path).unlink(missing_ok=True)

                    print(f"Successfully uploaded: {filename}")
                    return response

                except Exception as e:
                    print(f"Upload attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(2 * (attempt + 1))  # Exponential backoff

                    if attempt == max_retries - 1:
                        print(f"Failed to upload {img_path} after {max_retries} attempts")
                        return None

            return None

        try:

            def get_country_for_city(city: str) -> str:
                """Get country for a given city using Google Maps Geocoding API."""
                try:
                    # Use existing gmaps client to get country info
                    result = self.gmaps.geocode(city)
                    if result:
                        # Extract country from address components
                        for component in result[0]['address_components']:
                            if 'country' in component['types']:
                                return component['long_name']
                    return "Unknown Country"
                except Exception as e:
                    print(f"Error getting country for {city}: {e}")
                    return "Unknown Country"

            # Get country for the city
            country = get_country_for_city(city_name)

            # Create new post
            post = WordPressPost()
            post.title = title
            post.post_status = 'publish'
            post.terms_names = {
                'post_tag': [country, city_name],
                'category': [country]
            }

            # Upload images and replace URLs in content
            thumbnail_id = None
            uploaded_images = {}

            # First, upload all images
            for img_placeholder, img_path in images.items():
                response = upload_image_with_retry(str(img_path))
                if response:
                    uploaded_images[img_placeholder] = response['url']
                    if thumbnail_id is None:
                        thumbnail_id = response['id']
                else:
                    print(f"Failed to upload {img_path}")
                    return False

            # Replace all image placeholders in content
            updated_content = content
            for placeholder, url in uploaded_images.items():
                updated_content = updated_content.replace(placeholder, url)

            # Handle POI image placeholders
            for i in range(1, 4):  # Assuming maximum 3 POIs
                poi_placeholder = f"POI_IMAGE_PLACEHOLDER_{i}"
                poi_image_key = f'poi_{i}_photo_1.jpg'
                if poi_image_key in uploaded_images:
                    updated_content = updated_content.replace(poi_placeholder, uploaded_images[poi_image_key])

            # Set content and thumbnail
            post.content = updated_content
            if thumbnail_id:
                post.thumbnail = thumbnail_id

            # Publish post with retries
            for attempt in range(3):
                try:
                    post_id = self.wp_client.call(NewPost(post))
                    print(f"Successfully published post with ID: {post_id}")
                    return True
                except Exception as e:
                    print(f"Publish attempt {attempt + 1} failed: {str(e)}")
                    if attempt < 2:  # Don't sleep on last attempt
                        time.sleep(2 * (attempt + 1))

            return False

        except Exception as e:
            print(f"Error publishing post: {e}")
            return False


def generate_random_image(image_generator, country: str, actual_day_dir) -> str:
    """
    Randomly selects and calls either the postcard or map generation function.

    Args:
        image_generator: Instance of the image generator class
        country: Name of the country
        actual_day_dir: Path to the day's directory

    Returns:
        str: Path to the generated image
    """
    # Randomly choose between postcard and map
    generator_function = random.choice([
        image_generator.generate_postcard,
        image_generator.generate_map
    ])

    # Call the selected function with the same parameters
    return generator_function(country, actual_day_dir)


def create_daily_blog_post(blog_generator: WordPressBlogGenerator, day_dir: Path,
                           image_generator: GnomeImageGenerator,
                           selfie_generator: GnomeSelfieGenerator) -> bool:
    """Create and publish daily blog post with generated images."""
    try:
        # Get the actual directory containing the data
        current_day_num = int(day_dir.name.split('_')[1])
        actual_day_dir = day_dir.parent / f"day_{current_day_num - 1}"

        if not actual_day_dir.exists():
            print(f"Cannot find data directory: {actual_day_dir}")
            return False

        print(f"Using data from directory: {actual_day_dir}")

        # Get country information for image generation
        with open(actual_day_dir / 'position_info.json', 'r') as f:
            position_info = json.load(f)

        result = blog_generator.gmaps.reverse_geocode(
            (position_info['current_city']['lat'],
             position_info['current_city']['lng'])
        )
        country = next((
            component['long_name']
            for component in result[0]['address_components']
            if 'country' in component['types']
        ), None)

        # Update journey context
        with open(actual_day_dir / 'position_info.json', 'r') as f:
            position_info = json.load(f)

        # Get previous waypoints from journey state
        with open(day_dir.parent / "journey_state.json", 'r') as f:
            journey_state = json.load(f)
            waypoints = [loc[2] for loc in journey_state['journey']]

        blog_generator.update_journey_context(
            total_distance=position_info['total_distance'],
            days=current_day_num,
            waypoints=waypoints
        )

        # Generate postcard first
        print(f"Generating postcard for {country}...")

        city_name = position_info['current_city']['name']

        postcard_path = image_generator.generate_random_image(country, day_dir, city_name)

        if not postcard_path:
            print("Failed to generate postcard")
            return False
        print(f"Postcard generated successfully at: {postcard_path}")

        # Generate blog content
        print("Generating blog content...")
        content = blog_generator.generate_blog_content(actual_day_dir)
        if not content:
            print("Failed to generate blog content")
            return False

        # Prepare images dictionary
        images = {
            'postcard.png': postcard_path,
            'recent_map.png': actual_day_dir / 'recent_map.png',
            'overview_map.png': actual_day_dir / 'overview_map.png'
        }

        # Add random selfie
        selfies = list(Path("selfies").glob("gnome_*.jpg"))
        if selfies:
            blog_generator.selected_selfie = random.choice(selfies)  # Store the selection
            images[blog_generator.selected_selfie.name] = blog_generator.selected_selfie

        # Create HTML post
        print("Creating HTML post...")
        html_content = blog_generator.create_html_post(actual_day_dir, content)
        if not html_content:
            print("Failed to create HTML content")
            return False

        # Add street views if available
        street_view_dir = actual_day_dir / "street_view"
        if street_view_dir.exists():
            for i in range(1, 4):
                view_path = street_view_dir / f"view_{i}.jpg"
                if view_path.exists():
                    images[f'street_view_{i}.jpg'] = view_path

        # Get POI information
        try:
            with open(actual_day_dir / 'places_of_interest.json', 'r') as f:
                pois = json.load(f)
        except FileNotFoundError:
            print("No places of interest found")
            pois = []

        # Add POI photos - match POI order from places_of_interest.json
        places_dir = actual_day_dir / "places"
        # Process POI photos with selfies
        if places_dir.exists() and pois:
            print("Processing POI photos with selfies...")
            for i, poi in enumerate(pois[:3], 1):
                poi_dir = places_dir / poi['name'].replace('/', '_')
                if poi_dir.exists():
                    photo_path = next(poi_dir.glob("photo_1.jpg"), None)
                    if photo_path:
                        # Generate selfie version
                        selfie_output_path = poi_dir / f"selfie_photo_{i}.png"
                        combined_path = selfie_generator.combine_with_background(
                            background_path=photo_path,
                            output_path=selfie_output_path
                        )
                        if combined_path:
                            print(f"Created selfie photo for POI {i}: {poi['name']}")
                            images[f'poi_{i}_photo_1.jpg'] = combined_path
                        else:
                            print(f"Failed to create selfie photo for POI {i}: {poi['name']}")
                            images[f'poi_{i}_photo_1.jpg'] = photo_path
                    else:
                        print(f"No photo found for POI {i}: {poi['name']}")
                else:
                    print(f"No directory found for POI {i}: {poi['name']}")

        # Print debug information about images
        print("\nImages to be uploaded:")
        for img_key, img_path in images.items():
            print(f"{img_key}: {img_path}")

        # Publish post
        print("\nPublishing post...")
        city_name = position_info['current_city']['name']
        title = f"{city_name}, {country}"
        success = blog_generator.publish_post(title, html_content, images, city_name)

        if success:
            print(f"Blog post about {city_name} published successfully!")
            # Optional: Save a local copy of the HTML
            local_html_path = actual_day_dir / "blog_post.html"
            with open(local_html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"Local copy saved to: {local_html_path}")
        else:
            print("Failed to publish blog post")

        return success

    except Exception as e:
        print(f"Error creating daily blog post: {e}")
        print(f"Attempted directory: {day_dir}")
        print(f"Actual directory: {actual_day_dir if 'actual_day_dir' in locals() else 'unknown'}")
        return False
