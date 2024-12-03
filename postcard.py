from datetime import datetime
import requests
from pathlib import Path
from typing import Optional, Tuple
from openai import OpenAI
import random
from PIL import Image, ImageDraw, ImageFont

from image_compressor import ImageCompressor


class GnomeImageGenerator:
    def __init__(self, openai_key: str):
        """Initialize the image generator with OpenAI API key."""
        self.client = OpenAI(api_key=openai_key)
        self.image_settings = {
            "model": "dall-e-3",
            "size": "1792x1024",  # Landscape format
            "quality": "standard",
            "n": 1,
        }
        self.haiku_context = """You are a traveling garden gnome who writes haikus about your daily adventures.
        Create a haiku that captures the essence of today's journey and experiences.
        Make it whimsical and fun, but also thoughtful. Include local elements when possible."""

        # Add compression settings
        self.compressor = ImageCompressor()

    def generate_haiku(self, country: str, city_name: str, date: str) -> str:
        """Generate a haiku based on the day's experiences."""
        try:
            prompt = f"""Write a haiku about my visit to {city_name} in {country}.

            The haiku should capture the essence of the place and season. Today is {date}.
            Format it with line breaks using \n."""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.haiku_context},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating haiku: {e}")
            return "Mountain winds whisper\nTraveling gnome remembers\nAdventures ahead"  # Fallback haiku

    def _generate_and_save_image(self, prompt: str, save_path: Path, image_type: str) -> Optional[Path]:
        """Generate and save a compressed image with the given prompt."""
        try:
            response = self.client.images.generate(
                **self.image_settings,
                prompt=prompt
            )

            img_response = requests.get(response.data[0].url)
            img_response.raise_for_status()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = save_path / f"gnome_{image_type}_{timestamp}.png"

            # Create directory if it doesn't exist
            image_path.parent.mkdir(parents=True, exist_ok=True)

            # Compress image before saving
            compressed_bytes = ImageCompressor.compress_image_to_bytes(
                img_response.content,
                max_size_kb=800,  # Higher quality for AI-generated images
                quality=85,
                max_width=1024
            )

            with open(image_path, 'wb') as f:
                f.write(compressed_bytes)

            print(f"Compressed {image_type} generated and saved to: {image_path}")
            return image_path

        except Exception as e:
            print(f"Error generating {image_type}: {e}")
            return None

    @staticmethod
    def _get_text_dimensions(text_string: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
        """Calculate dimensions of text string with given font."""
        if not text_string:
            return (0, 80)

        ascent, descent = font.getmetrics()
        text_width = font.getmask(text_string).getbbox()[2]
        text_height = font.getmask(text_string).getbbox()[3] + descent
        return (text_width, text_height)

    def _create_text_image(self,
                           bg_image_path: str,
                           output_path: str,
                           text: str,
                           font_path: str,
                           text_color: Tuple[int, int, int] = (0, 0, 0),
                           left_padding: int = 20) -> str:
        """Create a compressed image with text overlay."""
        # Load and compress the background image first
        compressed_bg = ImageCompressor.compress_image(
            bg_image_path,
            max_size_kb=800,
            quality=85,
            max_width=1024
        )

        # Load the compressed background image
        image = Image.open(compressed_bg)

        # Rest of the text drawing code remains the same
        font_size = 50
        font = ImageFont.truetype(font_path, font_size)
        lines = text.split('\n')
        draw = ImageDraw.Draw(image)
        total_height = sum(GnomeImageGenerator._get_text_dimensions(line, font)[1] for line in lines)
        y_offset = (image.size[1] - total_height) / 2

        for line in lines:
            draw.text(
                (left_padding + 100, y_offset + 90),
                line,
                fill=text_color,
                font=font
            )
            y_offset += GnomeImageGenerator._get_text_dimensions(line, font)[1]

        # Save the final image with compression
        ImageCompressor.compress_image(
            image,
            output_path,
            max_size_kb=800,
            quality=85
        )
        return output_path

    def generate_postcard(self, country: str, day_path: Path) -> Optional[Path]:
        """Generate a vintage postcard with a gnome."""
        prompt = f"""Draw an old postcard with stamps featuring a retro looking garden gnome in {country}. 
        Use landscape layout. The style should be vintage and weathered, with decorative borders typical 
        of postcards from the early 20th century. Include postal marks and at least one stamp that's 
        characteristic of {country}. The gnome should be interacting with something iconic from {country}."""
        return self._generate_and_save_image(prompt, day_path, "postcard")

    def generate_map(self, country: str, day_path: Path) -> Optional[Path]:
        """Generate a whimsical map of the country."""
        prompt = f"""Create a playful and imaginative doodle of a map of {country}, as if it were drawn by a child. 
        The map should have a naive, unpolished style with simple shapes, uneven lines, and colors used in {country}'s 
        flag. 
        Include fun and exaggerated elements like crooked mountains, smiley-faced suns, stick-figure towns, and 
        wavy rivers that might not follow realistic geography. Add whimsical icons like oversized flowers, tiny houses, 
        and fantastical creatures scattered across the landscape."""
        return self._generate_and_save_image(prompt, day_path, "postcard")

    def generate_haiku_postcard(self,
                                country: str,
                                city_name: str,
                                day_path: Path,
                                bg_image_path: str,
                                font_path: str) -> Optional[Path]:
        """Generate a compressed postcard with haiku text overlay."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            haiku = self.generate_haiku(country, city_name, timestamp)
            print(haiku)

            postcard_text = f"{haiku}\n\nYours truly,\nGarden Gnome Oliver\n from {city_name} with love"
            output_path = day_path / "postcard.png"
            print(output_path)

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Create and compress the image
            compressed_path = self._create_text_image(
                bg_image_path=bg_image_path,
                output_path=str(output_path),
                text=postcard_text,
                font_path=font_path
            )
            print(f"Compressed postcard created: {compressed_path}")

            return Path(compressed_path)

        except Exception as e:
            print(f"Error generating haiku postcard: {e}")
            return None

    def generate_random_image(self,
                              country: str,
                              day_path: Path,
                              city_name: str = "") -> Optional[Path]:
        """Randomly select and generate either a regular postcard, map, or haiku postcard."""
        generators = [
            (self.generate_postcard, [country, day_path]),
            (self.generate_map, [country, day_path]),
            (self.generate_haiku_postcard, [country, city_name, day_path,
                                            "selfies/postcard_background.jpg", "selfies/handwriting.ttf"]),
        ]

        chosen_generator, args = random.choice(generators)
        return chosen_generator(*args)

