from PIL import Image as PILImage
import random
from pathlib import Path
from typing import Optional, Tuple
from io import BytesIO

from image_compressor import ImageCompressor


class GnomeSelfieGenerator:
    def __init__(self, selfies_dir: Path):
        """Initialize with directory containing gnome selfie portraits."""
        self.selfies_dir = selfies_dir
        self.selfie_paths = list(selfies_dir.glob("selfie_*.png"))
        if not self.selfie_paths:
            raise ValueError(f"No selfie images found in {selfies_dir}")

    def get_random_selfie_with_alignment(self) -> Tuple[Path, str]:
        """Get a random selfie and determine its alignment from filename."""
        selfie_path = random.choice(self.selfie_paths)
        alignment = selfie_path.stem.split('_')[-1]
        if alignment not in ['left', 'right', 'center']:
            alignment = 'right'
        return selfie_path, alignment

    def resize_image(self, image, max_size: int = 1920):
        """Resize image while maintaining aspect ratio."""
        ratio = min(max_size / image.size[0], max_size / image.size[1])
        if ratio < 1:
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            return image.resize(new_size)
        return image

    def ensure_landscape_orientation(self, image) -> PILImage:
        """Convert vertical image to landscape by cropping the top portion."""
        width, height = image.size
        if height > width:
            crop_box = (0, 0, width, width)
            return image.crop(crop_box)
        return image

    def combine_with_background(self, background_path: Path, output_path: Path) -> Optional[Path]:
        """Combine background image with a random gnome selfie."""
        try:
            # Get random selfie and its alignment
            portrait_path, gnome_alignment = self.get_random_selfie_with_alignment()

            # Load and compress background image
            with PILImage.open(background_path) as background:
                if background.mode in ('RGBA', 'P'):
                    background = background.convert('RGB')

                background = self.ensure_landscape_orientation(background)
                background = self.resize_image(background)

                # Load portrait
                with PILImage.open(portrait_path) as portrait:
                    if portrait.mode != 'RGBA':
                        portrait = portrait.convert('RGBA')

                    # Calculate dimensions
                    bg_width, bg_height = background.size
                    portrait_width, portrait_height = portrait.size

                    # Calculate scaling
                    random_scale_factor = random.uniform(0.15, 0.3)
                    scale_factor = (bg_width / portrait_width) * random_scale_factor

                    # Calculate new dimensions
                    new_width = int(portrait_width * scale_factor)
                    new_height = int(portrait_height * scale_factor)

                    # Adjust height if needed
                    min_height = bg_height * 0.5
                    if new_height < min_height:
                        scale_factor = min_height / portrait_height
                        new_width = int(portrait_width * scale_factor)
                        new_height = int(portrait_height * scale_factor)

                    max_height = bg_height * 0.8
                    if new_height > max_height:
                        scale_factor = max_height / portrait_height
                        new_width = int(portrait_width * scale_factor)
                        new_height = int(portrait_height * scale_factor)

                    # Resize portrait
                    portrait = portrait.resize((new_width, new_height))

                    # Calculate position
                    margin = random.randint(10, 30)
                    if gnome_alignment == 'right':
                        x = bg_width - new_width + margin
                    elif gnome_alignment == 'left':
                        x = -margin
                    else:  # center
                        x = (bg_width - new_width) // 2 + random.randint(-150, 150)

                    # Calculate vertical position
                    min_y = int(bg_height - (new_height * 0.9))
                    max_y = int(bg_height - (new_height * 0.6))
                    y = random.randint(min_y, max_y)

                    # Create composite
                    background.paste(portrait, (x, y), portrait)

                    # Save with ImageCompressor
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    ImageCompressor.compress_image(
                        background,  # Now accepts PIL Image directly
                        str(output_path),
                        max_size_kb=500,
                        quality=85,
                        max_width=1920
                    )

            return output_path

        except Exception as e:
            print(f"Error combining images: {e}")
            return None


