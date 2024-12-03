from PIL import Image
from pathlib import Path
import io


class ImageCompressor:
    @staticmethod
    def compress_image(input_path_or_image: Path | str | Image.Image, output_path: Path | str = None,
                       max_size_kb: int = 500, quality: int = 85,
                       max_width: int = 1024) -> Path:
        """
        Compress an image file to reduce its size while maintaining reasonable quality.
        """
        try:
            # Handle PIL Image object
            if isinstance(input_path_or_image, Image.Image):
                img = input_path_or_image
                if output_path is None:
                    raise ValueError("output_path must be provided when input is a PIL Image")
            else:
                # Convert paths to strings
                input_path = str(input_path_or_image)
                img = Image.open(input_path)
                if output_path is None:
                    output_path = input_path

            output_path = str(output_path)

            # Convert to RGB if needed
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            # Simple resize if width exceeds maximum
            if img.size[0] > max_width:
                ratio = max_width / img.size[0]
                new_height = int(img.size[1] * ratio)
                img = img.resize((max_width, new_height))  # Using default resize method

            # Save with compression
            img.save(output_path, 'JPEG', quality=quality, optimize=True)

            # Reduce quality if file is too large
            while Path(output_path).stat().st_size > max_size_kb * 1024 and quality > 20:
                quality -= 5
                img.save(output_path, 'JPEG', quality=quality, optimize=True)

            return Path(output_path)

        except Exception as e:
            print(f"Error compressing image: {e}")
            if isinstance(input_path_or_image, (str, Path)):
                return Path(input_path_or_image)
            raise

    @staticmethod
    def compress_image_to_bytes(image_bytes: bytes, max_size_kb: int = 500,
                                quality: int = 85, max_width: int = 1024) -> bytes:
        """
        Compress image bytes without saving to disk.
        """
        try:
            # Create PIL Image from bytes
            img = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            # Simple resize if needed
            if img.size[0] > max_width:
                ratio = max_width / img.size[0]
                new_height = int(img.size[1] * ratio)
                img = img.resize((max_width, new_height))

            # Save to bytes buffer with compression
            buffer = io.BytesIO()
            img.save(buffer, 'JPEG', quality=quality, optimize=True)

            # Reduce quality if needed
            while buffer.tell() > max_size_kb * 1024 and quality > 20:
                quality -= 5
                buffer = io.BytesIO()
                img.save(buffer, 'JPEG', quality=quality, optimize=True)

            return buffer.getvalue()

        except Exception as e:
            print(f"Error compressing image bytes: {e}")
            return image_bytes
