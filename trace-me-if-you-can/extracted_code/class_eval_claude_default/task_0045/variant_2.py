from PIL import Image, ImageEnhance, ImageChops
from contextlib import contextmanager


class ImageProcessor:
    def __init__(self):
        self.image = None

    def load_image(self, image_path):
        try:
            self.image = Image.open(image_path)
        except (FileNotFoundError, IOError):
            self.image = None
            raise

    def save_image(self, save_path):
        with self._ensure_image():
            self.image.save(save_path)

    def resize_image(self, width, height):
        with self._ensure_image():
            self.image = self.image.resize((width, height))

    def rotate_image(self, degrees):
        with self._ensure_image():
            self.image = self.image.rotate(degrees)

    def adjust_brightness(self, factor):
        with self._ensure_image():
            brightness_enhancer = ImageEnhance.Brightness(self.image)
            self.image = brightness_enhancer.enhance(factor)

    @contextmanager
    def _ensure_image(self):
        if not self.image:
            raise ValueError("No image loaded")
        yield
