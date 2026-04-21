from PIL import Image, ImageEnhance, ImageChops
from functools import wraps


def requires_image(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'image') or self.image is None:
            return None
        return func(self, *args, **kwargs)
    return wrapper


def image_operation(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if result is not None:
            self.image = result
        return self
    return wrapper


class ImageProcessor:
    def __init__(self):
        self.image = None

    def load_image(self, image_path):
        self.image = Image.open(image_path)

    @requires_image
    def save_image(self, save_path):
        self.image.save(save_path)

    @requires_image
    @image_operation
    def resize_image(self, width, height):
        return self.image.resize((width, height))

    @requires_image
    @image_operation
    def rotate_image(self, degrees):
        return self.image.rotate(degrees)

    @requires_image
    @image_operation
    def adjust_brightness(self, factor):
        brightness_tool = ImageEnhance.Brightness(self.image)
        return brightness_tool.enhance(factor)
