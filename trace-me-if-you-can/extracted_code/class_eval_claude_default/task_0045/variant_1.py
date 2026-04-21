from PIL import Image, ImageEnhance, ImageChops
from functools import reduce


class ImageProcessor:
    def __init__(self):
        self._image = None
        self._operations = []

    def load_image(self, image_path):
        self._image = Image.open(image_path)
        return self

    def save_image(self, save_path):
        if self._image:
            self._image.save(save_path)
        return self

    def resize_image(self, width, height):
        self._operations.append(lambda img: img.resize((width, height)) if img else img)
        self._apply_operations()
        return self

    def rotate_image(self, degrees):
        self._operations.append(lambda img: img.rotate(degrees) if img else img)
        self._apply_operations()
        return self

    def adjust_brightness(self, factor):
        self._operations.append(lambda img: ImageEnhance.Brightness(img).enhance(factor) if img else img)
        self._apply_operations()
        return self

    def _apply_operations(self):
        if self._image and self._operations:
            self._image = reduce(lambda img, op: op(img), self._operations, self._image)
            self._operations.clear()

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        self._image = value
