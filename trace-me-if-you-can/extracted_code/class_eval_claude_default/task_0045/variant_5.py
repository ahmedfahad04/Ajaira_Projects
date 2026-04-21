from PIL import Image, ImageEnhance, ImageChops
from enum import Enum


class ProcessorState(Enum):
    EMPTY = "empty"
    LOADED = "loaded"


class ImageProcessor:
    def __init__(self):
        self._image_data = None
        self._state = ProcessorState.EMPTY

    def load_image(self, image_path):
        self._image_data = Image.open(image_path)
        self._state = ProcessorState.LOADED

    def save_image(self, save_path):
        self._execute_if_ready(lambda: self._image_data.save(save_path))

    def resize_image(self, width, height):
        self._execute_if_ready(
            lambda: setattr(self, '_image_data', self._image_data.resize((width, height)))
        )

    def rotate_image(self, degrees):
        self._execute_if_ready(
            lambda: setattr(self, '_image_data', self._image_data.rotate(degrees))
        )

    def adjust_brightness(self, factor):
        def brightness_operation():
            enhancer = ImageEnhance.Brightness(self._image_data)
            self._image_data = enhancer.enhance(factor)
        
        self._execute_if_ready(brightness_operation)

    def _execute_if_ready(self, operation):
        if self._state == ProcessorState.LOADED and self._image_data:
            operation()

    @property
    def image(self):
        return self._image_data

    @image.setter
    def image(self, value):
        self._image_data = value
        self._state = ProcessorState.LOADED if value else ProcessorState.EMPTY
