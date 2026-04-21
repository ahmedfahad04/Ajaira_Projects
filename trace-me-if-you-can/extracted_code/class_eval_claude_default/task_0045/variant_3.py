from PIL import Image, ImageEnhance, ImageChops
from abc import ABC, abstractmethod


class ImageCommand(ABC):
    @abstractmethod
    def execute(self, image):
        pass


class ResizeCommand(ImageCommand):
    def __init__(self, width, height):
        self.dimensions = (width, height)
    
    def execute(self, image):
        return image.resize(self.dimensions) if image else image


class RotateCommand(ImageCommand):
    def __init__(self, degrees):
        self.degrees = degrees
    
    def execute(self, image):
        return image.rotate(self.degrees) if image else image


class BrightnessCommand(ImageCommand):
    def __init__(self, factor):
        self.factor = factor
    
    def execute(self, image):
        if image:
            return ImageEnhance.Brightness(image).enhance(self.factor)
        return image


class ImageProcessor:
    def __init__(self):
        self.image = None

    def load_image(self, image_path):
        self.image = Image.open(image_path)

    def save_image(self, save_path):
        if self.image:
            self.image.save(save_path)

    def resize_image(self, width, height):
        command = ResizeCommand(width, height)
        self.image = command.execute(self.image)

    def rotate_image(self, degrees):
        command = RotateCommand(degrees)
        self.image = command.execute(self.image)

    def adjust_brightness(self, factor):
        command = BrightnessCommand(factor)
        self.image = command.execute(self.image)
