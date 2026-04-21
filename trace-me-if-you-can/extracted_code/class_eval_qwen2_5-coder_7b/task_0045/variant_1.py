from PIL import Image, ImageEnhance, ImageOps


class ImageProcessor:
    def __init__(self):
        self.current_image = None

    def read_image(self, path):
        self.current_image = Image.open(path)

    def export_image(self, path):
        if self.current_image:
            self.current_image.save(path)

    def change_dimensions(self, width, height):
        if self.current_image:
            self.current_image = self.current_image.resize((width, height))

    def modify_orientation(self, angle):
        if self.current_image:
            self.current_image = self.current_image.rotate(angle)

    def modify_brightness(self, increment):
        if self.current_image:
            enhancer = ImageEnhance.Brightness(self.current_image)
            self.current_image = enhancer.enhance(increment)
