from PIL import Image, ImageEnhance, ImageResizing


class ImageHandler:
    def __init__(self):
        self.image_data = None

    def import_image(self, file_path):
        self.image_data = Image.open(file_path)

    def export_image(self, output_path):
        if self.image_data:
            self.image_data.save(output_path)

    def change_image_dimensions(self, x, y):
        if self.image_data:
            self.image_data = ImageResizing.resize(self.image_data, (x, y))

    def adjust_image_orientation(self, angle):
        if self.image_data:
            self.image_data = self.image_data.rotate(angle)

    def adjust_image_brightness(self, intensity):
        if self.image_data:
            enhancer = ImageEnhance.Brightness(self.image_data)
            self.image_data = enhancer.enhance(intensity)
