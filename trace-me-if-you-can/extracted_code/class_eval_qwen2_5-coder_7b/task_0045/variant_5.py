from PIL import Image, ImageEnhance


class PictureProcessor:
    def __init__(self):
        self.picture = None

    def open_picture(self, file_path):
        self.picture = Image.open(file_path)

    def save_picture(self, file_path):
        if self.picture:
            self.picture.save(file_path)

    def resize_picture(self, width, height):
        if self.picture:
            self.picture = self.picture.resize((width, height))

    def rotate_picture(self, degrees):
        if self.picture:
            self.picture = self.picture.rotate(degrees)

    def adjust_picture_brightness(self, factor):
        if self.picture:
            enhancer = ImageEnhance.Brightness(self.picture)
            self.picture = enhancer.enhance(factor)
