from PIL import Image, ImageEnhance


class ImageManipulator:
    def __init__(self):
        self.my_image = None

    def load_picture(self, image_file):
        self.my_image = Image.open(image_file)

    def store_picture(self, destination):
        if self.my_image:
            self.my_image.save(destination)

    def adjust_size(self, new_width, new_height):
        if self.my_image:
            self.my_image = self.my_image.resize((new_width, new_height))

    def turn_image(self, degree):
        if self.my_image:
            self.my_image = self.my_image.rotate(degree)

    def tweak_brightness(self, value):
        if self.my_image:
            enhancer = ImageEnhance.Brightness(self.my_image)
            self.my_image = enhancer.enhance(value)
