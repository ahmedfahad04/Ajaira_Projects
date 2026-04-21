from PIL import Image, ImageEnhance


class ImageTool:
    def __init__(self):
        self.image_content = None

    def load_img(self, path):
        self.image_content = Image.open(path)

    def save_img(self, path):
        if self.image_content:
            self.image_content.save(path)

    def resize_img(self, w, h):
        if self.image_content:
            self.image_content = self.image_content.resize((w, h))

    def rotate_img(self, deg):
        if self.image_content:
            self.image_content = self.image_content.rotate(deg)

    def enhance_img_brightness(self, factor):
        if self.image_content:
            enhancer = ImageEnhance.Brightness(self.image_content)
            self.image_content = enhancer.enhance(factor)
