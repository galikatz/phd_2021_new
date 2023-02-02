import tensorflow as tf
from PIL import Image # used for loading images
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    path = "/Users/gali.k/phd/phd_2021/stimuli/equate_1/images_1/incong86_7_6_350.jpg"
    img = Image.open(path)
    frac = 0.70
    left = img.size[0] * ((1 - frac) / 2)
    upper = img.size[1] * ((1 - frac) / 2)
    right = img.size[0] - ((1 - frac) / 2) * img.size[0]
    bottom = img.size[1] - ((1 - frac) / 2) * img.size[1]
    cropped_img = img.crop((left, upper, right, bottom))
    cropped_img.show()
    rgb_image = cropped_img.resize((100, 100), Image.ANTIALIAS).show()

