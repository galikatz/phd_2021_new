if __name__ == '__main__':
    from PIL import Image

    # Open the image
    im = Image.open("/Users/gali.k/phd/phd_2021/stimuli/equate_1/images_1/cong50_5_10_34.jpg")

    # Get the width and height of the image
    width, height = im.size

    # Crop the image into two vertical halves
    left = im.crop((0, 0, width // 2, height))
    right = im.crop((width // 2, 0, width, height))

    # Save the two cropped images
    left.save("left.png")
    right.save("right.jpg")
