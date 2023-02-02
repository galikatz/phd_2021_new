from PIL import Image
import os
# Giving The Original image Directory
# Specified
old_name = "/Users/gali.k/phd/phd_2021/stimuli/equate_3/images_13/cong71_5_7_46.jpg"
Original_Image = Image.open(old_name)

# Rotate Image By 180 Degree
rotated_image1 = Original_Image.rotate(180)
new_name = "/Users/gali.k/phd/phd_2021/stimuli/equate_3/images_13/cong71_5_7_46.jpg"
rotated_image1.save(new_name)

os.remove(old_name)

# Original_Image.show()
# rotated_image1.show()
