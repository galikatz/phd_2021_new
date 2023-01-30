from PIL import Image
import os
# Giving The Original image Directory
# Specified
old_name = "/Users/gali.k/phd/phd_2021/stimuli_fix/equate_1/rotate/cong50_5_10_12.jpg"
Original_Image = Image.open(old_name)

# Rotate Image By 180 Degree
rotated_image1 = Original_Image.rotate(180)
new_name = "/Users/gali.k/phd/phd_2021/stimuli_fix/equate_1/rotate/cong50_10_5_12.png"
rotated_image1.save(new_name)

os.remove(old_name)

# Original_Image.show()
# rotated_image1.show()
