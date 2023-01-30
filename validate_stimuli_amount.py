import os
import glob
import shutil
import pathlib
from PIL import Image
from evolution_utils import RATIOS, RATIO_NUMBERS
GEN_REQUIRED_FILES = 1200
CONG_INGCONG_REQUIRED_FILES = 600
PER_RATIO_REQUIRED_FILES = 100
from pathlib import Path
import random



def moving_stimuli_to_temp_folder(source_dir, num_of_files_to_transfer, target_dir, the_files_per_ratio):
    if not pathlib.Path(target_dir).exists():
        os.mkdir(target_dir)
    i = 0
    while i < num_of_files_to_transfer:
        file = the_files_per_ratio[i].split(os.sep)[-1]
        #os.rename(source_dir + file, target_dir + file)
        #os.replace(source_dir + file, target_dir + file)
        shutil.move(source_dir + file, target_dir + file)
        i = i + 1


def validate_files_per_ratio(path_to_gen_folder, files_per_ratio, congruency) -> (bool, str):
    num_files_per_ratio = len(files_per_ratio)
    if num_files_per_ratio < PER_RATIO_REQUIRED_FILES:
        msg = f"{'cong' if congruency else 'incong'} stimuli amount per ratio is smaller than needed: {num_files_per_ratio}/{PER_RATIO_REQUIRED_FILES}"
        return False, msg
    elif num_files_per_ratio > PER_RATIO_REQUIRED_FILES:
        msg = f"{'cong' if congruency else 'incong'} stimuli amount per ratio is larger than needed: {num_files_per_ratio}/{PER_RATIO_REQUIRED_FILES}, moving to temp_folder"
        print(msg)
        num_of_files_to_transfer = num_files_per_ratio - PER_RATIO_REQUIRED_FILES
        target_dir = path_to_gen_folder.replace("images_", "temp_")
        moving_stimuli_to_temp_folder(path_to_gen_folder, num_of_files_to_transfer, target_dir, files_per_ratio)
        return True, f"moved {num_of_files_to_transfer} files"
    else:
        msg = f"{'cong' if congruency else 'incong'} stimuli amount per ratio just right: {num_files_per_ratio}/{PER_RATIO_REQUIRED_FILES}"
        return True, msg


def validate_right_left_sides(ratio, path_to_gen_folder, files_per_ratio, congruency, is_colors) -> (bool, str):
    left_list, right_list = validate_sides(files_per_ratio)
    left = len(left_list)
    right = len(right_list)
    # if left == right:
    #     print(f'Ratio: {ratio} isCongruent: {congruency} left: {left} and right: {right} sides are balanced')
    if left != right:
        number_of_required_stimuli_per_side = (left + right)/2
        #print(f"ratio: {ratio} isCongruent: {congruency} stimuli with big number in left side: {left} and number of stimuli with big number in right side: {right}, rotating image and renaming...")
        left_stimuli_to_fix = left - number_of_required_stimuli_per_side
        rename_and_rotate(ratio, left_stimuli_to_fix, left_list, "left", path_to_gen_folder, is_colors)
        right_stimuli_to_fix = right - number_of_required_stimuli_per_side
        rename_and_rotate(ratio, right_stimuli_to_fix, right_list, "right", path_to_gen_folder, is_colors)
    return True, f"Ratio: {ratio} isCongruent: {congruency} all is balanced"


def validate_sides(files_per_ratio):
    left_num_is_bigger_list = []
    right_num_is_bigger_list = []
    for file_name in files_per_ratio:
        description = file_name[file_name.rindex(os.sep) + 1:file_name.index('.jpg')]
        labels = description.split('_')
        left_num = int(labels[1])
        right_num = int(labels[2])
        if left_num > right_num:
            left_num_is_bigger_list.append(file_name)
        else:
            right_num_is_bigger_list.append(file_name)
    return left_num_is_bigger_list, right_num_is_bigger_list


def rename_and_rotate(ratio, stimuli_to_fix, inventory_list, side, path_to_gen_folder, is_colors):
    if stimuli_to_fix < 0:
        return

    #print(f"ratio :{ratio} moving {stimuli_to_fix} stimuli from side {side}")
    to_delete = []
    files_to_fix = inventory_list[:int(stimuli_to_fix)]
    for old_name in files_to_fix:
        # prepare new name
        description = old_name[old_name.rindex(os.sep) + 1:old_name.index('.jpg')]
        labels = description.split('_')
        if is_colors: #the magenta and cyan are also rotate to different side
            new_name = path_to_gen_folder + labels[0] + "_" + labels[2] + "_" + labels[1] + "_" + labels[3] + "_" + labels[4] + "_" + labels[7] + "_" + labels[6] + "_" + labels[5] + ".jpg"
        else:
            new_name = path_to_gen_folder + labels[0] + "_" + labels[2] + "_" + labels[1] + "_" + ''.join(labels[3:]) + ".jpg"
        # rotate
        original_image = Image.open(old_name)
        rotated_image = original_image.rotate(180)
        # save new image
        if Path(new_name).exists():
            labels[3] = str(random.randint(1000000, 2000000))
            if is_colors:
                new_name = path_to_gen_folder + labels[0] + "_" + labels[2] + "_" + labels[1] + "_" + labels[
                    3] + "_" + labels[4] + "_" + labels[7] + "_" + labels[6] + "_" + labels[5] + ".jpg"
            else:
                new_name = path_to_gen_folder + labels[0] + "_" + labels[2] + "_" + labels[1] + "_" + ''.join(labels[3:]) +".jpg"
        rotated_image.save(new_name)
        to_delete.append(old_name)
    if len(to_delete) > 0:
        #print(f"ratio: {ratio} deleting {len(to_delete)} files")
        for file_to_delete in to_delete:
            os.remove(file_to_delete)


def rename_and_recolor(ratio, stimuli_to_fix, color_at_side_inventory_list, color, gen_folder):
    if stimuli_to_fix < 0:
        return
    print(f"ratio :{ratio} recoloring {stimuli_to_fix} stimuli from color {color}")
    files_to_fix = color_at_side_inventory_list[:int(stimuli_to_fix)]
    for old_name in files_to_fix:
        original_image = Image.open(old_name)





def print_stimuli_stat(gen_folder):
    print("****** Stats ******")
    for ratio in RATIOS:
        incong_files_per_ratio = glob.glob(gen_folder + os.sep + f"incong{ratio}*.jpg")
        cong_files_per_ratio = glob.glob(gen_folder + os.sep + f"cong{ratio}*.jpg")
        cong_left_list, cong_right_list = validate_sides(cong_files_per_ratio)
        incong_left_list, incong_right_list = validate_sides(incong_files_per_ratio)
        print(f"ratio {ratio}: congruent: {str(len(cong_files_per_ratio))}, [cong_left: {str(len(cong_left_list))}, cong_right: {str(len(cong_right_list))}]"
              f" incongruent: {str(len(incong_files_per_ratio))} [incong_left: {str(len(incong_left_list))}, incong_right: {str(len(incong_right_list))}]")
    print("****** End ******")


def validate_congruity_per_ratio(gen_folder):
    for ratio in RATIOS:  # cong86_6_7_2980_r_c_l_m.jpg
        incong_files_per_ratio = glob.glob(gen_folder + os.sep + f"incong{ratio}*.jpg")
        cong_files_per_ratio = glob.glob(gen_folder + os.sep + f"cong{ratio}*.jpg")
        is_valid, msg = validate_files_per_ratio(gen_folder + os.sep, incong_files_per_ratio, False)
        if not is_valid:
            return False, f"Checking {gen_folder}: {msg}"
        is_valid, msg = validate_files_per_ratio(gen_folder + os.sep, cong_files_per_ratio, True)
        if not is_valid:
            return False, f"Checking {gen_folder}: {msg}"
    return True, 'balanced'


def validate_big_number_equally_found_in_left_and_right_sides(gen_folder, is_colors):
    for ratio in RATIOS:  # cong86_6_7_2980_r_c_l_m.jpg
        incong_files_per_ratio = glob.glob(gen_folder + os.sep + f"incong{ratio}*.jpg")
        cong_files_per_ratio = glob.glob(gen_folder + os.sep + f"cong{ratio}*.jpg")
        is_valid, msg = validate_right_left_sides(ratio=ratio, path_to_gen_folder=gen_folder + os.sep, files_per_ratio=incong_files_per_ratio, congruency=False, is_colors=is_colors)
        if not is_valid:
            return False, f"Ratio {ratio} incongruent: {msg}"
        is_valid, msg = validate_right_left_sides(ratio=ratio, path_to_gen_folder=gen_folder + os.sep, files_per_ratio=cong_files_per_ratio, congruency=True, is_colors=is_colors)
        if not is_valid:
            return False, f"Ratio {ratio} congruent: {msg}"
    return True, 'balanced'


def print_stat_colors(gen_folder) -> (bool, str):
    for ratio in RATIOS:
        #congruent
        cong_files_per_ratio = glob.glob(gen_folder + os.sep + f"cong{ratio}*.jpg")
        cong_left_list, cong_right_list = validate_sides(cong_files_per_ratio)
        # check left list
        cyan_at_left_list, magenta_at_left_list = count_colors(cong_left_list)
        print(f"Ratio {ratio} isCongruent: True, Big number at left: from {str(len(cong_left_list))} big number in left side files: [c={str(len(cyan_at_left_list))}, m={str(len(magenta_at_left_list))}]")

        # check right list
        cyan_at_right_list, magenta_at_right_list = count_colors(cong_right_list)
        print(f"Ratio {ratio}: isCongruent: True, Big number at right: from {str(len(cong_right_list))} big number in right side files: [c={str(len(cyan_at_right_list))}, m={str(len(magenta_at_right_list))}]")

        #incongruent:
        incong_files_per_ratio = glob.glob(gen_folder + os.sep + f"incong{ratio}*.jpg")
        incong_left_list, incong_right_list = validate_sides(incong_files_per_ratio)
        cyan_at_left_list, magenta_at_left_list = count_colors(incong_left_list)
        print( f"Ratio {ratio} isCongruent: False, Big number at left: from {str(len(incong_left_list))} big number in left side files: [c={str(len(cyan_at_left_list))}, m={str(len(magenta_at_left_list))}]")

        # check right list
        cyan_at_right_list, magenta_at_right_list = count_colors(incong_right_list)
        print( f"Ratio {ratio}: isCongruent: False, Big number at right: from {str(len(incong_right_list))} big number in right side files: [c={str(len(cyan_at_right_list))}, m={str(len(magenta_at_right_list))}]")


def count_colors(one_side_inventory_list):
    cyan_one_side = []
    magenta_one_side = []
    for file_name in one_side_inventory_list:
        description = file_name[file_name.rindex(os.sep) + 1:file_name.index('.jpg')]
        labels = description.split('_')
        if labels[5] == 'c':
            cyan_one_side.append(file_name)
        else:
            magenta_one_side.append(file_name)

    return cyan_one_side, magenta_one_side


def validate_colors(gen_folder) -> (bool, str):
    for ratio in RATIOS:
        congruency = True
        cong_files_per_ratio = glob.glob(gen_folder + os.sep + f"cong{ratio}*.jpg")
        cong_left_list, cong_right_list = validate_sides(cong_files_per_ratio)
        # check left list
        cyan_at_left_list, magenta_at_left_list = count_colors(cong_left_list)
        cyan_left = len(cyan_at_left_list)
        magenta_left = len(magenta_at_left_list)
        if cyan_left == magenta_left:
            print(f'Ratio: {ratio} isCongruent: {congruency} cyan left: {cyan_left} and megenta left: {magenta_left} are balanced')
        if cyan_left != magenta_left:
            number_of_required_stimuli_per_side = (cyan_left + magenta_left) / 2
            print(f"ratio: {ratio} isCongruent: {congruency} stimuli with cyan left side: {cyan_left} and magenta left: {magenta_left}, coloring and renaming...")
            left_stimuli_to_fix = cyan_left - number_of_required_stimuli_per_side
            rename_and_recolor(ratio, left_stimuli_to_fix, cyan_at_left_list, "cyan", gen_folder)
            right_stimuli_to_fix = magenta_left - number_of_required_stimuli_per_side
            rename_and_recolor(ratio, right_stimuli_to_fix, magenta_at_left_list, "magenta", gen_folder)
        return True, f"Ratio: {ratio} isCongruent: {congruency} colors are all balanced"


    is_valid = True
    msg = ''
    return is_valid, msg


def validate_files(path, is_colors) -> (bool, str):
    is_valid = False
    num_of_gen_folders = glob.glob(path + os.sep + f"images_*")
    temp = []
    for file in num_of_gen_folders:
        if os.path.isdir(file):
            temp.append(file)
    temp.sort()
    num_of_gen_folders = temp

    for gen_folder in num_of_gen_folders:
        num_images_in_single_gen = len(os.listdir(gen_folder))
        if num_images_in_single_gen < GEN_REQUIRED_FILES:
            msg = f"Stimuli amount in a single gen is {'valid' if is_valid else 'invalid'}, stimuli in gen {gen_folder} is: {num_images_in_single_gen}/{GEN_REQUIRED_FILES}"
            print(msg)
            continue

        print(f"CHECKING gen {gen_folder}")
        # print_stimuli_stat(gen_folder)
        print("******** validate_congruity_per_ratio **********")
        # validate that we have equal number of stimuli per ratio
        is_valid, msg = validate_congruity_per_ratio(gen_folder)
        if not is_valid:
            return False, msg

        # print_stimuli_stat(gen_folder)

        print("******** validate_left_and_right_sides **********")
        # Validate that the big number is found equally on the right side and on the left side
        is_valid, msg = validate_big_number_equally_found_in_left_and_right_sides(gen_folder, is_colors)
        if not is_valid:
            return False, msg

        if is_colors:
            print("******** validate_colors **********")
            print_stat_colors(gen_folder)
            # is_valid, msg = validate_colors(gen_folder)
            # if not is_valid:
            #     return False, msg


        print_stimuli_stat(gen_folder)
        print(f"DONE VALIDATING gen {gen_folder} {str(len(os.listdir(gen_folder)))}/{GEN_REQUIRED_FILES}")

    is_valid = True
    return is_valid, 'done'


# PATH = "/Users/gali.k/phd/phd_2021/stimuli_fix/equate_3"
PATH = "/Users/gali.k/phd/phd_2021/stimuli_fix/colors/equate_3"
is_colors = True if 'colors' in PATH else False
is_valid, msg = validate_files(PATH, is_colors=is_colors)
print(msg)


