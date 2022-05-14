import os
import glob
import shutil
import pathlib
from evolution_utils import RATIOS, RATIO_NUMBERS
GEN_REQUIRED_FILES = 1200
CONG_INGCONG_REQUIRED_FILES = 600
PER_RATIO_REQUIRED_FILES = 100


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


def validate_files(path) -> (bool, str):
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

        for ratio in RATIOS: #cong86_6_7_2980_r_c_l_m.jpg
            incong_files_per_ratio = glob.glob(gen_folder + os.sep + f"incong{ratio}*.jpg")
            cong_files_per_ratio = glob.glob(gen_folder + os.sep + f"cong{ratio}*.jpg")
            is_valid, msg = validate_files_per_ratio(gen_folder + os.sep, incong_files_per_ratio, False)
            if not is_valid:
                return False, f"Checking {gen_folder}: {msg}"
            is_valid, msg = validate_files_per_ratio(gen_folder + os.sep, cong_files_per_ratio, True)
            if not is_valid:
                return False, f"Checking {gen_folder}: {msg}"

        print(f"Stimuli amount is valid in current gen {gen_folder}")
    is_valid = True
    return is_valid, f"Stimuli amount is valid in per all ratios in all gens"

PATH = "/Users/gali.k/phd/Genereating_dot_arrays"


if __name__ == '__main__':
    is_valid, msg = validate_files(PATH)
    print(msg)
