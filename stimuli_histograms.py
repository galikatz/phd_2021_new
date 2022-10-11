import os

import pandas as pd
import plotly.express as px
import glob
physical_properties = ["equate_1", "equate_2", "equate_3"]


def plot_stimuli_histogram(dir):
    histogram_df = pd.DataFrame(columns=["phyisical_property", "generation", "stimuli"])
    for phys_prop in physical_properties:
        phys_prop_df = pd.DataFrame(columns=["phyisical_property", "generation", "stimuli"])
        phys_list = []
        gen_list = []
        stimuli_list = []
        phys_prop_dir = dir + os.sep + phys_prop
        num_of_images_folders = len(glob.glob(phys_prop_dir + os.sep + f"images_*"))
        for i in range(1, num_of_images_folders + 1):
            images_dir = phys_prop_dir + os.sep + "images_" + str(i)
            num_of_stimuli = len(os.listdir(images_dir))
            phys_list.append(phys_prop)
            gen_list.append(i)
            stimuli_list.append(num_of_stimuli)
        phys_prop_df["phyisical_property"] = phys_list
        phys_prop_df["generation"] = gen_list
        phys_prop_df["stimuli"] = stimuli_list
        histogram_df = pd.concat([histogram_df, phys_prop_df])

    fig = px.bar(histogram_df, x='phyisical_property', y='stimuli', color='generation', barmode='group', text_auto=True, title=f"Number of stimuli per physical property from dir {dir}")
    fig.show()




if __name__ == '__main__':
    path = "C:\gali_phd" + os.sep
    plot_stimuli_histogram(path)

    path = "C:\gali_phd\colors" + os.sep
    plot_stimuli_histogram(path)