from dependency import *
import numpy as np
# Label Encoding
from tensorflow.keras.utils import to_categorical
def encode_labels(sub_img_info, idx_num):

    #DIAG
    diag_label= sub_img_info['diagnosis'][idx_num]
    for index, label in enumerate(label_list):
        if diag_label in label:
            diag_index= index
        else:
            continue

    #Seven Point Labels
    #1.
    pigment_network_label= sub_img_info['pigment_network'][idx_num]
    for index, label in enumerate(label_list):
        if pigment_network_label in label:
            pigment_network_index=index
        else:
            continue

    #2.
    streaks_label= sub_img_info['streaks'][idx_num]
    for index, label in enumerate(label_list):
        if streaks_label in label:
            streaks_index= index
        else:
            continue

    #3.
    pigmentation_label= sub_img_info['pigmentation'][idx_num]
    for index, label in enumerate(label_list):
        if pigmentation_label in label:
            pigmentation_index=index
        else:
            continue

    #4.
    regression_structures_label= sub_img_info['regression_structures'][idx_num]
    for index, label in enumerate(label_list):
        if regression_structures_label in label:
            regression_structures_index=index
        else:
            continue

    #5.
    dots_and_globules_label= sub_img_info['dots_and_globules'][idx_num]
    for index, label in enumerate(label_list):
        if dots_and_globules_label in label:
            dots_and_globules_index=index
        else:
            continue

    #6.
    blue_whitish_veil_label= sub_img_info['blue_whitish_veil'][idx_num]
    for index, label in enumerate(label_list):
        if blue_whitish_veil_label in label:
            blue_whitish_veil_index=index
        else:
            continue

    #7.
    vascular_structures_label= sub_img_info['vascular_structures'][idx_num]
    for index, label in enumerate(label_list):
        if vascular_structures_label in label:
            vascular_structures_index=index
        else:
            continue

    return np.array([diag_index,
                     pigment_network_index,
                     streaks_index,
                     pigmentation_index,
                     regression_structures_index,
                     dots_and_globules_index,
                     blue_whitish_veil_index,
                     vascular_structures_index])


# test label encoding
def encode_test_labels(sub_img_info, idx_num):

    #DIAG
    diag_label= sub_img_info['diagnosis'][idx_num]
    for index, label in enumerate(label_list):
        if diag_label in label:
            diag_index= index
            diag_label_one_hot= to_categorical(diag_index, num_label)
        else:
            continue

    #Seven Point Labels
    #1.
    pigment_network_label= sub_img_info['pigment_network'][idx_num]
    for index, label in enumerate(label_list):
        if pigment_network_label in label:
            pigment_network_index=index
            pigment_network_label_one_hot= to_categorical(pigment_network_index, num_label)
        else:
            continue

    #2.
    streaks_label= sub_img_info['streaks'][idx_num]
    for index, label in enumerate(label_list):
        if streaks_label in label:
            streaks_index= index
            streaks_label_one_hot= to_categorical(streaks_index, num_label)
        else:
            continue

    #3.
    pigmentation_label= sub_img_info['pigmentation'][idx_num]
    for index, label in enumerate(label_list):
        if pigmentation_label in label:
            pigmentation_index=index
            pigmentation_label_one_hot= to_categorical(pigmentation_index, num_label)
        else:
            continue

    #4.
    regression_structures_label= sub_img_info['regression_structures'][idx_num]
    for index, label in enumerate(label_list):
        if regression_structures_label in label:
            regression_structures_index=index
            regression_structures_label_one_hot= to_categorical(regression_structures_index, num_label)
        else:
            continue

    #5.
    dots_and_globules_label= sub_img_info['dots_and_globules'][idx_num]
    for index, label in enumerate(label_list):
        if dots_and_globules_label in label:
            dots_and_globules_index=index
            dots_and_globules_label_one_hot= to_categorical(dots_and_globules_index, num_label)
        else:
            continue

    #6.
    blue_whitish_veil_label= sub_img_info['blue_whitish_veil'][idx_num]
    for index, label in enumerate(label_list):
        if blue_whitish_veil_label in label:
            blue_whitish_veil_index=index
            blue_whitish_veil_label_one_hot= to_categorical(blue_whitish_veil_index, num_label)
        else:
            continue

    #7.
    vascular_structures_label= sub_img_info['vascular_structures'][idx_num]
    for index, label in enumerate(label_list):
        if vascular_structures_label in label:
            vascular_structures_index=index
            vascular_structures_label_one_hot= to_categorical(vascular_structures_index, num_label)
        else:
            continue

    return np.array([diag_index,
                     pigment_network_index,
                     streaks_index,
                     pigmentation_index,
                     regression_structures_index,
                     dots_and_globules_index,
                     blue_whitish_veil_index,
                     vascular_structures_index]), np.array([diag_label_one_hot,
                                                            pigment_network_label_one_hot,
                                                            streaks_label_one_hot,
                                                            pigmentation_label_one_hot,
                                                            regression_structures_label_one_hot,
                                                            dots_and_globules_label_one_hot,
                                                            blue_whitish_veil_label_one_hot,
                                                            vascular_structures_label_one_hot])


def encode_metadata_labels(sub_img_info, idx_num):

    level_of_diagnostic_difficulty_label= sub_img_info["level_of_diagnostic_difficulty"][idx_num]
    for index, label in enumerate(level_of_diagnostic_difficulty_label):
        if level_of_diagnostic_difficulty_label in label:
            level_of_diagnostic_difficulty_label_index= index
            level_of_diagnostic_difficulty_label_one_hot= to_categorical(index, num_level_of_diagnostic_difficulty_label_list)
        else:
            continue

    elevation_label=sub_img_info['elevation'][idx_num]
    for index,label in enumerate(elevation_list):
        if elevation_label in label:
            elevation_label_index= index
            elevation_label_one_hot= to_categorical(index, num_elevation_list)
        else:
            continue

    location_label=sub_img_info['location'][idx_num]
    for index,label in enumerate(location_list):
        if location_label in label:
            location_label_index= index
            location_label_one_hot= to_categorical(index, num_location_list)
        else:
            continue

    # sex_label=sub_img_info['sex'][idx_num]
    # for index,label in enumerate(sex_list):
    #     if sex_label in label:
    #         sex_label_index= index
    #         sex_label_one_hot= to_categorical(index, num_sex_list)
    #     else:
    #         continue

    management_label=sub_img_info['management'][idx_num]
    for index,label in enumerate(management_list):
        if management_label in label:
            management_label_index= index
            management_label_one_hot= to_categorical(index, num_management_list)
        else:
            continue

    meta_data_vector_onehot = np.hstack([
        level_of_diagnostic_difficulty_label_one_hot,
        elevation_label_one_hot,
        location_label_one_hot,
        management_label_one_hot
    ])
    meta_data_vector = np.hstack([
        level_of_diagnostic_difficulty_label_index,
        elevation_label_index,
        location_label_index,
        management_label_index
    ])
    return meta_data_vector_onehot,meta_data_vector

