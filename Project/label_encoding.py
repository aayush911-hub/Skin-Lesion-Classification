from dependency import *
import numpy as np
from tensorflow.keras.utils import to_categorical


def _encode_grouped_label(value, groups, num_classes=None):
    for index, group in enumerate(groups):
        if value in group:
            if num_classes is None:
                return index
            return index, to_categorical(index, num_classes)
    raise ValueError(f"Unknown label {value!r} for {groups}")


def _encode_metadata_field(value, categories, num_classes):
    for index, label in enumerate(categories):
        if value in label:
            return index, to_categorical(index, num_classes)
    raise ValueError(f"Unknown metadata value {value!r} for {categories}")


def encode_labels(sub_img_info, idx_num):
    diag_index = _encode_grouped_label(sub_img_info['diagnosis'][idx_num], label_list)
    pigment_network_index = _encode_grouped_label(
        sub_img_info['pigment_network'][idx_num], pigment_network_label_list)
    streaks_index = _encode_grouped_label(
        sub_img_info['streaks'][idx_num], streaks_label_list)
    pigmentation_index = _encode_grouped_label(
        sub_img_info['pigmentation'][idx_num], pigmentation_label_list)
    regression_structures_index = _encode_grouped_label(
        sub_img_info['regression_structures'][idx_num], regression_structures_label_list)
    dots_and_globules_index = _encode_grouped_label(
        sub_img_info['dots_and_globules'][idx_num], dots_and_globules_label_list)
    blue_whitish_veil_index = _encode_grouped_label(
        sub_img_info['blue_whitish_veil'][idx_num], blue_whitish_veil_label_list)
    vascular_structures_index = _encode_grouped_label(
        sub_img_info['vascular_structures'][idx_num], vascular_structures_label_list)

    return np.array([
        diag_index,
        pigment_network_index,
        streaks_index,
        pigmentation_index,
        regression_structures_index,
        dots_and_globules_index,
        blue_whitish_veil_index,
        vascular_structures_index,
    ])


def encode_test_labels(sub_img_info, idx_num):
    diag_index, diag_label_one_hot = _encode_grouped_label(
        sub_img_info['diagnosis'][idx_num], label_list, num_label)
    pigment_network_index, pigment_network_label_one_hot = _encode_grouped_label(
        sub_img_info['pigment_network'][idx_num], pigment_network_label_list, num_pigment_network_label)
    streaks_index, streaks_label_one_hot = _encode_grouped_label(
        sub_img_info['streaks'][idx_num], streaks_label_list, num_streaks_label)
    pigmentation_index, pigmentation_label_one_hot = _encode_grouped_label(
        sub_img_info['pigmentation'][idx_num], pigmentation_label_list, num_pigmentation_label)
    regression_structures_index, regression_structures_label_one_hot = _encode_grouped_label(
        sub_img_info['regression_structures'][idx_num], regression_structures_label_list, num_regression_structures_label)
    dots_and_globules_index, dots_and_globules_label_one_hot = _encode_grouped_label(
        sub_img_info['dots_and_globules'][idx_num], dots_and_globules_label_list, num_dots_and_globules_label)
    blue_whitish_veil_index, blue_whitish_veil_label_one_hot = _encode_grouped_label(
        sub_img_info['blue_whitish_veil'][idx_num], blue_whitish_veil_label_list, num_blue_whitish_veil_label)
    vascular_structures_index, vascular_structures_label_one_hot = _encode_grouped_label(
        sub_img_info['vascular_structures'][idx_num], vascular_structures_label_list, num_vascular_structures_label)

    return np.array([
        diag_index,
        pigment_network_index,
        streaks_index,
        pigmentation_index,
        regression_structures_index,
        dots_and_globules_index,
        blue_whitish_veil_index,
        vascular_structures_index,
    ]), [
        diag_label_one_hot,
        pigment_network_label_one_hot,
        streaks_label_one_hot,
        pigmentation_label_one_hot,
        regression_structures_label_one_hot,
        dots_and_globules_label_one_hot,
        blue_whitish_veil_label_one_hot,
        vascular_structures_label_one_hot,
    ]


def encode_metadata_labels(sub_img_info, idx_num):
    level_of_diagnostic_difficulty_label_index, level_of_diagnostic_difficulty_label_one_hot = _encode_metadata_field(
        sub_img_info["level_of_diagnostic_difficulty"][idx_num],
        level_of_diagnostic_difficulty_label_list,
        num_level_of_diagnostic_difficulty_label_list,
    )
    elevation_label_index, elevation_label_one_hot = _encode_metadata_field(
        sub_img_info['elevation'][idx_num], elevation_list, num_elevation_list)
    location_label_index, location_label_one_hot = _encode_metadata_field(
        sub_img_info['location'][idx_num], location_list, num_location_list)
    management_label_index, management_label_one_hot = _encode_metadata_field(
        sub_img_info['management'][idx_num], management_list, num_management_list)

    meta_data_vector_onehot = np.hstack([
        level_of_diagnostic_difficulty_label_one_hot,
        elevation_label_one_hot,
        location_label_one_hot,
        management_label_one_hot,
    ])
    meta_data_vector = np.hstack([
        level_of_diagnostic_difficulty_label_index,
        elevation_label_index,
        location_label_index,
        management_label_index,
    ])
    return meta_data_vector_onehot, meta_data_vector
