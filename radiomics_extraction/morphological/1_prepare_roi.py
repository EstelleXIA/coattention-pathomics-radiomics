# This file is to get ROI with the shape that can be divided by 32*32*12
import SimpleITK as sitk
import os
import cc3d
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="the task name")
args = parser.parse_args()

task = args.task
img_path = f"./data/{task}/TCIA-{task}/img_nii/"
mask_path = f"./data/{task}/TCIA-{task}/mask_nii/"
save_base = f"./data/{task}/TCIA-{task}/img_nii_roi/"
os.makedirs(save_base, exist_ok=True)
files = sorted(os.listdir(img_path))
basic_unit = (12, 32, 32)


def cal_boundary_xy(center, magnify):
    ''' to calculate the x,y boundary for the roi '''
    b_min = int(center - magnify / 2)
    b_min_0 = max(0, b_min)
    b_max = int(center - magnify / 2 + magnify)
    b_max_512 = min(512, b_max)
    if (b_min_0 == 0) and (b_max_512 < 512):
        return 0, magnify
    elif (b_min_0 > 0) and (b_max_512 == 512):
        return 512 - magnify, 512
    else:
        return b_min, b_max


def cal_boundary_z(center, magnify, max_z):
    ''' to calculate the z boundary for the roi '''
    b_min = int(center - magnify / 2)
    b_min_0 = max(0, b_min)
    b_max = int(center - magnify / 2 + magnify)
    b_max_z = min(b_max, max_z)
    if (b_min_0 == 0) and (b_max_z < max_z):
        return 0, magnify
    elif (b_min_0 > 0) and (b_max_z == max_z):
        return max_z - magnify, max_z
    elif (b_min_0 > 0) and (b_max_z < max_z):
        return b_min, b_max
    else:
        return None


for file in files:
    ct_img = sitk.ReadImage(os.path.join(img_path, file))
    ct_array = sitk.GetArrayFromImage(ct_img)
    mask_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_path, file.replace("_0000", ""))))
    mask_cc3d, number = cc3d.connected_components(mask_array, connectivity=26, return_N=True)
    stats = cc3d.statistics(mask_cc3d)
    candidates_idx = [x + 1 for x in range(stats["voxel_counts"][1:].shape[0])]
    bboxes = [stats["bounding_boxes"][x] for x in candidates_idx]
    bbox_candidates = [[bboxes[i][0].start, bboxes[i][0].stop,
                        bboxes[i][1].start, bboxes[i][1].stop,
                        bboxes[i][2].start, bboxes[i][2].stop, ] for i in range(len(bboxes))]

    if len(bbox_candidates) > 1:
        bbox_candidates = list(filter(lambda x: (x[1] - x[0] >= 10) and (x[3] - x[2] >= 12) and (x[5] - x[4] >= 12),
                                      bbox_candidates))

    for box_id, box in enumerate(bbox_candidates):
        z_range = abs(box[1] - box[0])
        x_range = abs(box[3] - box[2])
        y_range = abs(box[5] - box[4])

        z_magnify = ((z_range // basic_unit[0]) + 1) * basic_unit[0]
        x_magnify = ((x_range // basic_unit[1]) + 1) * basic_unit[1]
        y_magnify = ((y_range // basic_unit[2]) + 1) * basic_unit[2]

        z_center = (box[1] + box[0]) / 2
        x_center = (box[3] + box[2]) / 2
        y_center = (box[5] + box[4]) / 2

        if cal_boundary_z(z_center, z_magnify, ct_array.shape[0]):
            ct_roi = ct_array[int(cal_boundary_z(z_center, z_magnify, ct_array.shape[0])[0]):
                              int(cal_boundary_z(z_center, z_magnify, ct_array.shape[0])[1]),
                              int(cal_boundary_xy(x_center, x_magnify)[0]):
                              int(cal_boundary_xy(x_center, x_magnify)[1]),
                              int(cal_boundary_xy(y_center, y_magnify)[0]):
                              int(cal_boundary_xy(y_center, y_magnify)[1])]
        else:
            pad_zero_top = np.zeros([int((z_magnify - z_range) / 2), 512, 512])
            pad_zero_bottom = np.zeros([int((z_magnify - z_range)) - int((z_magnify - z_range) / 2), 512, 512])
            ct_array_new = np.concatenate([pad_zero_top, ct_array, pad_zero_bottom], axis=0)
            ct_roi = ct_array_new[:, int(cal_boundary_xy(x_center, x_magnify)[0]):
                                  int(cal_boundary_xy(x_center, x_magnify)[1]),
                                  int(cal_boundary_xy(y_center, y_magnify)[0]):
                                  int(cal_boundary_xy(y_center, y_magnify)[1])]

        save_roi_img = sitk.GetImageFromArray(ct_roi)
        save_roi_img.SetOrigin(ct_img.GetOrigin())
        save_roi_img.SetSpacing(ct_img.GetSpacing())
        save_roi_img.SetDirection(ct_img.GetDirection())

        sitk.WriteImage(save_roi_img, os.path.join(save_base, f"{file.replace('_0000.nii.gz', '')}_{box_id}.nii.gz"))


