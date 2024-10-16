import os
from skimage.filters import threshold_multiotsu
import cv2
import numpy as np
import skimage.io
import staintools
import pandas as pd
from tqdm import tqdm
import itertools
import openslide
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="the task name")
parser.add_argument("--idx", default=None, help="to fasten the cutting process")

args = parser.parse_args()


def standard_transform(standard_img):
    standard_img = staintools.LuminosityStandardizer.standardize(standard_img)
    stain_method = staintools.StainNormalizer(method="macenko")
    stain_method.fit(standard_img)
    return stain_method


task = args.task
base_path = f"./data/{task}/TCGA-{task}/pathology_svs/"
save_path = f"./data/{task}/TCGA-{task}/patches/"

files = sorted(os.listdir(base_path))
ref_img = np.load("./pathomics_extraction/preparation/target.npy")
stain_method = standard_transform(ref_img)

codebook = pd.read_csv(f"./data/{task}/TCGA-{task}/pathology_svs/svs_{task.lower()}.csv", header=None)
codebook_dict = dict(zip(codebook.iloc[:, 1].tolist(), codebook.iloc[:, 0].tolist()))

if __name__ == '__main__':
    if args.idx:
        process_files = [files[args.idx]]
    else:
        process_files = files
    for file in tqdm(process_files):
        file_name = codebook_dict[file]
        patient_id = file_name.split("_")[0]

        slide = openslide.OpenSlide(os.path.join(base_path, file))
        data_dim = slide.level_count
        if data_dim == 4:
            patch_size = (768, 768)
        else:
            assert data_dim == 3
            patch_size = (384, 384)

        os.makedirs(os.path.join(save_path, patient_id), exist_ok=True)

        img_path = os.path.join(base_path, file)
        img = skimage.io.imread(img_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_down = gray[::16, ::16]
        thresholds = threshold_multiotsu(gray_down[gray_down < 240])
        regions = (gray > thresholds[1])
        h, w = img.shape[:2]

        for i in range(0, h-patch_size[0]+1, int(patch_size[0])):
            for j in range(0, w-patch_size[1]+1, int(patch_size[1])):
                patch = img[i:i + patch_size[0], j:j + patch_size[1]]
                gray_mask = regions[i:i + patch_size[0], j:j + patch_size[1]]
                if gray_mask.mean() < 0.7:
                    try:
                        patch = staintools.LuminosityStandardizer.standardize(patch)
                        patch = stain_method.transform(patch)
                        if patch_size == (384, 384):
                            patch = cv2.resize(patch, (768, 768), interpolation=cv2.INTER_LINEAR)
                            patch_path = os.path.join(save_path, patient_id,
                                                      f"{file_name.split('.')[0]}_{i * 2}-{j * 2}.jpg")
                        else:
                            patch_path = os.path.join(save_path, patient_id, f"{file_name.split('.')[0]}_{i}-{j}.jpg")
                        skimage.io.imsave(patch_path, patch.astype(np.uint8))
                    except:
                        pass

