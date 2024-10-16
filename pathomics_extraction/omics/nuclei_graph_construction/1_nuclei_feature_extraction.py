import os
import skimage.io
import numpy as np
import pandas as pd
import histomicstk as htk
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="the task name")
parser.add_argument("--idx", default=None, help="to fasten the histomics extraction process")

args = parser.parse_args()

task = args.task
img_path = f"./data/{task}/TCGA-{task}/patches/"
pickle_path = f"./data/{task}/TCGA-{task}/pickle/"
save_path = f"./data/{task}/TCGA-{task}/histo_features/"
os.makedirs(save_path, exist_ok=True)


def extract_histomicstk(sample):

    img = skimage.io.imread(os.path.join(img_path, sample.split("_")[0],
                                         sample.replace("pickle", "jpg")))[:, :, :3]
    # color deconvolution, weight for H&E
    W = np.array([[0.650, 0.072, 0], [0.704, 0.990, 0], [0.286, 0.105, 0]])

    # perform standard color deconvolution
    im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(img, W).Stains

    with open(os.path.join(pickle_path, sample.split("_")[0], sample), "rb") as f:
        mask_info = pickle.load(f)

    if (len(mask_info["inst_uid"]) >= 20) and ((mask_info["inst_type"] == 1).sum() > 10):

        im_nuclei_seg_mask = mask_info["inst_map"].toarray()
        im_nuclei_stain = im_stains[:, :, 0]
        nuclei_features = htk.features.compute_nuclei_features(im_nuclei_seg_mask, im_nuclei_stain)
        # nuclei_features = nuclei_features.astype('float16')
        nuclei_features["Label"] = nuclei_features["Label"].astype(int)

        type_prob = pd.DataFrame(mask_info["inst_type_prod"], columns=["Prob"])
        type_class = pd.DataFrame(mask_info["inst_type"], columns=["Type"])

        type_info = pd.concat([type_class, type_prob], axis=1)
        type_info["Label"] = type_info.index.astype(int)

        merge_df = pd.merge(type_info, nuclei_features, on="Label")

        merge_df = merge_df.drop(["Identifier.Xmin", "Identifier.Ymin",
                                  "Identifier.Xmax", "Identifier.Ymax",
                                  "Identifier.WeightedCentroidX", "Identifier.WeightedCentroidY"], axis=1)
        patient_name = sample.split("_")[0]
        os.makedirs(os.path.join(save_path, patient_name), exist_ok=True)
        merge_df.to_csv(os.path.join(save_path, patient_name, sample.replace("pickle", "csv")),
                        header=True, index=False)


patients = sorted(os.listdir(pickle_path))

if args.idx:
    process_patients = [patients[args.idx]]
else:
    process_patients = patients

for patient in tqdm(process_patients):
    samples = os.listdir(os.path.join(pickle_path, patient))
    p = Pool(64)
    _ = p.map(extract_histomicstk, samples)
