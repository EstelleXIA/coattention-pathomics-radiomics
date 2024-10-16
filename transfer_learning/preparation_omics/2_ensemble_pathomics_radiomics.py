import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch
import os
import numpy as np
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, choices=["LIHC", "HSOC", "STAD", "BLCA"])
parser.add_argument("--omics", type=str, choices=["pathomics", "radiomics"])
args = parser.parse_args()

task = args.task
omics_type = args.omics
ss = StandardScaler()

save_combined_omics_single = f"./data/{task}/TCGA-{task}/coattn_{omics_type}/combined/"
os.makedirs(save_combined_omics_single, exist_ok=True)

omics_base_path = f"./data/{task}/TCGA-{task}/coattn_pathomics/"
omics_files = glob.glob(os.path.join(omics_base_path, "fold_*", "*"))

final_patients = sorted(list(set([os.path.basename(x)[:12] for x in omics_files])))

print("Starting combine all training files")
for patient in tqdm(final_patients):
    patient_omics_files = list(filter(lambda x: os.path.basename(x).startswith(patient), omics_files))
    patient_omics = []

    for feature_file in patient_omics_files:
        single_feature = torch.load(feature_file)
        transformed_single_feature = ss.fit_transform(single_feature)
        patient_omics.append(torch.from_numpy(transformed_single_feature))
    torch.save(torch.mean(torch.stack(patient_omics), dim=0), os.path.join(save_combined_omics_single, f"{patient}.pt"))

print("Starting combine to a single file")
omics_single_file = []

for patient in tqdm(final_patients):
    omics_feat = torch.load(os.path.join(save_combined_omics_single, f"{patient}.pt"))
    omics_single_file.append(omics_feat.numpy())

omics_stack = np.stack(omics_single_file, axis=0)
omics_save = np.reshape(omics_stack, (omics_stack.shape[0], -1))
omics_pd = pd.DataFrame(omics_save, index=final_patients)
omics_pd.to_csv(os.path.join(omics_base_path, f"ensembled_{omics_type}.csv"))

