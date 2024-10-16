import os
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="the task name")
args = parser.parse_args()

task = args.task
histo_feature_path = f"./data/{task}/TCGA-{task}/histo_features/"
save_json_path = f"./data/{task}/TCGA-{task}/resnet_json/"
os.makedirs(save_json_path, exist_ok=True)
patients = sorted(os.listdir(histo_feature_path))

for patient in tqdm(patients):
    files = sorted(os.listdir(os.path.join(histo_feature_path, patient)))
    jpg_files = [x.replace("csv", "jpg") for x in files]
    with open(os.path.join(save_json_path, f"{patient}.json"), "w") as f:
        json.dump(jpg_files, f)
