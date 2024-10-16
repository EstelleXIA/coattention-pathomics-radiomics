import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="the task name")

args = parser.parse_args()

task = args.task
base_path = f"./data/{task}/TCGA-{task}/pathology_svs/"
files = sorted(os.listdir(base_path))
patients = list(set([x[:12] for x in files]))

svs_codebook = []
for patient in patients:
    select_files = list(filter(lambda x: x.startswith(patient), files))
    for idx, f in enumerate(select_files):
        svs_codebook.append([f"{patient}_{idx}", f])

svs_codebook = pd.DataFrame(svs_codebook)
save_base = f"./data/{task}/TCGA-{task}/"
svs_codebook.to_csv(os.path.join(save_base, f"svs_{task.lower()}.csv"), index=False, header=False)
