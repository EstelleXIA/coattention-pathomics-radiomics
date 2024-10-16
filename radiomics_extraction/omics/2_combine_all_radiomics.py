# This file is to combine all the single files predicted in 1_extract_pyradiomics.py
import pandas as pd
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="the task name")
args = parser.parse_args()

task = args.task
files = glob.glob(f"./data/{task}/TCIA-{task}/radiomics_single_files/*.csv")
df = pd.concat([pd.read_csv(f) for f in files], axis=0)
df["imageFile"] = df["imageFile"].apply(lambda x: os.path.basename(x).split("_")[0])
df.to_csv(f"./data/{task}/TCIA-{task}/summary_radiomics.csv", index=False)

