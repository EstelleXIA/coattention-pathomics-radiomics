import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import json
import numpy as np


class PaRaData(Dataset):
    def __init__(self, task="BLCA", mode="train"):
        super(PaRaData, self).__init__()

        # dataset split
        json_path = f"./transfer_learning/splits/"
        with open(os.path.join(json_path, f"{task}_splits.json"), "r") as f:
            self.patients = json.load(f)[mode]

        self.label_path = f"./transfer_learning/preparation/clinical_info_clean_{task.lower()}.csv"
        self.labels = pd.read_csv(self.label_path, index_col="patient")

        self.omics_path = f"./data/{task}/TCGA-{task}/coattn_pathomics/"
        self.patho_csv = pd.read_csv(os.path.join(self.omics_path, "ensembled_pathomics.csv"), index_col=0)
        self.radio_csv = pd.read_csv(os.path.join(self.omics_path, "ensembled_radiomics.csv"), index_col=0)

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, item):
        patient_name = self.patients[item]

        labels = self.labels.loc[patient_name]
        surv_discrete = labels["time_bins"]
        surv_time = labels["time"]
        censor = 1 - labels["event"]

        patho = torch.from_numpy(np.reshape(self.patho_csv.loc[patient_name].values, (6, 256)))
        radio = torch.from_numpy(np.reshape(self.radio_csv.loc[patient_name].values, (6, 256)))

        data = [patho, radio, surv_discrete, surv_time, censor, patient_name]

        return data


