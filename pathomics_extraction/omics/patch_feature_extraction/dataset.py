import os
from torch_geometric.data import Dataset
from utils import NucleiData
import glob
from torch.utils import data
import pandas as pd
import torch
import json
from tqdm import tqdm


class PatchDataInMemory(Dataset):
    def __init__(self, num_bins=4, fold=0, mode="train", task="KIRC", patch_num=200):
        super(PatchDataInMemory, self).__init__()

        json_path = f"./pathomics_extraction/omics/patch_feature_extraction/splits/json_{task.lower()}/"
        self.fold = fold
        assert mode in ("train", "val", ("train", "val"))
        if mode in ("train", "val"):
            mode = (mode,)
        self.patients = []
        for mode_single in mode:
            with open(os.path.join(json_path, f"fold_{self.fold}.json"), "r") as f:
                self.patients.extend(json.load(f)[mode_single][:])

        # get discrete survival label
        self.bins = num_bins
        self.label_path = f"./pathomics_extraction/omics/patch_feature_extraction/clean_label_{task.lower()}.csv"
        label_data = pd.read_csv(self.label_path)
        label_bins, cut_point = pd.qcut(label_data["time"], self.bins,
                                        labels=[i for i in range(num_bins)], retbins=True)
        label_bins = label_bins.to_frame("time_bins")
        self.labels = pd.concat([label_data, label_bins], axis=1).set_index("patient")

        self.graph_path = f"./data/{task}/TCGA-{task}/graph_pt/"
        self.graph_data_list = []

        self.patient_to_pt_index = {}
        self.pt_index_to_patient = {}

        to_delete = []
        for patient in tqdm(self.patients):
            load = torch.load(os.path.join(self.graph_path, f"{patient}_graph.pt"))
            if len(load) >= patch_num:
                self.patient_to_pt_index[patient] = list(range(len(self.graph_data_list), len(self.graph_data_list) + len(load)))
                self.graph_data_list.extend(load)
                self.pt_index_to_patient.update({k: patient for k in self.patient_to_pt_index[patient]})
            else:
                to_delete.append(patient)
        for delete_patient in to_delete:
            self.patients.remove(delete_patient)

    def len(self):
        return len(self.graph_data_list)

    def get(self, item):
        labels = self.labels.loc[self.pt_index_to_patient[item]]
        surv_discrete = labels["time_bins"]
        surv_time = labels["time"]
        censor = 1 - labels["event"]
        graph_data = self.graph_data_list[item]

        return graph_data, torch.tensor(surv_discrete), torch.tensor(surv_time), torch.tensor(censor), \
               self.pt_index_to_patient[item]


class PatchDataPatient(Dataset):
    def __init__(self, num_bins=4, fold=0, mode="train", task="KIRC"):
        super(PatchDataPatient, self).__init__()
        assert mode in ("train", "val")
        json_path = f"./pathomics_extraction/omics/patch_feature_extraction/splits/json_{task.lower()}/"
        self.fold = fold
        with open(os.path.join(json_path, f"fold_{self.fold}.json"), "r") as f:
            self.patients = json.load(f)[mode]

        # get discrete survival label
        self.bins = num_bins
        self.label_path = f"./pathomics_extraction/omics/patch_feature_extraction/clean_label_{task.lower()}.csv"
        label_data = pd.read_csv(self.label_path)
        label_bins, cut_point = pd.qcut(label_data["time"], self.bins,
                                        labels=[i for i in range(num_bins)], retbins=True)
        label_bins = label_bins.to_frame("time_bins")
        self.labels = pd.concat([label_data, label_bins], axis=1).set_index("patient")

        self.patient_to_pt_index = {}
        self.pt_index_to_patient = {}
        self.pts = []
        print(f"prepare {mode} dataset")
        for patient in tqdm(self.patients):
            patient_pts = glob.glob(os.path.join(self.graph_path, f"{patient}", "*.pt"))
            self.patient_to_pt_index[patient] = list(range(len(self.pts), len(self.pts) + len(patient_pts)))
            self.pt_index_to_patient.update({k: patient for k in self.patient_to_pt_index[patient]})
            self.pts.extend(patient_pts)

        self.train = mode == "train"

    def len(self):
        return len(self.pts)

    def get(self, idx):
        graph_data = torch.load(self.pts[idx])
        labels = self.labels.loc[self.pt_index_to_patient[idx]]
        surv_discrete = labels["time_bins"]
        surv_time = labels["time"]
        censor = 1 - labels["event"]

        return graph_data, torch.tensor(surv_discrete), torch.tensor(surv_time), torch.tensor(censor), \
               self.pt_index_to_patient[idx]


class PatientSampler(data.sampler.Sampler):
    def __init__(self, data_source: PatchDataPatient, batch_size: int, generator: torch.Generator):
        super(PatientSampler, self).__init__(data_source)
        self.batch_size = batch_size
        self.data_source = data_source
        self.generator = generator

    def __iter__(self):
        index = []
        patient_index = torch.randperm(len(self.data_source.patients), generator=self.generator).tolist()
        for idx in patient_index:
            patient = self.data_source.patients[idx]
            patches_index = self.data_source.patient_to_pt_index[patient]
            index.extend([patches_index[ii] for ii in torch.randperm(len(patches_index),
                                                                     generator=self.generator).tolist()[:self.batch_size]])
        yield from index

    def __len__(self):
        return len(self.data_source.patients) * self.batch_size


class PatchGraphDataInfer(Dataset):
    def __init__(self, task):
        super(PatchGraphDataInfer, self).__init__()

        self.graph_path = f"./data/{task}/TCGA-{task}/graph_pt/"
        self.patients = sorted([x[:12] for x in os.listdir(self.graph_path)])

    def len(self):
        return len(self.patients)

    def get(self, item):
        graph_data = torch.load(os.path.join(self.graph_path, f"{self.patients[item]}_graph.pt"))
        return graph_data, self.patients[item]

