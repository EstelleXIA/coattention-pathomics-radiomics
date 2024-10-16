import __main__
import os
from sklearn.preprocessing import StandardScaler
from utils import *
setattr(__main__, "NucleiData", NucleiData)
import torch
from dataset import PatchGraphDataInfer
from torch_geometric.data import DataLoader
from models import CellSpatialNetInfer
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="task")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task = args.task

save_inference = f"./data/{task}/TCGA-{task}/patch_features_cellular/"
save_inference_per_fold = f"./data/{task}/TCGA-{task}/patch_features_cellular_per_fold/"
os.makedirs(save_inference, exist_ok=True)
os.makedirs(save_inference_per_fold, exist_ok=True)

val_dataset = PatchGraphDataInfer(task=task)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

feat_corr_path = f"./pathomics_extraction/omics/nuclei_graph_construction/remove_feature_list/" \
                 f"{task}_removed_feature_list.txt"

with open(feat_corr_path, "r") as file:
    removed_feature_list = [line.strip() for line in file]

feat_size = int(80 - len(removed_feature_list))
model = CellSpatialNetInfer(feat_size, 4, batch=True).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Param Num: {total_params}")


for i in range(5):
    print(f"Use the checkpoint from fold {i}!")
    ckpts_base = f"./pathomics_extraction/omics/patch_feature_extraction/ckpts/{task}"
    checkpoint = torch.load(os.path.join(ckpts_base, f"fold_{i}.pth"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
        model.eval()

        for batch_idx, data in tqdm(enumerate(val_loader)):
            graph_data, patient_name = data
            patient_features = []
            # calculate on every patch
            for d in graph_data:
                embedding, _ = model(d.to(device))
                patient_features.append(embedding.squeeze(0))

            torch.save(torch.stack(patient_features).cpu(),
                       os.path.join(save_inference_per_fold, f"{patient_name[0]}_{i}.torch"))
            del graph_data


# combine five-fold features
ss = StandardScaler()
all_feature_files = os.listdir(save_inference_per_fold)
final_patients = sorted(list(set([x.split("_")[0] for x in all_feature_files])))
for patient in tqdm(final_patients):
    patient_feature_files = list(filter(lambda x: x.startswith(patient), all_feature_files))
    patient_features = []
    for feature_file in patient_feature_files:
        single_feature = torch.load(os.path.join(save_inference_per_fold, feature_file))
        transformed_single_feature = ss.fit_transform(single_feature)
        patient_features.append(torch.from_numpy(transformed_single_feature))
    torch.save(torch.mean(torch.stack(patient_features), dim=0), os.path.join(save_inference, f"{patient}.pt"))

