import os
import numpy as np
import random
from utils import nll_loss, NucleiData
import torch
from dataset import PatientSampler, PatchDataInMemory
from torch_geometric.loader import DataLoader
from sksurv.metrics import concordance_index_censored
from models import CellSpatialNet
from tqdm import tqdm
import json
import torch.optim as optim
import argparse


parser = argparse.ArgumentParser(description="Arguments for model training.")
parser.add_argument("--fold", type=int, help="fold number")
parser.add_argument("--lr", default=5e-4, type=float, help="learning rate")
parser.add_argument("--decay", default=1e-4, type=float, help="weight decay")
parser.add_argument("--task", type=str, help="training task")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fold = args.fold
lr = args.lr
decay = args.decay
task = args.task
max_epochs = 200
patch_select_num = 200

save_ckpt_base = f"./pathomics_extraction/omics/patch_feature_extraction/ckpts/{task}"
os.makedirs(os.path.join(save_ckpt_base, f"fold_{fold}"), exist_ok=True)

json_path = f"./pathomics_extraction/omics/patch_feature_extraction/splits/json_{task.lower()}/"
with open(os.path.join(json_path, f"fold_{fold}.json"), "r") as f:
    all_patients = json.load(f)

print(f"Fold {fold}, train on {len(all_patients['train'])} patients,"
      f" validate on {len(all_patients['val'])} patients.")

generator = torch.Generator()
generator.manual_seed(330)
train_dataset = PatchDataInMemory(fold=fold, mode="train", task=task, patch_num=patch_select_num)
val_dataset = PatchDataInMemory(fold=fold, mode="val", task=task, patch_num=patch_select_num)

train_dataloader = DataLoader(dataset=train_dataset,
                              sampler=PatientSampler(train_dataset, batch_size=patch_select_num, generator=generator),
                              batch_size=patch_select_num,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=0)
val_dataloader = DataLoader(dataset=val_dataset,
                            shuffle=False,
                            batch_size=400,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=0)

feat_corr_path = f"./pathomics_extraction/omics/nuclei_graph_construction/remove_feature_list/" \
                 f"{task}_removed_feature_list.txt"
with open(feat_corr_path, "r") as file:
    removed_feature_list = [line.strip() for line in file]
feat_size = int(80 - len(removed_feature_list))

model = CellSpatialNet(feat_size, 4, batch=True).to(device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=decay)

all_train_losses, all_train_indexes, all_val_losses, all_val_indexes = [], [], [], []
best_c_index_val = 0
min_loss_val = 999
patience = 20
min_delta = 0.001
wait = 0

for epoch in range(max_epochs):
    random.seed(epoch)
    train_loss, val_loss = 0., 0.
    model.train()

    all_risk_scores = np.zeros((len(train_dataloader.dataset.patients)))
    all_censorships = np.zeros((len(train_dataloader.dataset.patients)))
    all_event_times = np.zeros((len(train_dataloader.dataset.patients)))

    for batch_idx, data in (enumerate(train_dataloader)):
        graph_data, surv_discrete, surv_time, censor, this_patient = data
        surv_discrete, surv_time, censor = surv_discrete[[0]], surv_time[[0]], censor[[0]]
        hazards_all = []
        d = graph_data
        d.to(device)
        hazards = model(d.to(device))
        S_patch = torch.cumprod(1 - hazards, dim=1)
        S_patch_sum = -torch.sum(S_patch, dim=1)
        max_index = torch.argmax(S_patch_sum)
        S = S_patch[[max_index.item()]]
        hazard_max = hazards[[max_index.item()]]
        risk = S_patch_sum[max_index]
        loss = nll_loss(hazards=hazard_max, S=S, Y=surv_discrete.to(device), c=censor.to(device))
        loss_value = loss.item()
        train_loss += loss_value

        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = censor[0].item()
        all_event_times[batch_idx] = surv_time.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    c_index_train = concordance_index_censored((1 - all_censorships).astype(bool),
                                               all_event_times, all_risk_scores,
                                               tied_tol=1e-08)[0]
    epoch_train_loss = train_loss / len(train_dataloader.dataset.patients)
    all_train_losses.append(epoch_train_loss)
    all_train_indexes.append(c_index_train)
    print(f"[Train] Epoch: {epoch}, overall loss {epoch_train_loss:.4f}, c-index {c_index_train:.4f}.")

    # # validation stage
    with torch.no_grad():
        model.eval()
        all_risk_scores_val = np.zeros((len(val_dataloader.dataset.patients)))
        all_censorships_val = np.zeros((len(val_dataloader.dataset.patients)))
        all_event_times_val = np.zeros((len(val_dataloader.dataset.patients)))

        collect = {}
        for batch_idx, data in enumerate(val_dataloader):
            graph_data, surv_discrete, surv_time, censor, this_patients = data

            # calculate on every patch
            graph_data.to(device)
            hazards = model(graph_data.to(device))

            this_patients = np.array(this_patients)
            unique_patients = np.unique(this_patients)
            for unique_patient in unique_patients:
                if unique_patient not in collect:
                    collect[unique_patient] = {
                        'hazards': [hazards[this_patients == unique_patient].cpu()],
                        'surv_discrete': surv_discrete[this_patients == unique_patient][0],
                        'surv_time': surv_time[this_patients == unique_patient][0],
                        'censor': censor[this_patients == unique_patient][0]
                    }
                else:
                    collect[unique_patient]['hazards'].append(hazards[this_patients == unique_patient].cpu())

        for batch_idx, value in enumerate(collect.values()):

            hazards = torch.cat(value['hazards'], dim=0)
            S_patch = torch.cumprod(1 - hazards, dim=1)
            S_patch_sum = -torch.sum(S_patch, dim=1)
            max_index = torch.argmax(S_patch_sum)
            S = S_patch[[max_index.item()]]
            hazard_max = hazards[[max_index.item()]]
            risk = S_patch_sum[max_index]
            surv_discrete = value['surv_discrete'].unsqueeze(0)
            censor = value['censor'].unsqueeze(0)
            surv_time = value['surv_time'].unsqueeze(0)
            loss = nll_loss(hazards=hazard_max, S=S, Y=surv_discrete, c=censor)

            loss_value = loss.item()
            val_loss += loss_value

            all_risk_scores_val[batch_idx] = risk
            all_censorships_val[batch_idx] = censor.item()
            all_event_times_val[batch_idx] = surv_time.item()

        del graph_data

        epoch_val_loss = val_loss / len(val_dataloader.dataset.patients)
        c_index_val = concordance_index_censored((1 - all_censorships_val).astype(bool),
                                                 all_event_times_val, all_risk_scores_val,
                                                 tied_tol=1e-08)[0]
        if c_index_val >= best_c_index_val:
            best_c_index_val = c_index_val
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(save_ckpt_base, f"{task}_{settings}", f"fold_{fold}",
                                                f"fold_{fold}_{c_index_val:.4f}_{epoch_val_loss:.4f}.pth"))
            print(f"Save the best checkpoint!")

        epoch_val_loss = val_loss / len(val_dataloader.dataset.patients)
        all_val_losses.append(epoch_val_loss)
        all_val_indexes.append(c_index_val)
        print(f"[Validation] Epoch: {epoch}, overall loss {epoch_val_loss:.4f}, c-index {c_index_val:.4f}, "
              f"best c-index {best_c_index_val:.4f}.")

    if epoch > 20:
        if val_loss + min_delta < min_loss_val:
            min_loss_val = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break
