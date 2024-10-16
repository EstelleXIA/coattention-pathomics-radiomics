import os
import numpy as np
import random
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_utils import cal_task_metrics, generate_patient_task_codebook
import torch.nn.parallel
from loss import define_loss
from model import PaRaMILUnion
from dataset import CoAttnData


parser = argparse.ArgumentParser(description="Arguments for model training.")
parser.add_argument("--fold", type=int, default=0, help="fold number")
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("--decay", type=float, default=2e-4, help="weight decay")
parser.add_argument("--loss", type=str, default="nll_surv", help="loss type")
args = parser.parse_args()

seed = 330
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


fold = args.fold
lr = args.lr
decay = args.decay
start_epoch = 0
max_epochs = 200
best_c_index_val = 0
gc = 16
min_loss_val = 999
patience = 10
min_delta = 0.001
wait = 0
task = ["KIRC", "LUNG", "BRAT"]
patient_codebook = generate_patient_task_codebook(task)

save_ckpt_base = "/dssg/home/acct-clsyzs/clsyzs/xiayujia/CoPaRa/union_coattention/ckpts/"
os.makedirs(os.path.join(save_ckpt_base, f"{'_'.join(task)}_{lr}_{decay}", f"fold_{fold}"),
            exist_ok=True)

# initialize dataset and put into dataloader
train_dataset = CoAttnData(fold=fold, task=task, codebook_dict=patient_codebook, mode="train")
val_dataset = CoAttnData(fold=fold, task=task, codebook_dict=patient_codebook, mode="val")
print(f"--- Fold {fold} ---")
print(f"Union training on task: {', '.join(task)}")
print(f"Training on {len(train_dataset)} samples")
print(f"Validating on {len(val_dataset)} samples")

train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)

model = PaRaMILUnion(tasks=task, codebook_dict=patient_codebook).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Param Num: {total_params}")

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                             weight_decay=decay)

criterion = define_loss(args.loss)

all_train_losses, all_train_indexes, all_val_losses, all_val_indexes = [], [], [], []
val_c_index_tasks = np.zeros((len(task)))

# training loop
for epoch in range(start_epoch, max_epochs):
    random.seed(epoch)
    train_loss, val_loss = 0., 0.
    model.train()

    all_risk_scores = np.zeros((len(train_loader)))
    all_censorships = np.zeros((len(train_loader)))
    all_event_times = np.zeros((len(train_loader)))

    all_patients_train = []

    for batch_idx, data in tqdm(enumerate(train_loader)):
        patho, radio_shape, radio_order, radio_texture, radio_log_sigma, radio_wavelet, radio_cnn, \
            surv_discrete, surv_time, censor, patient_name = data

        data_WSI = patho.to(device)
        data_omic1 = radio_shape.type(torch.FloatTensor).to(device)
        data_omic2 = radio_order.type(torch.FloatTensor).to(device)
        data_omic3 = radio_texture.type(torch.FloatTensor).to(device)
        data_omic4 = radio_log_sigma.type(torch.FloatTensor).to(device)
        data_omic5 = radio_wavelet.type(torch.FloatTensor).to(device)
        data_omic6 = radio_cnn.type(torch.FloatTensor).to(device)
        surv_discrete = surv_discrete.type(torch.LongTensor).to(device)
        censor = censor.type(torch.FloatTensor).to(device)

        hazards, S, _, _ = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2,
                                 x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5,
                                 x_omic6=data_omic6, patient_name=patient_name)
        all_patients_train.append(patient_name[0])

        # calculate the loss
        loss = criterion(hazards=hazards, S=S, Y=surv_discrete, c=censor)  # nll_loss
        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = censor.item()
        all_event_times[batch_idx] = surv_time.item()

        loss_value = loss.item()
        train_loss += loss_value

        loss = loss / gc
        loss.backward()
        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()

    c_index_train_dict, c_index_train = cal_task_metrics(tasks=task, patient_id=all_patients_train,
                                                         censorships=all_censorships, event_times=all_event_times,
                                                         risk_scores=all_risk_scores, codebook_dict=patient_codebook)
    epoch_train_loss = train_loss / len(train_loader)
    all_train_losses.append(epoch_train_loss)
    all_train_indexes.append(c_index_train)
    print(f"[Train] Epoch: {epoch}, overall loss {epoch_train_loss:.4f}, c-index {c_index_train:.4f}, "
          f"task-level {c_index_train_dict}.")

    # validation stage
    all_patients_val = []
    with torch.no_grad():
        model.eval()
        all_risk_scores_val = np.zeros((len(val_loader)))
        all_censorships_val = np.zeros((len(val_loader)))
        all_event_times_val = np.zeros((len(val_loader)))

        for batch_idx, data in tqdm(enumerate(val_loader)):
            patho, radio_shape, radio_order, radio_texture, radio_log_sigma, radio_wavelet, radio_cnn, \
            surv_discrete, surv_time, censor, patient_name = data

            data_WSI = patho.to(device)
            data_omic1 = radio_shape.type(torch.FloatTensor).to(device)
            data_omic2 = radio_order.type(torch.FloatTensor).to(device)
            data_omic3 = radio_texture.type(torch.FloatTensor).to(device)
            data_omic4 = radio_log_sigma.type(torch.FloatTensor).to(device)
            data_omic5 = radio_wavelet.type(torch.FloatTensor).to(device)
            data_omic6 = radio_cnn.type(torch.FloatTensor).to(device)
            surv_discrete = surv_discrete.type(torch.LongTensor).to(device)
            censor = censor.type(torch.FloatTensor).to(device)

            hazards, S, _, _ = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2,
                                     x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5,
                                     x_omic6=data_omic6, patient_name=patient_name)
            all_patients_val.append(patient_name[0])

            # calculate the loss
            loss = criterion(hazards=hazards, S=S, Y=surv_discrete, c=censor)  # nll_loss

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores_val[batch_idx] = risk
            all_censorships_val[batch_idx] = censor.item()
            all_event_times_val[batch_idx] = surv_time.item()

            loss_value = loss.item()
            val_loss += loss_value

            del patho, radio_shape, radio_order, radio_texture, radio_log_sigma, radio_wavelet, radio_cnn

        c_index_val_dict, c_index_val = cal_task_metrics(tasks=task, patient_id=all_patients_val,
                                                         censorships=all_censorships_val,
                                                         event_times=all_event_times_val,
                                                         risk_scores=all_risk_scores_val,
                                                         codebook_dict=patient_codebook)
        if c_index_val > best_c_index_val:
            best_c_index_val = c_index_val
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(save_ckpt_base, f"{'_'.join(task)}_{lr}_{decay}",
                                                f"fold_{fold}", f"fold_{fold}_{c_index_val:.4f}.pth"))
            print(f"Save the best checkpoint!")
            for task_id, task_name in enumerate(task):
                val_c_index_tasks[task_id] = c_index_val_dict[task_name]
                print(f"Current {task_name}: {val_c_index_tasks[task_id]:.4f}")

        epoch_val_loss = val_loss / len(val_loader)
        all_val_losses.append(epoch_val_loss)
        all_val_indexes.append(c_index_val)
        print(f"[Validation] Epoch: {epoch}, overall loss {epoch_val_loss:.4f}, c-index {c_index_val:.4f}, "
              f"task-level {c_index_val_dict}, best c-index {best_c_index_val:.4f}.")

    if epoch > 20:
        if val_loss + min_delta < min_loss_val:
            min_loss_val = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

print("Finished!")
for task_id, task_name in enumerate(task):
    print(f"{task_name}: {val_c_index_tasks[task_id]:.4f}")
