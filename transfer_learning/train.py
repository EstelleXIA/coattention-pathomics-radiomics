import os
import numpy as np
import random
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from sksurv.metrics import concordance_index_censored
import torch.nn.parallel
from loss import define_loss
from model import PaRaFrozen
from dataset import PaRaData


parser = argparse.ArgumentParser(description="Arguments for model training.")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--decay", type=float, default=2e-4, help="weight decay")
parser.add_argument("--task", type=str, choices=["LIHC", "HSOC", "STAD", "BLCA"], help="task")
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

task = args.task
lr = args.lr
decay = args.decay

start_epoch = 0
max_epochs = 200
min_loss_val = 999
patience = 10
min_delta = 0.001
wait = 0
best_c_index_val = 0
gc = 16

model = PaRaFrozen().to(device)

save_ckpt_base = "./transfer_learning/ckpts/"
os.makedirs(os.path.join(save_ckpt_base, f"{task}_{lr}_{decay}"), exist_ok=True)

# initialize dataset and put into dataloader
train_dataset = PaRaData(task=task, mode="train")
val_dataset = PaRaData(task=task, mode="val")
print(f"Training on {len(train_dataset)} samples")
print(f"Validating on {len(val_dataset)} samples")

train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameter Number: {total_params}")
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                             weight_decay=decay)
criterion = define_loss(args.loss)

all_train_losses, all_train_indexes, all_val_losses, all_val_indexes = [], [], [], []

# training loop
for epoch in range(start_epoch, max_epochs):
    random.seed(epoch)
    train_loss, val_loss = 0., 0.
    model.train()

    all_risk_scores = np.zeros((len(train_loader)))
    all_censorships = np.zeros((len(train_loader)))
    all_event_times = np.zeros((len(train_loader)))

    for batch_idx, data in tqdm(enumerate(train_loader)):
        patho, radio, surv_discrete, surv_time, censor, patient_name = data

        data_pathomics = patho.type(torch.FloatTensor).to(device)
        data_radiomics = radio.type(torch.FloatTensor).to(device)

        surv_discrete = surv_discrete.type(torch.LongTensor).to(device)
        censor = censor.type(torch.FloatTensor).to(device)

        hazards, S, _, _ = model(x_path=data_pathomics, x_omic=data_radiomics)

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

    c_index_train = concordance_index_censored((1 - all_censorships).astype(bool),
                                               all_event_times, all_risk_scores,
                                               tied_tol=1e-08)[0]
    epoch_train_loss = train_loss / len(train_loader)
    all_train_losses.append(epoch_train_loss)
    all_train_indexes.append(c_index_train)
    print(f"[Train] Epoch: {epoch}, overall loss {epoch_train_loss:.4f}, c-index {c_index_train:.4f}.")

    # validation stage
    with torch.no_grad():
        model.eval()
        all_risk_scores_val = np.zeros((len(val_loader)))
        all_censorships_val = np.zeros((len(val_loader)))
        all_event_times_val = np.zeros((len(val_loader)))

        for batch_idx, data in tqdm(enumerate(val_loader)):
            patho, radio, surv_discrete, surv_time, censor, patient_name = data

            data_pathomics = patho.type(torch.FloatTensor).to(device)
            data_radiomics = radio.type(torch.FloatTensor).to(device)

            surv_discrete = surv_discrete.type(torch.LongTensor).to(device)
            censor = censor.type(torch.FloatTensor).to(device)

            hazards, S, _, _ = model(x_path=data_pathomics, x_omic=data_radiomics)

            # calculate the loss
            loss = criterion(hazards=hazards, S=S, Y=surv_discrete, c=censor)  # nll_loss

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores_val[batch_idx] = risk
            all_censorships_val[batch_idx] = censor.item()
            all_event_times_val[batch_idx] = surv_time.item()

            loss_value = loss.item()
            val_loss += loss_value

            del patho, radio

        c_index_val = concordance_index_censored((1 - all_censorships_val).astype(bool),
                                                 all_event_times_val, all_risk_scores_val,
                                                 tied_tol=1e-08)[0]

        epoch_val_loss = val_loss / len(val_loader)

        if c_index_val > best_c_index_val:
            best_c_index_val = c_index_val
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(save_ckpt_base, f"{task}_{args.lr}_{decay}",
                                                f"tl_{c_index_val:.4f}_{epoch_val_loss:.4f}.pth"))
            print(f"Save the best checkpoint!")

        all_val_losses.append(epoch_val_loss)
        all_val_indexes.append(c_index_val)
        print(f"[Validation] Epoch: {epoch}, overall loss {epoch_val_loss:.4f}, c-index {c_index_val:.4f}, "
              f"best c-index {best_c_index_val:.4f}.")

    if epoch > 100:
        if val_loss + min_delta < min_loss_val:
            min_loss_val = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break
