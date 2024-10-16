import os
import numpy as np
import random
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from sksurv.metrics import concordance_index_censored
import torch.nn.parallel
from models import PaRaMIL, PaRaMILwo, PathMIL, RadMIL
from loss import define_loss
from dataset import CoAttnData

parser = argparse.ArgumentParser(description="Arguments for model training.")
parser.add_argument("--task", type=str, help="task name")
parser.add_argument("--fold", type=int, default=0, help="fold number")
parser.add_argument("--model", type=str, choices=["PaRa_MIL", "PaRa_MIL_wo", "Path_MIL", "Rad_MIL"],
                    default="PaRa_MIL", help="model type")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
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
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

fold = args.fold
lr = args.lr
decay = args.decay
task = args.task

start_epoch = 0
max_epochs = 200
min_loss_val = 999
patience = 10
min_delta = 0.001
wait = 0
best_c_index_val = 0
gc = 16

save_ckpt_base = "./coattention-pathomics-radiomics/survival_modeling/ckpts/"
os.makedirs(os.path.join(save_ckpt_base, f"{task}_{args.model}_{lr}_{decay}", f"fold_{fold}"), exist_ok=True)

# initialize dataset and put into dataloader
train_dataset = CoAttnData(fold=fold, task=task, mode="train", settings=args.model)
val_dataset = CoAttnData(fold=fold, task=task, mode="val", settings=args.model)
print(f"--- Fold {fold} ---")
print(f"Training on {len(train_dataset)} samples")
print(f"Validating on {len(val_dataset)} samples")

train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)

if args.model == "PaRa_MIL":
    model = PaRaMIL().to(device)
elif args.model == "PaRa_MIL_wo":
    model = PaRaMILwo().to(device)
elif args.model == "Path_MIL":
    model = PathMIL().to(device)
elif args.model == "Rad_MIL":
    model = RadMIL().to(device)
else:
    raise NotImplementedError

total_params = sum(p.numel() for p in model.parameters())
print(f"Param Num: {total_params}")
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
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
                                 x_omic6=data_omic6)

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
                                     x_omic6=data_omic6)

            # calculate the loss
            loss = criterion(hazards=hazards, S=S, Y=surv_discrete, c=censor)  # nll_loss

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores_val[batch_idx] = risk
            all_censorships_val[batch_idx] = censor.item()
            all_event_times_val[batch_idx] = surv_time.item()

            loss_value = loss.item()
            val_loss += loss_value

            del patho, radio_shape, radio_order, radio_texture, radio_log_sigma, radio_wavelet, radio_cnn

        c_index_val = concordance_index_censored((1 - all_censorships_val).astype(bool),
                                                 all_event_times_val, all_risk_scores_val,
                                                 tied_tol=1e-08)[0]
        if c_index_val > best_c_index_val:
            best_c_index_val = c_index_val
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(save_ckpt_base, f"{task}_{args.model}_{lr}_{decay}", f"fold_{fold}",
                                                f"fold_{fold}_{c_index_val:.4f}.pth"))
            print(f"Save the best checkpoint!")

        epoch_val_loss = val_loss / len(val_loader)
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
