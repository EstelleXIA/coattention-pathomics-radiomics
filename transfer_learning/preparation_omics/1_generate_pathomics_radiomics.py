from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
import pandas as pd
import json
from utils import RadPathModel
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.parallel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, choices=["LIHC", "HSOC", "STAD", "BLCA"], help="running task")
args = parser.parse_args()


# define dataset
class UniData(Dataset):
    def __init__(self, task="BLCA"):
        super(UniData, self).__init__()

        self.radio_cnn_path = f"./data/{task}/TCIA-{task}/img_nii_features/"
        self.pathomics_base = f"./data/{task}/TCGA-{task}/patch_features_merged/"

        json_path = f"./transfer_learning/splits/"
        with open(os.path.join(json_path, f"{task}_splits.json"), "r") as f:
            all_patients = json.load(f)
        self.patients = all_patients["train"] + all_patients["val"]

        # define radiomics path generated by PyRadiomics
        radiomics_base = f"./data/{task}/TCIA-{task}/summary_radiomics.csv"
        radiomics_data = pd.read_csv(radiomics_base).set_index("imageFile")
        all_columns = list(radiomics_data.columns)
        diagnostic_cols = sorted(list(filter(lambda x: "diagnostics" in x, all_columns)))
        shape_cols = sorted(list(filter(lambda x: "original_shape" in x, all_columns)))
        order_cols = sorted(list(filter(lambda x: "original_firstorder" in x, all_columns)))
        texture_cols = sorted(list(filter(lambda x: ("original_glcm" in x) or ("original_gldm" in x)
                                          or ("original_glrlm" in x) or ("original_glszm" in x)
                                          or ("original_ngtdm" in x), all_columns)))
        log_cols = sorted(list(filter(lambda x: "log-sigma" in x, all_columns)))
        wavelet_cols = sorted(list(filter(lambda x: "wavelet" in x, all_columns)))
        assert len(all_columns) == len(diagnostic_cols) + len(shape_cols) + len(order_cols) \
               + len(texture_cols) + len(log_cols) + len(wavelet_cols)

        # perform ss for radiomics data
        radiomics_data = radiomics_data.drop(diagnostic_cols, axis=1)
        ss = StandardScaler()
        radio_data = pd.DataFrame(ss.fit_transform(radiomics_data),
                                  index=radiomics_data.index,
                                  columns=radiomics_data.columns)
        self.shape = radio_data[shape_cols]
        self.order = radio_data[order_cols]
        self.texture = radio_data[texture_cols]
        self.log_sigma = radio_data[log_cols]
        self.wavelet = radio_data[wavelet_cols]

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, item):
        patient_name = self.patients[item]

        # radiological data
        radio_shape = torch.from_numpy(self.shape.loc[patient_name, :].values).to(torch.float32)
        radio_order = torch.from_numpy(self.order.loc[patient_name, :].values).to(torch.float32)
        radio_texture = torch.from_numpy(self.texture.loc[patient_name, :].values).to(torch.float32)
        radio_log_sigma = torch.from_numpy(self.log_sigma.loc[patient_name, :].values).to(torch.float32)
        radio_wavelet = torch.from_numpy(self.wavelet.loc[patient_name, :].values).to(torch.float32)

        # RadFM data
        radio_cnn = torch.load(os.path.join(self.radio_cnn_path, f"{patient_name}.pt")).squeeze(0).to(torch.float32)

        patho = torch.load(os.path.join(self.pathomics_base, f"{patient_name}.pt")).to(torch.float32)
        data = [patho, radio_shape, radio_order, radio_texture, radio_log_sigma, radio_wavelet, radio_cnn, patient_name]

        return data


# get radiomics from union training
task = args.task
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold in range(5):
    # initialize dataset and put into dataloader
    dataset = UniData(task=task)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    print(f"Use the checkpoint from fold {fold}!")
    ckpt_path = f"./union_training/ckpts/fold_{fold}.pth"

    model = RadPathModel().to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # save base
    save_pathomics = f"./data/{task}/TCGA-{task}/coattn_pathomics/fold_{fold}/"
    save_radiomics = f"./data/{task}/TCGA-{task}/coattn_radiomics/fold_{fold}/"
    os.makedirs(save_pathomics, exist_ok=True)
    os.makedirs(save_radiomics, exist_ok=True)

    # make predictions
    with torch.no_grad():
        model.eval()
        for batch_idx, data in tqdm(enumerate(loader)):
            patho, radio_shape, radio_order, radio_texture, radio_log_sigma,\
            radio_wavelet, radio_cnn, patient_name = data

            data_WSI = patho.to(device)
            data_omic1 = radio_shape.type(torch.FloatTensor).to(device)
            data_omic2 = radio_order.type(torch.FloatTensor).to(device)
            data_omic3 = radio_texture.type(torch.FloatTensor).to(device)
            data_omic4 = radio_log_sigma.type(torch.FloatTensor).to(device)
            data_omic5 = radio_wavelet.type(torch.FloatTensor).to(device)
            data_omic6 = radio_cnn.type(torch.FloatTensor).to(device)

            pathomics, radiomics = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2,
                                         x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5,
                                         x_omic6=data_omic6, patient_name=patient_name)

            torch.save(pathomics.squeeze(1).cpu(), os.path.join(save_pathomics, f"{patient_name[0]}.pt"))
            torch.save(radiomics.squeeze(1).cpu(), os.path.join(save_radiomics, f"{patient_name[0]}.pt"))
