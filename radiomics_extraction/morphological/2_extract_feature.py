import sys
sys.path.append("./radiomics_extraction/morphological/RadFM/Quick_demo")
from Model.RadFM.vit_3d import ViT
import torch
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import SimpleITK as sitk
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="the task name")
args = parser.parse_args()


class RadFMDataset(Dataset):
    def __init__(self, roi_path):
        super(RadFMDataset, self).__init__()
        self.path = roi_path
        self.files = os.listdir(self.path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        img_path = os.path.join(self.path, self.files[item])
        img_array = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        img_array = img_array.astype(float)
        img = torch.from_numpy(img_array.transpose([1, 2, 0]))
        img_min, img_max = img.min(), img.max()
        img = (torch.stack((img, img, img), dim=0).float() - img_min) / (img_max - img_min)
        return self.files[item], img


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task = args.task
img_roi_path = f"./data/{task}/TCIA-{task}/img_nii_roi/"
save_base_roi = f"./data/{task}/TCIA-{task}/img_nii_features_per_roi/"
save_base_patient = f"./data/{task}/TCIA-{task}/img_nii_features/"
os.makedirs(save_base_roi, exist_ok=True)
os.makedirs(save_base_patient, exist_ok=True)

model = ViT(
    image_size=512,  # image size
    frames=512,  # max number of frames
    image_patch_size=32,  # image patch size 32*32*12 -> 768
    frame_patch_size=4,  # frame patch size
    dim=768,  # embedding size
    depth=12,
    heads=8,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

ckpt = torch.load("./radiomics_extraction/morphological/pytorch_model.bin", map_location=device)
load_ckpt = {key.replace("embedding_layer.vision_encoder.", ""): value for key, value in ckpt.items()}
model.load_state_dict(load_ckpt, strict=False)
model = model.to(device)
model.eval()

input_data = RadFMDataset(roi_path=img_roi_path)
input_loader = DataLoader(dataset=input_data, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

for file_name, vision_x in tqdm(input_loader):
    with torch.no_grad():
        # B*C*H*W*D, should have min_size=32*32*12
        vision_embed, pos_embed = model(vision_x.to(device))
        vision_embed_final = vision_embed.mean(dim=1).detach().cpu()
        torch.save(vision_embed_final, os.path.join(save_base_roi, file_name[0].replace(".nii.gz", ".pt")))

# combine roi results
patients = sorted(list(set([x.split("_")[0] for x in os.listdir(save_base_roi)])))
for patient in patients:
    patient_feature_files = list(filter(lambda x: x.startswith(patient), os.listdir(save_base_roi)))
    patient_features = []
    for feature_file in patient_feature_files:
        patient_features.append(torch.load(os.path.join(save_base_roi, feature_file)))
    torch.save(torch.mean(torch.stack(patient_features), dim=0), os.path.join(save_base_patient, f"{patient}.pt"))

