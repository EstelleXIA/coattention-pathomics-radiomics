from torchvision.models import resnet34
import torch
from torch.utils.data import DataLoader, Dataset
import os
import json
from tqdm import tqdm
import skimage.io
from torchvision import transforms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="the task name")
args = parser.parse_args()


class ResNetDataset(Dataset):
    def __init__(self, task_id, patient_id):
        super(ResNetDataset, self).__init__()
        json_path = f"./data/{task_id}/TCGA-{task_id}/resnet_json/"
        with open(os.path.join(json_path, f"{patient_id}.json"), "r") as f:
            self.files = json.load(f)
        self.base = f"./data/{task_id}/TCGA-{task_id}/patches/"
        self.path = os.path.join(self.base, patient_id)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        trans_norm = transforms.Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])
        img_path = os.path.join(self.path, self.files[item])
        img = skimage.io.imread(img_path).transpose(2, 0, 1)
        img = torch.from_numpy(img).float() / 255
        img_norm = trans_norm(img)
        return img_norm


task = args.task
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet34(pretrained=True)
extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
extractor.to(device)
extractor.eval()

save_feat_path = f"./data/{task}/TCGA-{task}/patch_features_resnet/"
os.makedirs(save_feat_path, exist_ok=True)
all_jsons = os.listdir(f"./data/{task}/TCGA-{task}/resnet_json/")
patients = sorted([x.replace(".json", "") for x in all_jsons])

for patient in tqdm(patients):
    input_data = ResNetDataset(task_id=task, patient_id=patient)
    input_loader = DataLoader(dataset=input_data, batch_size=64, shuffle=False,
                              num_workers=0, drop_last=False)
    output = []
    for idx, data in tqdm(enumerate(input_loader)):
        with torch.no_grad():
            out = extractor(data.to(device))
            output.append(out.squeeze(dim=2).squeeze(dim=2))
    output = torch.cat(output, dim=0)
    torch.save(output.detach().cpu(), os.path.join(save_feat_path, f"{patient}.pt"))



