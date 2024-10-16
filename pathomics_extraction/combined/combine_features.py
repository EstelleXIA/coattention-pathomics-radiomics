import json
import os
import pandas as pd
import torch
from tqdm import tqdm
import __main__
from torch_geometric.data import Data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="the task name")
args = parser.parse_args()


class NucleiData(Data):
    """Add some attributes to Data object.

    Args:
        * All args mush be torch.tensor. So string is not supported.
        x: Matrix for nodes
        cell_type: cell type
        edge_index: 2*N matrix
        edge_attr: edge type
        region_x: x coordinate
        region_y: y coordinate
    """

    def __init__(self, x=None, cell_type=None, edge_index=None, edge_attr=None, pos=None,
                 wsi_id=None, region_x=None, region_y=None):
        super().__init__(x, edge_index, edge_attr, pos)
        self.cell_type = cell_type
        self.wsi_id = wsi_id
        self.region_x = region_x
        self.region_y = region_y

    def __repr__(self):
        info = ['{}={}'.format(key, self.size_repr(item)) for key, item in self]
        return '{}({})'.format(self.__class__.__name__, ', '.join(info))

    @staticmethod
    def size_repr(value):
        if torch.is_tensor(value):
            return list(value.size())
        elif isinstance(value, int) or isinstance(value, float) or isinstance(value, str):
            return [1]
        else:
            raise ValueError('Unsupported attribute type.')


setattr(__main__, "NucleiData", NucleiData)


task = args.task
graph_path = f"./data/{task}/TCGA-{task}/graph_pt/"
cellular_path = f"./data/{task}/TCGA-{task}/patch_features_cellular/"
morph_path = f"./data/{task}/TCGA-{task}/patch_features_resnet/"
json_path = f"./data/{task}/TCGA-{task}/resnet_json/"

save_merged_features = f"./data/{task}/TCGA-{task}/patch_features_merged/"
os.makedirs(save_merged_features, exist_ok=True)

patients = sorted([x.replace(".pt", "") for x in os.listdir(cellular_path)])

for patient in tqdm(patients):
    cnn_data = torch.load(os.path.join(morph_path, f"{patient}.pt"))
    with open(os.path.join(json_path, f"{patient}.json"), "r") as f:
        order = json.load(f)
    cnn_order = [x[13:].replace(".jpg", "") for x in order]

    cnn_pd = pd.DataFrame(cnn_data, index=cnn_order).reset_index()

    graph_data = torch.load(os.path.join(graph_path, f"{patient}_graph.pt"))
    cellular_data = torch.load(os.path.join(cellular_path, f"{patient}.pt"))
    graph_order = []
    for graph in graph_data:
        wsi_id, region_x, region_y = graph.wsi_id.item(), graph.region_x.item(), graph.region_y.item()
        graph_order.append(f"{wsi_id}_{region_x}-{region_y}")

    cellular_pd = pd.DataFrame(cellular_data, index=graph_order).reset_index()
    merged_features = pd.merge(cnn_pd, cellular_pd, on="index").set_index("index").values

    torch.save(torch.from_numpy(merged_features), os.path.join(save_merged_features, f"{patient}.pt"))

