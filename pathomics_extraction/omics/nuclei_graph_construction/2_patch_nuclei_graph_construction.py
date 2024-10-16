import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import os
import pickle
import numpy as np
import pandas as pd
import sklearn.neighbors as skgraph
from scipy import sparse as sp
import torch
from torch_geometric.data import Data
from multiprocessing import Pool
from tqdm import tqdm
import tempfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="the task name")
parser.add_argument("--idx", default=None, help="to fasten the histomics extraction process")

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


def get_edge_type(edge, cell_type):
    """
    Args:
        edge: (in_cell_index, out_cell_index, 1/edge_length)
    Returns:
        edge type index.
    """
    mapping = {"0-0": 0, "0-1": 1, "0-2": 2, "0-3": 3, "0-4": 4, "0-5": 5,
               "1-0": 6, "1-1": 7, "1-2": 8, "1-3": 9, "1-4": 10, "1-5": 11,
               "2-0": 12, "2-1": 13, "2-2": 14, "2-3": 15, "2-4": 16, "2-5": 17,
               "3-0": 18, "3-1": 19, "3-2": 20, "3-3": 21, "3-4": 22, "3-5": 23,
               "4-0": 24, "4-1": 25, "4-2": 26, "4-3": 27, "4-4": 28, "4-5": 29,
               "5-0": 30, "5-1": 31, "5-2": 32, "5-3": 33, "5-4": 34, "5-5": 35}
    return mapping['{}-{}'.format(cell_type[edge[0]], cell_type[edge[1]])]


def get_nuclei_orientation_diff(edge, nuclei_orientation):
    return np.cos(nuclei_orientation[edge[0]] - nuclei_orientation[edge[1]])


task = args.task

histo_feature_path = f"./data/{task}/TCGA-{task}/histo_features/"
save_base = f"./data/{task}/TCGA-{task}/graph_pt/"
os.makedirs(save_base, exist_ok=True)
patients = os.listdir(histo_feature_path)

norm_mean = pd.read_csv("./pathomics_extraction/omics/nuclei_graph_construction/"
                        f"feature_mean_std/{task}_mean.csv", index_col=0)
norm_std = pd.read_csv("./pathomics_extraction/omics/nuclei_graph_construction/"
                       f"feature_mean_std/{task}_std.csv", index_col=0)

feat_corr_path = f"./pathomics_extraction/omics/nuclei_graph_construction/remove_feature_list/" \
                 f"{task}_removed_feature_list.txt"

with open(feat_corr_path, "r") as file:
    removed_feature_list = [line.strip() for line in file]
print("Remove feature number:", len(removed_feature_list))

drop_cols = ["Type", "Label", "Identifier.CentroidX", "Identifier.CentroidY"] + removed_feature_list


def generate_graph_data(file_name):
    patch_summary = pd.read_csv(os.path.join(histo_feature_path, file_name.split("_")[0], file_name))

    # create 8 nearest neighbors graph
    graph = skgraph.kneighbors_graph(np.array(patch_summary.loc[:, ['Identifier.CentroidX',
                                                                    'Identifier.CentroidY']]),
                                     n_neighbors=8, mode='distance')
    I, J, V = sp.find(graph)
    edges = list(zip(I, J, 1 / V))
    edge_index = np.transpose(np.array(edges)[:, 0:2])

    # feature normalization
    x_feat = (np.array(patch_summary.drop(drop_cols, axis=1)) -
              norm_mean.drop(drop_cols, axis=0).values[:, 0]) / norm_std.drop(drop_cols, axis=0).values[:, 0]

    cell_type = np.array(patch_summary["Type"])
    orientation = np.array(patch_summary["Orientation.Orientation"])

    # edge features
    edge_type = list(map(lambda x: get_edge_type(x, cell_type), edges))
    nuclei_orientation = list(map(lambda x: get_nuclei_orientation_diff(x, orientation), edges))
    edge_attr = np.transpose(np.array([edge_type, nuclei_orientation, 1 / V]))

    wsi_num = int(file_name.split("_")[1])
    coordinate_x = int(file_name.replace(".csv", "").split("_")[2].split("-")[0])
    coordinate_y = int(file_name.replace(".csv", "").split("_")[2].split("-")[1])

    # generate single patch graph data
    data = NucleiData(x=torch.tensor(x_feat, dtype=torch.float),
                      cell_type=torch.tensor(cell_type, dtype=torch.long),
                      edge_index=torch.tensor(edge_index, dtype=torch.long),
                      edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                      wsi_id=torch.tensor([wsi_num]),
                      region_x=torch.tensor([coordinate_x]),
                      region_y=torch.tensor([coordinate_y]))
    if not (torch.isinf(torch.tensor(x_feat, dtype=torch.float)).any() or torch.isnan(torch.tensor(x_feat, dtype=torch.float)).any()):
        return data


if args.idx:
    process_patients = [patients[args.idx]]
else:
    process_patients = patients

for patient in tqdm(process_patients):
    all_patches = os.listdir(os.path.join(histo_feature_path, patient))
    p = Pool(8)
    if len(all_patches) > 6000:
        each = 1000
        total = len(all_patches) // each + int(len(all_patches) % each != 0)
        prep_dataset = []

        with tempfile.TemporaryDirectory() as tmpdirname:
            for i in list(range(total)):
                torch.save(p.map(generate_graph_data, all_patches[each * i: each * (i + 1)]),
                           os.path.join(tmpdirname, f"{patient}_graph_{i}.pt"),
                           pickle_protocol=pickle.HIGHEST_PROTOCOL)

            for i in list(range(total)):
                prep_dataset.extend(torch.load(os.path.join(tmpdirname, f"{patient}_graph_{i}.pt")))

    else:
        prep_dataset = p.map(generate_graph_data, all_patches)

    torch.save(list(filter(lambda x: x is not None, prep_dataset)), os.path.join(save_base, f"{patient}_graph.pt"),
               pickle_protocol=pickle.HIGHEST_PROTOCOL)
