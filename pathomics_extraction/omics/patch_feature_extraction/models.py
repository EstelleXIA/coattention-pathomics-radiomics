from torch_geometric.nn import NNConv
from torch_scatter import scatter_mean
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class EdgeNN(nn.Module):
    """
    Design: embedding according to edge type, and then modulated by edge features.
    """

    def __init__(self, in_channels, out_channels, n_edge_types=36):
        super(EdgeNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.edge_type_embedding = nn.Embedding(n_edge_types, out_channels)
        self.fc_h = nn.Linear(in_channels, out_channels)
        self.fc_g = nn.Linear(in_channels, out_channels)
        # self.bn = nn.BatchNorm(out_channels, eps=0.001)

    def forward(self, x):
        """
        Args:
            x: [batch_size, n_edges, 1(edge type) + in_channels]

        return: [batch_size, out_channels]
        """
        y = self.edge_type_embedding(x[..., 0].clone().detach().type(torch.long))
        h = self.fc_h(x[..., 1:(self.in_channels + 1)].clone().detach().type(self.fc_h.weight.dtype))
        g = self.fc_g(x[..., 1:(self.in_channels + 1)].clone().detach().type(self.fc_g.weight.dtype))
        y = y * h + g
        # x = self.bn(x)
        return F.relu(y, inplace=True)


class CellSpatialNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, batch=True, edge_features=2, n_edge_types=36):
        """
        Args:
            in_channels: No. of node features
            out_channels: No. of output
            batch: True if from DataLoader; False if single Data object
            edge_features: No. of edge features (excluding edge type)
            n_edge_types: No. of edge types
        """
        super(CellSpatialNet, self).__init__()
        self.batch = batch

        hidden_out_channel = 8
        conv_out_channel = 64
        self.conv1 = NNConv(in_channels, hidden_out_channel,
                            EdgeNN(edge_features, in_channels * hidden_out_channel, n_edge_types=n_edge_types),
                            aggr='mean', root_weight=True, bias=True)
        self.conv2 = NNConv(hidden_out_channel, hidden_out_channel,
                            EdgeNN(edge_features, hidden_out_channel * hidden_out_channel, n_edge_types=n_edge_types),
                            aggr='mean', root_weight=True, bias=True)
        self.conv3 = NNConv(hidden_out_channel, hidden_out_channel,
                            EdgeNN(edge_features, hidden_out_channel * hidden_out_channel, n_edge_types=n_edge_types),
                            aggr='mean', root_weight=True, bias=True)
        self.conv4 = NNConv(hidden_out_channel, conv_out_channel,
                            EdgeNN(edge_features, hidden_out_channel * conv_out_channel, n_edge_types=n_edge_types),
                            aggr='mean', root_weight=True, bias=True)

        self.classifier = nn.Linear(conv_out_channel, out_channels)

    def forward(self, data):
        """
        Args:
            data: Data in torch_geometric.data
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index, edge_attr)
        x = F.relu(x)
        '''
        if self.batch:
            x = global_max_pool(x, batch=data.batch).to(device)
        else:
            x = global_max_pool(x, batch=torch.tensor(np.zeros(x.shape[0]), dtype=torch.long).to(device))
        '''
        gate = torch.eq(data.cell_type, 1).clone().detach().requires_grad_(False).type(torch.long)
        if self.batch:
            _batch_size = data.batch[-1] + 1
            x = scatter_mean(x, gate * (data.batch + 1), dim=0)[1:_batch_size + 1, :]  # Keep the batches
        else:
            x = scatter_mean(x, gate, dim=0)[1, :]

        logits = self.classifier(x)
        hazards = torch.sigmoid(logits)
        return hazards


class CellSpatialNetInfer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, batch=True, edge_features=2, n_edge_types=36):
        """
        Args:
            in_channels: No. of node features
            out_channels: No. of output node features (e.g., No. classes for classification)
            batch: True if from DataLoader; False if single Data object
            edge_features: No. of edge features (excluding edge type)
            n_edge_types: No. of edge types
        """
        super(CellSpatialNetInfer, self).__init__()
        self.batch = batch

        hidden_out_channel = 8
        conv_out_channel = 64
        self.conv1 = NNConv(in_channels, hidden_out_channel,
                            EdgeNN(edge_features, in_channels * hidden_out_channel, n_edge_types=n_edge_types),
                            aggr='mean', root_weight=True, bias=True)
        self.conv2 = NNConv(hidden_out_channel, hidden_out_channel,
                            EdgeNN(edge_features, hidden_out_channel * hidden_out_channel, n_edge_types=n_edge_types),
                            aggr='mean', root_weight=True, bias=True)
        self.conv3 = NNConv(hidden_out_channel, hidden_out_channel,
                            EdgeNN(edge_features, hidden_out_channel * hidden_out_channel, n_edge_types=n_edge_types),
                            aggr='mean', root_weight=True, bias=True)
        self.conv4 = NNConv(hidden_out_channel, conv_out_channel,
                            EdgeNN(edge_features, hidden_out_channel * conv_out_channel, n_edge_types=n_edge_types),
                            aggr='mean', root_weight=True, bias=True)

        self.classifier = nn.Linear(conv_out_channel, out_channels)

    def forward(self, data):
        """
        Args:
            data: Data in torch_geometric.data
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index, edge_attr)
        x = F.relu(x)
        '''
        if self.batch:
            x = global_max_pool(x, batch=data.batch).to(device)
        else:
            x = global_max_pool(x, batch=torch.tensor(np.zeros(x.shape[0]), dtype=torch.long).to(device))
        '''
        gate = torch.eq(data.cell_type, 1).clone().detach().requires_grad_(False).type(torch.long)
        if self.batch:
            _batch_size = data.batch[-1] + 1
            x = scatter_mean(x, gate * (data.batch + 1), dim=0)[1:_batch_size + 1, :]  # Keep the batches
        else:
            x = scatter_mean(x, gate, dim=0)[1, :]

        logits = self.classifier(x)
        hazards = torch.sigmoid(logits)
        return x, hazards


class CellSpatialNetCaptum(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_features=2, n_edge_types=36):
        super(CellSpatialNetCaptum, self).__init__()

        hidden_out_channel = 8
        conv_out_channel = 64
        self.conv1 = NNConv(in_channels, hidden_out_channel,
                            EdgeNN(edge_features, in_channels * hidden_out_channel, n_edge_types=n_edge_types),
                            aggr='mean', root_weight=True, bias=True)
        self.conv2 = NNConv(hidden_out_channel, hidden_out_channel,
                            EdgeNN(edge_features, hidden_out_channel * hidden_out_channel, n_edge_types=n_edge_types),
                            aggr='mean', root_weight=True, bias=True)
        self.conv3 = NNConv(hidden_out_channel, hidden_out_channel,
                            EdgeNN(edge_features, hidden_out_channel * hidden_out_channel, n_edge_types=n_edge_types),
                            aggr='mean', root_weight=True, bias=True)
        self.conv4 = NNConv(hidden_out_channel, conv_out_channel,
                            EdgeNN(edge_features, hidden_out_channel * conv_out_channel, n_edge_types=n_edge_types),
                            aggr='mean', root_weight=True, bias=True)

        self.classifier = nn.Linear(conv_out_channel, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index, edge_attr)
        x = F.relu(x)
        x_feat = torch.mean(x, dim=1)
        return x_feat

