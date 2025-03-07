import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F

from src.rmrm.model.intialisation import init_weights   

class RMRM(torch.nn.Module):
    def __init__(self, num_node_features):
        super(RMRM, self).__init__()
        self.conv1 = GATConv(num_node_features, 8, heads=8)
        self.conv2 = GATConv(8*8, num_node_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x
        
class OutputBlock(nn.Module):
    def __init__(self, in_ch, n_features, n_classes=1):
        super(OutputBlock, self).__init__()
        self.n_features = n_features

        for i in range(1, n_features+1):
            feature_mlp = nn.Sequential(
                nn.Linear(in_ch, 512, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(512, n_classes, bias=False),
                nn.Sigmoid()
            )
            setattr(self, 'feature_mlp_%d' %i, feature_mlp)

        for m in self.children():
            init_weights(m, init_type='glorot')

    def forward(self, inputs):
        if len(inputs.shape) < 2:
            inputs = inputs.unsqueeze(0)
           
        pred_all = None
            
        for i in range(1, self.n_features+1):
            mlp = getattr(self, 'feature_mlp_%d' %i)
            pred = mlp(inputs)

            if pred_all is None:
                pred_all = pred.clone()
            else:
                pred_all = torch.cat((pred_all, pred), 1)
        
        return pred_all
