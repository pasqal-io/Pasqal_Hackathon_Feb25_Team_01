import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GNN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_classes):
        """
        GNN for node classification based on clinical and image embeddings.

        Parameters:
        - input_channels: Number of features per node (sum of clinical + image embeddings)
        - hidden_channels: Hidden dimension for GCN layers
        - output_classes: Number of output classes for classification (binary or multi-class)
        """
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # Final classifier
        x = F.dropout(x, p=0.5, training=self.training) # Dropout for regularization
        x = self.lin(x)
        
        return x

# 522 --> image + clinical features
# 2 classes for binary classification (cancer/no cancer)
model = GNN(input_channels=522, hidden_channels=64, output_classes=2)
print(model)