import torch
import numpy as np
from torch.nn import AvgPool2d
from torch_geometric.data import Data
from src.rmrm.model.components import RMRM, OutputBlock

class Network(torch.nn.Module):
    def __init__(self, opts, n_out_features, batch_size, device):
        """
        Network for binary classification based on clinical and image embeddings.

        Parameters:
        - opts: model options
        - n_out_features: Number of output classes for classification (1 for binary classification)
        """
        super(Network, self).__init__()

        self.fv_dim = 128           # Embedding dimention

        self.n_clinical = 38        # Number of features in clinical embeddings
        self.n_pixel = 6*6          # Number of image regions
        self.n_nodes = self.n_clinical + self.n_pixel

        self.batch_size = batch_size
        self.edge_index = self.get_edges(self.n_clinical, self.n_nodes).to(device)

        # Region-based multimodal relational module (RMRM)
        self.graph_net = RMRM(self.fv_dim)

        # Global Average Pooling (GAP) for Image Embeddings
        self.gap = AvgPool2d(kernel_size=(6,6))

        # Output MLP for binary classification (0 for censored, 1 for progressed)
        self.output_mlp = OutputBlock(self.fv_dim * (self.n_clinical + 1), n_out_features)

    def get_edges(self, n_clinical, n_nodes):
        node_ids = np.expand_dims(np.arange(n_nodes, dtype=int), 0)
        self_edges = np.concatenate((node_ids, node_ids), 0)

        c_array_asc = np.expand_dims(np.arange(n_clinical), 0)
        all_edges = self_edges[:]

        for i in range(n_clinical, n_nodes):
            i_array = np.expand_dims(np.array([i]*n_clinical), 0)
            inter_edges_ic = np.concatenate((i_array, c_array_asc), 0)
            inter_edges_ci = np.concatenate((c_array_asc, i_array), 0)
            inter_edges_i = np.concatenate((inter_edges_ic, inter_edges_ci), 1)
            all_edges = np.concatenate((all_edges, inter_edges_i), 1)

        return torch.tensor(all_edges, dtype=torch.long)

    def forward(self, clinical_embeddings, image_embeddings):
        """
        clinical_embeddings: Pre-trained clinical embeddings [batch_size, n_clinical, fv_dim]
        image_embeddings: Pre-trained image embeddings [batch_size, n_pixel, img_dim]
        """

        # Ensure image embeddings match clinical feature dimension
        image_embeddings = image_embeddings.view(self.batch_size, self.n_pixel, -1)

        # Merge embeddings for graph input
        batch_semantic_fvs = torch.cat((clinical_embeddings, image_embeddings), dim=1)
        batch_semantic_fvs = batch_semantic_fvs.view(-1, self.fv_dim)

        # Process graph data
        batch_edge_index = self.edge_index.clone()
        for ind in range(1, self.batch_size):
            next_edge_index = self.edge_index + self.n_nodes * ind
            batch_edge_index = torch.cat((batch_edge_index, next_edge_index), 1)

        data = Data(x=batch_semantic_fvs, edge_index=batch_edge_index)
        batch_graph_fvs = self.graph_net(data)

        # Reshape RMRM Outputs
        for ind in range(self.batch_size):
            if ind == 0:
                graph_fvs_c = batch_graph_fvs[:self.n_clinical, :].unsqueeze(0)
                graph_fvs_i = batch_graph_fvs[self.n_clinical:self.n_clinical+self.n_pixel, :].unsqueeze(0)
            else:
                graph_fvs_c = torch.cat((graph_fvs_c, 
                                         batch_graph_fvs[ind*self.n_nodes:ind*self.n_nodes+self.n_clinical, :].unsqueeze(0)), 
                                         0)
                graph_fvs_i = torch.cat((graph_fvs_i, 
                                         batch_graph_fvs[ind*self.n_nodes+self.n_clinical:ind*self.n_nodes+self.n_clinical+self.n_pixel, :].unsqueeze(0)), 
                                         0)

        # Apply GAP to image graph features
        graph_fvs_i = torch.transpose(graph_fvs_i, 1, 2)
        gap = self.gap(graph_fvs_i.view(self.batch_size, self.fv_dim, 6, 6)).squeeze(-1).squeeze(-1)

        # Combine Clinical and Image Features
        combined = torch.cat((graph_fvs_c, gap.unsqueeze(1)), 1)
        combined = combined.view(self.batch_size, -1)

        # Survival Prediction
        feature_preds = self.output_mlp(combined)

        return feature_preds