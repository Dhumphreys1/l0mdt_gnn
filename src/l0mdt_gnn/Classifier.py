import torch
from torch import nn
from torch_geometric.nn import MessagePassing

class EdgeNodeLayer(MessagePassing):
    def __init__(self, node_in, edge_in, hidden, dropout=0.0):
        super().__init__(aggr='add')

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_in + edge_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(node_in + hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_mlp(msg_input)

    def update(self, aggr_out, x):
        node_input = torch.cat([x, aggr_out], dim=-1)
        return self.node_mlp(node_input)

class SimpleEdgeGNN(nn.Module):
    def __init__(self, node_in, edge_in, hidden, out_dim, dropout=0.0):
        super().__init__()
        self.layer1 = EdgeNodeLayer(node_in, edge_in, hidden, dropout)
        self.layer2 = EdgeNodeLayer(hidden, edge_in, hidden, dropout)

        self.output_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.layer1(x, edge_index, edge_attr)
        x = self.layer2(x, edge_index, edge_attr)
        return self.output_mlp(x)

