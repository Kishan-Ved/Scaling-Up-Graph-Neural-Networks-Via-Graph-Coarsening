import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.num_layers = args.num_layers
        self.conv = torch.nn.ModuleList()
        self.conv.append(GCNConv(args.num_features, args.hidden))
        for i in range(self.num_layers - 2):
            self.conv.append(GCNConv(args.hidden, args.hidden))
        self.conv.append(GCNConv(args.hidden, args.num_classes))

    def reset_parameters(self):
        for module in self.conv:
            module.reset_parameters()

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.conv[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        return F.log_softmax(x, dim=1)