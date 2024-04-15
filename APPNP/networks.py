import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import APPNP

class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.num_layers = args.num_layers
        self.conv = torch.nn.ModuleList()
        self.conv.append(Linear(args.num_features, args.hidden))
        for i in range(self.num_layers - 2):
            self.conv.append(Linear(args.hidden, args.hidden))
        self.conv.append(Linear(args.hidden, args.num_classes))
        # self.conv.append(GCNConv(args.hidden, args.num_classes))
        # self.lin1 = Linear(args.num_features, args.hidden)
        # self.lin2 = Linear(args.hidden, args.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)

    def reset_parameters(self):
        for module in self.conv:
            module.reset_parameters()

    def forward(self, x, edge_index):
        for i in range(self.num_layers-1):
            x = F.dropout(x, training=self.training)
            x = F.relu(self.conv[i](x))
            x = F.dropout(x, training=self.training)
            x = self.conv[i+1](x)
        x = self.prop1(x, edge_index)

        return F.log_softmax(x, dim=1)