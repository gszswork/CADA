from torch_geometric.nn import GCNConv
import copy
import sys,os
sys.path.append(os.getcwd())
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from model.rgl import ReverseLayerF

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

class TDrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1=copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2=copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x=F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)
        x= scatter_mean(x, data.batch, dim=0)
        return x
    
class BUrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)
        x= scatter_mean(x, data.batch, dim=0)
        return x


class bigcn_feature_extractor(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(bigcn_feature_extractor, self).__init__()
        self.conv1 = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.conv2 = BUrumorGCN(in_feats, hid_feats, out_feats)

    def forward(self, data):
        TD_x = self.conv1(data)
        BU_x = self.conv2(data)
        x = th.cat((BU_x,TD_x), 1)
        return x
    

class bigcn_label_predictor(th.nn.Module):
    #  The label_predictor of the CADA model. It's originally the fully connected layer in the bigcn model. 
    def __init__(self, feat_dim, class_dim):
        super(bigcn_label_predictor, self).__init__()
        self.fc=th.nn.Linear(feat_dim,class_dim)
    def forward(self, x):
        x=self.fc(x)
        return x
    
class bigcn_domain_classifier(th.nn.Module):
    # The domain_classifier of the CADA model. It's in the same design as the label_predictor.
    # Need to add the gradient reversal layer.
    def __init__(self, feat_dim, domain_dim):
        super(bigcn_domain_classifier, self).__init__()
        self.fc=th.nn.Linear(feat_dim, domain_dim)

    def forward(self, x, alpha):
        # How to calculate alpha:
        # p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        # alpha = 2. / (1. + np.exp(-10 * p)) - 1
        x = ReverseLayerF.apply(x, alpha)
        x=self.fc(x)
        return x
    


    