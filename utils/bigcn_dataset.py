import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class GraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=10000000, droprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.droprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))

def collate_fn(data):
    return data

class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, tddroprate=0,budroprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph'), domain_y=None):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate
        self.domain_y = domain_y

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        seqlen = torch.LongTensor([(data['seqlen'])]).squeeze()
        #seqlen = torch.from_numpy(data['seqlen']).squeeze()
        # The correct way to create a Tensor from a numpy array is to use:
        # tensor = torch.from_numpy(array)

        #x_tokens_feat=list(data['x'])
        #x = np.zeros([len(x_tokens_feat), max_len, 768])
        #for item in range(len(x_tokens_feat)):
        #    for i in range(x_tokens_feat[item].size()[0]):
        #        x[item][i] = x_tokens_feat[item][i].detach().numpy()


        x_features=torch.tensor([item.detach().numpy() for item in list(data['x'])],dtype=torch.float32)
        #x_features = torch.stack([torch.from_numpy(item.detach().cpu().numpy()) for item in list(data['x'])]).type(torch.float32)

        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]

        if self.domain_y is not None:
            ret_data = Data(x=x_features,
                        edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
                y=torch.LongTensor([int(data['y'])]), root=torch.from_numpy(data['root']),
                rootindex=torch.LongTensor([int(data['rootindex'])]), seqlen=seqlen, domain_y=torch.LongTensor([self.domain_y]))
        else: 
            ret_data = Data(x=x_features,
                        edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
                y=torch.LongTensor([int(data['y'])]), root=torch.from_numpy(data['root']),
                rootindex=torch.LongTensor([int(data['rootindex'])]), seqlen=seqlen)
        return ret_data


class UdGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..','..','data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row.extend(burow)
        col.extend(bucol)
        if self.droprate > 0:
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
        new_edgeindex = [row, col]

        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))

def my_loadTree(dataname):
    if dataname == 'Twitter':
        id_treePath = os.path.join(os.getcwd(),'data/in-domain/Twitter/Twitter_data_all.txt')
        ood_treePath = os.path.join(os.getcwd(),'data/out-of-domain/Twitter/Twitter_data_all.txt')
        id_treeDic = {}
        for line in open(id_treePath):
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            time_span, Vec = float(line.split('\t')[3]), line.split('\t')[4]
            #max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
            if not id_treeDic.__contains__(eid):
                id_treeDic[eid] = {}
            id_treeDic[eid][indexC] = {'parent': indexP, 'time': time_span, 'vec': Vec}

        ood_treeDic = {}
        for line in open(ood_treePath):
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            #max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
            time_span, Vec = float(line.split('\t')[3]), line.split('\t')[4]
            if not ood_treeDic.__contains__(eid):
                ood_treeDic[eid] = {}
            ood_treeDic[eid][indexC] = {'parent': indexP, 'time': time_span, 'vec': Vec}
    
    return id_treeDic, ood_treeDic