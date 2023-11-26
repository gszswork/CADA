from utils.train_test_split import get_id_ood_twitter_ids
from utils.bigcn_dataset import BiGraphDataset, my_loadTree
from model.bigcn_module import * 
from torch_geometric.data import DataLoader
from CADA import CADA
import torch
import torch.nn.functional as F
import random

# Dataset: in-domain Twitter, size 1154; out-of-domain Twitter-COVID19, size 400.

#   'In the target event, we set the declarations size m to 100, and the remaining declarations serve
# as the test set to evaluate our framework.'
#                   train, test
# in-domain:        923, 230; 
# out-of-domain:    100, 300. 

id_train_num = 923
id_test_num = 230
ood_train_num = 100
ood_test_num = 300
bs = 16
n_epochs = 0


if __name__ == '__main__':
    dataname = 'Twitter'
    # Load the dataset. 

    id_twitter_ids, ood_twitter_ids = get_id_ood_twitter_ids()
    random.shuffle(id_twitter_ids)
    random.shuffle(ood_twitter_ids)
    id_train_ids = id_twitter_ids[:id_train_num]
    id_test_ids = id_twitter_ids[:-id_test_num]
    ood_train_ids = ood_twitter_ids[:ood_train_num]
    ood_test_ids = ood_twitter_ids[:-ood_test_num]

    print(len(id_twitter_ids), len(ood_twitter_ids))
    id_treeDic, ood_treeDic = my_loadTree('Twitter')
    #print(len(id_treeDic), len(ood_treeDic))

    in_data_path = os.path.join(os.getcwd(), 'data', 'in-domain', dataname + 'graph')
    ood_data_path = os.path.join(os.getcwd(), 'data', 'out-of-domain', dataname + 'graph')

    id_train_dataset = BiGraphDataset(id_train_ids, id_treeDic, lower=2, upper=100000, tddroprate=0.2, budroprate=0.2, data_path=in_data_path, domain_y=0)
    id_test_dataset = BiGraphDataset(id_test_ids, id_treeDic, lower=2, upper=100000, tddroprate=0.2, budroprate=0.2, data_path=in_data_path, domain_y=0)
    ood_train_dataset = BiGraphDataset(ood_train_ids, ood_treeDic, lower=2, upper=100000, tddroprate=0.2, budroprate=0.2, data_path=ood_data_path, domain_y=1)
    ood_test_dataset = BiGraphDataset(ood_test_ids, ood_treeDic, lower=2, upper=100000, tddroprate=0.2, budroprate=0.2, data_path=ood_data_path, domain_y=1)

    print(len(id_train_dataset), len(id_test_dataset), len(ood_train_dataset), len(ood_test_dataset))

    id_train_dataloader = DataLoader(id_train_dataset, batch_size=bs, shuffle=True)
    id_test_dataloader = DataLoader(id_test_dataset, batch_size=bs, shuffle=True)
    ood_train_dataloader = DataLoader(ood_train_dataset, batch_size=bs, shuffle=True)
    ood_test_dataloader = DataLoader(ood_test_dataset, batch_size=bs, shuffle=True)


    second_train_dataset = id_train_dataset + ood_train_dataset
    second_train_dataloader = DataLoader(second_train_dataset, batch_size=bs, shuffle=True)

    # Splice the model. 
    in_dim = 768
    hid_dim = 128
    out_dim = 64
    bigcn_feature_extractor = bigcn_feature_extractor(in_dim, hid_dim, out_dim)
    bigcn_label_predictor = bigcn_label_predictor((out_dim+hid_dim)*2, 2)
    bigcn_domain_classifier_0 = bigcn_domain_classifier((out_dim+hid_dim)*2, 2)
    bigcn_domain_classifier_1 = bigcn_domain_classifier((out_dim+hid_dim)*2, 2)
    bigcn_domain_classifier_list = [bigcn_domain_classifier_0, bigcn_domain_classifier_1]
    model = CADA(bigcn_feature_extractor, bigcn_label_predictor, bigcn_domain_classifier_list)

    # pre-train the model on in-domain data. 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    avg_loss, avg_acc = [], []
    batch_idx = 0
    for epoch in range(n_epochs):
        for batch in id_train_dataloader:
            batch.to(device)
            out_labels, _, _ = model(batch)
            
            label_loss = F.nll_loss(out_labels, batch.y)
            optimizer.zero_grad()
            label_loss.backward()
            optimizer.step()

            avg_loss.append(label_loss.item())
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(batch.y).sum().item()
            train_acc = correct / len(batch.y)
            avg_acc.append(train_acc)
            
            print('Epoch: ',epoch ,'batch_idx: ', batch_idx, 'label_loss: ', label_loss.item(), 'train_acc: ', train_acc)
            batch_idx += 1

    # 2nd round of training: Train the model with GRL.
    for batch in second_train_dataloader:
        #print(batch.domain_y)
        out_labels, out_domains, indices = model(batch)    
        # Only get label_loss from out-of-domain samples. Label prediction is performed on the original data order. 
        ood_indices = torch.where(batch.domain_y == 1)[0]
        label_loss = F.nll_loss(out_labels[ood_indices], batch.y[ood_indices])

        # Align the order of predicted domain and domain true labels.
        new_domain_y = batch.domain_y[indices]
        domain_loss = F.nll_loss(out_domains, new_domain_y)
        loss = label_loss - domain_loss   # Here it has to be minus domain_loss cause we want to maximize it. 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


