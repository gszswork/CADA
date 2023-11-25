from utils.train_test_split import get_id_ood_twitter_ids
from utils.bigcn_dataset import BiGraphDataset, my_loadTree
from model.bigcn_module import * 
from torch_geometric.data import DataLoader
from CADA import CADA


if __name__ == '__main__':
    dataname = 'Twitter'
    # Load the dataset. 
    id_twitter_ids, ood_twitter_ids = get_id_ood_twitter_ids()
    id_treeDic, ood_treeDic = my_loadTree('Twitter')
    #print(len(id_treeDic), len(ood_treeDic))

    in_data_path = os.path.join(os.getcwd(), 'data', 'in-domain', dataname + 'graph')
    ood_data_path = os.path.join(os.getcwd(), 'data', 'out-of-domain', dataname + 'graph')

    id_dataset = BiGraphDataset(id_twitter_ids, id_treeDic, lower=2, upper=100000, tddroprate=0.2, budroprate=0.2, data_path=in_data_path, domain_y=0)
    ood_dataset = BiGraphDataset(ood_twitter_ids, ood_treeDic, lower=2, upper=100000, tddroprate=0.2, budroprate=0.2, data_path=ood_data_path, domain_y=1)

    #print(id_dataset[0], id_dataset[0].domain_y)


    id_dataloader = DataLoader(id_dataset, batch_size=16, shuffle=True)

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

    for batch in id_dataloader:
        model(batch)


