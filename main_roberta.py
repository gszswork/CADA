import os
import torch
import numpy as np
from transformers import AdamW
from sklearn import metrics
from utils.roberta_dataset import *
import torch.nn.functional as F
from model.roberta_module import *
from CADA import CADA

cwd = os.getcwd()
parent_directory = os.path.dirname(cwd)
#data_path = os.path.join(parent_directory, 'data')
# TODO: In Github, we should use the following line.
data_path = cwd + '/data'

id_tree_path = os.path.join(data_path, 'in-domain/Twitter/Twitter_data_all.txt')
ood_tree_path = os.path.join(data_path, 'out-of-domain/Twitter/Twitter_data_all.txt')

id_treeDict = loadTree(id_tree_path)
ood_treeDict = loadTree(ood_tree_path)

# Load label dictionary.
# Why we have 2139 samples in the id_treeDict? 4-labels?  -- Yes, we have to filter with 
id_label_dict = {}
with open(os.path.join(data_path, 'in-domain/Twitter/Twitter_label_all.txt'), 'r') as file:
    for line in file:
        id, label = line.strip().split('\t')
        id_label_dict[id] = label

ood_label_dict = {}
with open(os.path.join(data_path, 'out-of-domain/Twitter/Twitter_label_all.txt'), 'r') as file:
    for line in file:
        id, label = line.strip().split('\t')
        ood_label_dict[id] = label

# Prepare the id/ood texts and labels for Dataloader. 
id_texts, id_labels = [], []
for key in id_label_dict.keys():
    id_texts.append(id_treeDict[key][1]['text'])
    id_labels.append(int(id_label_dict[key]))

ood_texts, ood_labels = [], [] 
for key in ood_label_dict.keys():
    ood_texts.append(ood_treeDict[key][1]['text'])
    ood_labels.append(int(ood_label_dict[key]))

    
# We need to split the in/ood dataset into 80-20 splits. 
id_train_texts, id_train_labels = id_texts[:int(0.8*len(id_texts))], id_labels[:int(0.8*len(id_labels))]
id_test_texts, id_test_labels = id_texts[int(0.8*len(id_texts)):], id_labels[int(0.8*len(id_texts)):]
print(len(id_train_texts), len(id_train_labels), len(id_test_texts), len(id_test_labels))

ood_test_texts, ood_test_labels = ood_texts[:int(0.8*len(ood_texts))], ood_labels[:int(0.8*len(ood_labels))]
ood_train_texts, ood_train_labels = ood_texts[int(0.8*len(ood_texts)):], ood_labels[int(0.8*len(ood_texts)):]
print(len(ood_train_texts), len(ood_train_labels), len(ood_test_texts), len(ood_test_labels))

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

id_train_dataset = TextClassificationDataset(id_train_texts, id_train_labels, tokenizer, max_len=128, domain_y=0)
id_train_loader = DataLoader(id_train_dataset, batch_size=32, shuffle=True)
id_test_dataset = TextClassificationDataset(id_test_texts, id_test_labels, tokenizer, max_len=128, domain_y=0)
id_test_loader = DataLoader(id_test_dataset, batch_size=32, shuffle=True)

ood_train_dataset = TextClassificationDataset(ood_train_texts, ood_train_labels, tokenizer, max_len=128, domain_y=1)
ood_train_loader = DataLoader(ood_train_dataset, batch_size=32, shuffle=True)
ood_test_dataset = TextClassificationDataset(ood_test_texts, ood_test_labels, tokenizer, max_len=128, domain_y=1)
ood_test_loader = DataLoader(ood_test_dataset, batch_size=32, shuffle=True)

if __name__ == '__main__':
    # init hyper-params. 
    number_of_classes = 2
    num_epochs = 200
    n_epochs_2nd = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Splice the Model.
    
    bert_feat_dim=768
    feature_extractor = RoBERTaExtractor()
    label_classifier = RoBERTaLabelClassifier()
    domain_classifier0 = RoBERTaDomainClassifier(768, 2)
    domain_classifier1 = RoBERTaDomainClassifier(768, 2)
    domain_classifier_list = [domain_classifier0, domain_classifier1]
    model = CADA(feature_extractor, label_classifier, domain_classifier_list)
    model.to(device)
    '''
    in_dim = 768
    hid_dim = 128
    out_dim = 64
    bigcn_feature_extractor = bigcn_feature_extractor(in_dim, hid_dim, out_dim)
    bigcn_label_predictor = bigcn_label_predictor((out_dim+hid_dim)*2, 2)
    bigcn_domain_classifier_0 = bigcn_domain_classifier((out_dim+hid_dim)*2, 2)
    bigcn_domain_classifier_1 = bigcn_domain_classifier((out_dim+hid_dim)*2, 2)
    bigcn_domain_classifier_list = [bigcn_domain_classifier_0, bigcn_domain_classifier_1]
    model = CADA(bigcn_feature_extractor, bigcn_label_predictor, bigcn_domain_classifier_list)
    '''
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)
    # Training loop

    # 1. Pre-training: Train the model on in-domain data.
    for epoch in range(num_epochs):
        model.train()
        for batch in id_train_loader:
            labels = batch['labels'].to(device)
            # Forward pass
            outputs = model.forward_label(batch)
            _, pred = outputs.max(dim=-1)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        valid_true, valid_pred = np.array([]), np.array([])
        for batch in id_test_loader:
            labels = batch['labels'].to(device)
            outputs = model.forward_label(batch)
            _, pred = outputs.max(dim=-1)
            
            valid_true = np.append(valid_true, labels.cpu().numpy())
            valid_pred = np.append(valid_pred, pred.cpu().numpy())
        print('1st round validation report:')
        print(metrics.classification_report(valid_true, valid_pred))
        
        best_accuracy = -1
        test_true, test_pred = np.array([]), np.array([])
        for batch in ood_test_loader:
            #input_ids = batch['input_ids']
            #ttention_mask = batch['attention_mask']
            labels = batch['labels'].to(device)
            #outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            outputs = model.forward_label(batch)
            _, pred = outputs.max(dim=-1)
            
            test_true = np.append(test_true, labels.cpu().numpy())
            test_pred = np.append(test_pred, pred.cpu().numpy())
        print('1st round test report:')
        print(metrics.classification_report(test_true, test_pred))
        cur_acc = metrics.accuracy_score(test_true, test_pred)
        if cur_acc > best_accuracy:
            best_accuracy = cur_acc
            torch.save(model.state_dict(), 'model.pt')
        print(f"Pre-training in Epoch {epoch+1}/{num_epochs} completed.")

    # 2nd round of training: Train the model with GRL. 
    model.load_state_dict(torch.load('model.pt'))
    for epoch in range(n_epochs_2nd):
        model.train()
        second_iter = iter(ood_train_loader)
        for i in range(len(ood_train_loader)):
            batch = next(second_iter)

            labels = batch['labels'].to(device)
            p = float(i + epoch * len(ood_train_loader)) / n_epochs_2nd / len(ood_train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            alpha = torch.tensor(alpha).to(device)
            #print(batch.domain_y)
            out_labels, out_domains, indices = model(batch, alpha)    
            # Only get label_loss from out-of-domain samples. Label prediction is performed on the original data order. 
            ood_indices = torch.where(batch['domain_y'] == 1)[0]
            label_loss = F.nll_loss(out_labels[ood_indices], labels[ood_indices])

            # Align the order of predicted domain and domain true labels.
            new_domain_y = batch['domain_y'][indices].to(device)
            domain_loss = F.nll_loss(out_domains, new_domain_y)

            loss = label_loss + domain_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch: ',epoch ,'batch_idx: ', i, 'label_loss: ', label_loss.item(), 'domain_loss: ', domain_loss.item())

        model.eval()
        valid_true, valid_pred = np.array([]), np.array([])
        for batch in id_test_loader:
            #input_ids = batch['input_ids']
            #attention_mask = batch['attention_mask']
            labels = batch['labels'].to(device)
            outputs = model.forward_label(batch)
            #outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, pred = outputs.max(dim=-1)
            
            valid_true = np.append(valid_true, labels.cpu().numpy())
            valid_pred = np.append(valid_pred, pred.cpu().numpy())
        print(metrics.classification_report(valid_true, valid_pred))
        
        best_accuracy = -1
        test_true, test_pred = np.array([]), np.array([])
        for batch in ood_test_loader:
            #input_ids = batch['input_ids']
            #attention_mask = batch['attention_mask']
            labels = batch['labels'].to(device)
            outputs = model.forward_label(batch)
            #outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, pred = outputs.max(dim=-1)
            
            test_true = np.append(test_true, labels.cpu().numpy())
            test_pred = np.append(test_pred, pred.cpu().numpy())
        print(metrics.classification_report(test_true, test_pred))
