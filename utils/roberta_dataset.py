import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
# treeDic help function
def loadTree(treePath):
    #treePath = os.path.join(cwd,'data/'+dataname+'/Twitter_data_all.txt')
    print("reading twitter tree")
    treeDic = {}
    for line in open(treePath):
        line = line.strip('\n')
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        time_delay, text = float(line.split('\t')[3]), str(line.split('\t')[4])
        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'time_delay': time_delay, 'text': text}
    print('tree no:', len(treeDic))
    return treeDic

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, domain_y=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.domain_y = domain_y

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'domain_y': torch.tensor(self.domain_y, dtype=torch.long)
        }
        
