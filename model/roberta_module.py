import torch
import torch.nn as nn

from transformers import RobertaTokenizer, RobertaModel
from model.rgl import ReverseLayerF

    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class RoBERTaExtractor(nn.Module):
    def __init__(self):
        super(RoBERTaExtractor, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')

    def forward(self, batch_data):
        pass
        input_ids = batch_data['input_ids'].to(device)
        attention_mask = batch_data['attention_mask'].to(device)
        _, pooled_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        return pooled_output


    
class RoBERTaLabelClassifier(nn.Module):
    def __init__(self, hidden_size=768, n_classes=2):
        super(RoBERTaLabelClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_size, n_classes)
    
    def forward(self, pooled_output):
        return self.classifier(pooled_output)
    
class RoBERTaDomainClassifier(nn.Module):
    def __init__(self, feat_dim, domain_dim):
        super(RoBERTaDomainClassifier, self).__init__()
        self.fc=nn.Linear(feat_dim, domain_dim)
    
    def forward(self, x, alpha):
        x = ReverseLayerF.apply(x, alpha)
        x=self.fc(x)
        return x

