import torch
from torch.autograd import Function
from model.bigcn_module import bigcn_feature_extractor
import torch.nn.functional as F

# CADA: 
# This is a plugging-in framework for cross-domain fake news detection. 
# The current version only support two classes. 

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

'''
class CADA(torch.nn.Module):
    def __init__(self, feature_extractor, label_predictor, domain_classifier_0, domain_classifier_1):
        super(CADA, self).__init__()
        pass
        self.feature_extractor = feature_extractor
        self.label_predictor = label_predictor
        self.domain_classifier_0 = domain_classifier_0
        self.domain_classifier_1 = domain_classifier_1


    def forward(self, news_input):
        x_feat = self.feature_extractor(news_input)
        label_output = self.label_predictor(x_feat)

        _, pred = label_output.max(dim=-1)
        # Be aware of the batch training. 
        one_indices = torch.where(pred == 1)
        zero_indices = torch.where(pred == 0)

        x_feat_0  = x_feat[zero_indices]
        x_feat_1  = x_feat[one_indices]

        domain_output_0 = self.domain_classifier_0(x_feat_0)
        domain_output_1 = self.domain_classifier_1(x_feat_1)
        domain_output = torch.cat((domain_output_0, domain_output_1), 0)
        

        return label_output, domain_output, one_indices, zero_indices
'''


class CADA(torch.nn.Module):
    def __init__(self, feature_extractor, label_predictor, domain_classifier_list):
        super(CADA, self).__init__()
        self.feature_extractor = feature_extractor
        self.label_predictor = label_predictor
        self.domain_classifier_list = domain_classifier_list
        self.class_num = len(domain_classifier_list)


    def forward(self, news_input):
        x_feat = self.feature_extractor(news_input)
        label_output = self.label_predictor(x_feat)

        _, pred = label_output.max(dim=-1)
        # Be aware of the batch training. 
        indices = [torch.where(pred == i)[0] for i in range(self.class_num)]
        x_domain_feat = [x_feat[indices[i]] for i in range(self.class_num)]
        domain_output = [self.domain_classifier_list[i](x_domain_feat[i], alpha=0.2) for i in range(self.class_num)]

        cat_domain_output = torch.cat(domain_output, dim=0)
        label_output = F.log_softmax(label_output, dim=1)
        cat_domain_output = F.log_softmax(cat_domain_output, dim=1)
        indices = torch.cat(indices, dim=0)

        return label_output, cat_domain_output, indices
        
