import torch
from torch.autograd import Function
from model.bigcn_module import bigcn_feature_extractor

# CADA: 
# This is a plugging-in framework for cross-domain fake news detection. 

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class CADA(torch.nn.Module):
    def __init__(self, feature_extractor, label_predictor, domain_classifier):
        super(CADA, self).__init__()
        pass
        self.feature_extractor = feature_extractor
        self.label_predictor = label_predictor
        self.domain_classifier = domain_classifier

    def forward(self, news_input):
        x_feat = self.feature_extractor(news_input)
        label_output = self.label_predictor(x_feat)
        domain_output = self.domain_classifier(x_feat)
        return label_output, domain_output






