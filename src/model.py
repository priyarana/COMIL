import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
from sklearn.cluster import KMeans
import warnings
from scipy.stats import entropy

from sparsemax import Sparsemax  # pip install sparsemax

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes
        

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes """

class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]        #PReLU, Leaky ReLU, or ELU or  LeakyReLU(negative_slope=0.01)  
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.50))
            self.attention_b.append(nn.Dropout(0.50))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)


    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)        
        A = a.mul(b)   #torch.add(a, b) #,torch.cat((a, b), dim=1),  torch.sum(dot_product, dim=1, keepdim=True), 
        A = self.attention_c(A)
        return A, x
  

class MultiHeadAttention(nn.Module):
    def __init__(self, L = 1024, D = 256, num_heads =4, dropout = False, n_classes = 1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        output_dim = D
        input_dim = L
        self.head_dim = output_dim // num_heads
        
        # Linear transformations for query, key, and value for each head
        self.query_linear = nn.Linear(input_dim, output_dim)
        self.key_linear = nn.Linear(input_dim, output_dim)
        self.value_linear = nn.Linear(input_dim, output_dim)
        
        # Final linear transformation after concatenating attention heads
        self.final_linear = nn.Linear(output_dim, output_dim)
        
    def forward(self, input_tensor):
        batch_size = 1
        num_patches, _ = input_tensor.size()
        
        
        # Apply linear transformations to obtain query, key, and value for each head
        query = self.query_linear(input_tensor)
        key = self.key_linear(input_tensor)
        value = self.value_linear(input_tensor)
        
        # Reshape query, key, and value for multi-head attention
        query = query.view(batch_size, num_patches, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, num_patches, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, num_patches, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute scaled dot-product attention
        attention_scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)
        
        # Reshape and concatenate attention heads
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, num_patches, -1)
        
        # Apply final linear transformation
        multihead_output = self.final_linear(attention_output)
        
        
        return multihead_output, input_tensor

        
class COMIL(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = True, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, epsilon=1e-5):
        super(COMIL, self).__init__()
        
        #self.size_dict = {"small": [512, 256, 128], "big": [512, 384, 192]}  #2208
        
        #self.size_dict = {"small": [2208, 1024, 512], "big": [2208, 1024, 768]}  
        
        #self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}      #for UNI   
        self.size_dict = {"small": [2048, 1024, 512], "big": [2048, 1024, 768]}     #2048 ResNet101
        #self.size_dict = {"small": [1920, 1024, 512], "big": [1920, 1024, 768]}   #DenseNet 1920 
        
        size = self.size_dict[size_arg]
      
        self.alpha = nn.Parameter(torch.ones(1, 7, 1))
        self.gamma = nn.Parameter(torch.zeros(1,7, 1))
        self.beta = nn.Parameter(torch.zeros(1, 7, 1))
        self.epsilon =epsilon
        
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.BatchNorm1d(size[1])]
        
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
            #attention_net =  MultiHeadAttention(L = size[1], D = size[2], num_heads =4, dropout = 0.5, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)


    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
        
        
    def assign_targets(self, cluster_labels, device):
        # Assign targets based on cluster membership
        # For simplicity, let's assume the majority cluster as positive and the rest as negative
        p_cluster = np.argmax(np.bincount(cluster_labels))
        p_targets = torch.tensor(cluster_labels == p_cluster, dtype=torch.float32, device=device)
        n_targets = torch.tensor(cluster_labels != p_cluster, dtype=torch.float32, device=device)
        return p_targets, n_targets

    
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        instance_loss, all_preds, all_targets = torch.tensor(0.0, device=device), None, None #torch.tensor(0.0, device=device)
        
        
        unique_elements = torch.unique(A[0,:])
        
        from scipy.stats import entropy, gaussian_kde


        if self.k_sample < A.size()[1]:
          
          device=h.device
          if len(A.shape) == 1:
              A = A.view(1, -1)
          top_p_ids = torch.topk(A, self.k_sample)[1][-1]
          top_p = torch.index_select(h, dim=0, index=top_p_ids)
          top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
          top_n = torch.index_select(h, dim=0, index=top_n_ids) 

          p_targets = self.create_positive_targets(self.k_sample, device)
          n_targets = self.create_negative_targets(self.k_sample, device)
  
          all_targets = torch.cat([p_targets, n_targets], dim=0)
          all_instances = torch.cat([top_p, top_n], dim=0)
          logits = classifier(all_instances)
          all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
          instance_loss = self.instance_loss_fn(logits, all_targets)   #instance_loss is tensor
          
          
        return instance_loss, all_preds, all_targets

    
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets
        
        
    def random_channel_mask(self, h, drop_prob=0.05, apply_prob=0.3):
        if not self.training or torch.rand(1).item() > apply_prob:
            return h
    
        B, C, D = h.shape
    
        # Generate mask for all channels
        mask = (torch.rand(B, C, 1, device=h.device) > drop_prob).float()
    
        # Force channels 0 and 1 to always be kept (mask = 1)
        mask[:, 0, :] = 1.0
        mask[:, 1, :] = 1.0
    
        return h * mask

    
    def add_channel_noise(self,h, noise_std=0.005):
        if not self.training:
            return h
        noise = torch.randn_like(h) * noise_std
        return h + noise


    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device   #h.size())   #[694, 2048]
        #h = self.random_channel_mask(h, drop_prob=0.05)
        #h = self.add_channel_noise(h, noise_std=0.002)

        alpha = self.alpha.to(device)
        gamma = self.gamma.to(device)
        beta = self.beta.to(device)
        
        embedding =  (h.pow(2).sum((2), keepdim=True) + self.epsilon).pow(0.5) * alpha  #([310, 7, 1])
        
        norm = gamma / (embedding.pow(2).mean(dim=1,keepdim=True) + self.epsilon).pow(0.5) #([310, 7, 1])
        
        gate = 1. + torch.tanh(embedding * norm + beta)   #([310, 7, 1])

        h = h * gate    #([310, 7, 2048])        
        h = F.adaptive_avg_pool1d(h.permute(0, 2, 1), 1).squeeze(dim=-1)     #([310, 2048]

        A, h = self.attention_net(h)  # NxK   
                       
        A = torch.transpose(A, 1, 0)  # KxN
        
        if attention_only:
            return A
        
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N     # A, h size :torch.Size([1, 1405]) torch.Size([1405, 1024])

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            
            
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)    #torch.Size([]) torch.Size([16]) torch.Size([16])
                    if preds is not None:
                      all_preds.extend(preds.cpu().numpy())
                    if targets is not None:
                      all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                
                
                total_inst_loss += instance_loss   # instance_loss and total_inst_loss are 'torch.Tensor'>
                
            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
              
        M = torch.mm(A, h)                        #A.size(), h.size() = torch.Size([1, 1282]) torch.Size([1282, 1024])
        
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
      
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict


