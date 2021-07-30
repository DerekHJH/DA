#!/usr/bin/env python
# coding: utf-8

# # DAViT

# ## Import Libraries

# In[1]:


import os
import sys
import datetime
import random
import numpy as np
import pandas as pd

import torch 
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,datasets 
from torch import einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from matplotlib import pyplot as plt 

from torchsummary import summary


# ## Utilities

# In[2]:


def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)


# ## Define Super-parameters

# In[3]:


seed = 42
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# about the data set
train_path = '/global/cfs/cdirs/m3691/data/office/amazon/images'
valid_path = '/global/cfs/cdirs/m3691/data/office/dslr/images'
task_name = train_path.split('/')[-2] + '->' + valid_path.split('/')[-2]
channels = 3
image_size = 224
num_classes = 31

# about the data loader
batch_size = 32
num_workers = 16

# about the model
patch_size = 32
patch_side = image_size // patch_size
num_patches = patch_side * patch_side
dim = 512
depth = 4
heads = 8
mlp_dim = 1024
emb_dropout = 0.1
dropout = 0.1

# about the optimizer
lr = 3e-5
gamma = 0.7

# about the training 
epochs = 200
log_step_freq = 1

# about saving and loading 
model_save_path = "./models/" + task_name
epochs_to_save = 1


# In[4]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)


# ## Load Data

# In[5]:


train_transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
#         transforms.RandomResizedCrop(image_size),
#         transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
#         transforms.Normalize(mean=(0.1307, 0.1307, 0.1307), std=(0.3081, 0.3081, 0.3081)),
    ]
)

valid_transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
#         transforms.RandomResizedCrop(image_size),
#         transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)


# In[6]:


train_data = datasets.ImageFolder(root = train_path, transform = train_transforms)
valid_data = datasets.ImageFolder(root = valid_path, transform = valid_transforms)

print(train_data.class_to_idx)
print()
print(len(train_data))
print(type(train_data[0]))
print(train_data[0][0].shape, train_data[0][1])


# In[7]:


train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = num_workers)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = batch_size, shuffle = True, num_workers = num_workers)

data, label = next(iter(train_loader))
print(data[0].shape, label[0])


# In[8]:


'''
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
plt.figure(figsize=(8,8)) 
for i in range(9):
    img,label = train_data[i]
    img = img.permute(1,2,0)
    ax=plt.subplot(3,3,i+1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d"%label)
    ax.set_xticks([])
    ax.set_yticks([]) 
plt.show()
'''

# ## Define Model

# In[9]:


class ReverseLayerF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.feature = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.class_classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(dim, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))

        self.pool = pool

    def forward(self, img, alpha):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        feature = self.feature(x)
        feature = feature[:, 0]
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output


# In[10]:


model = ViT(
        image_size = image_size,
        patch_size = patch_size,
        num_classes = num_classes,
        dim = dim,
        depth = depth,
        heads = heads, 
        mlp_dim = mlp_dim,
        dropout = dropout,
        emb_dropout = emb_dropout
        ).to(device)
# summary(model, input_size = [[channels, image_size, image_size], [1]], batch_size = batch_size)


# In[11]:


model.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
model.criterion = nn.CrossEntropyLoss()
model.metric_func = lambda y_pred, y_true: (y_pred.argmax(dim = 1) == y_true).float().mean()
model.metric_name = "accuracy"


# ## Train Model

# In[12]:


def test(model, data_loader):
    #dropout layer stops working
    model.eval()
    
    len_loader = len(data_loader)
    data_iter = iter(data_loader)

    n_total = 0
    n_correct = 0

    for i in range(len_loader):
        img, label = data_iter.next()

        batch_size = len(label)

        img = img.to(device)
        label = label.to(device)

        class_output, _ = model(img = img, alpha = 0)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu


# In[13]:


def train_model(model, epochs, train_loader, valid_loader, log_step_freq, model_save_path, epochs_to_save):
    
    dfhistory = pd.DataFrame(columns = ["epoch", "err_s_label", "err_s_domain", "err_t_domain", "err", "accu_s", "accu_t", "best_accu_s", "best_accu_t"])
    print("Start Training...")
#     nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     print("=========="*8 + "%s"%nowtime)
    start_epoch = 1
    
    # load everything if existed
#     if os.path.exists(model_save_path):
#         checkpoint = torch.load(model_save_path)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         dfhistory = checkpoint['dfhistory']
#         start_epoch = int(dfhistory.loc[dfhistory.shape[0] - 1]['epoch']) + 1
        
    best_accu_t = 0.0
    best_accu_s = 0.0
    for epoch in range(start_epoch, epochs + 1): 
        len_loader = min(len(train_loader), len(valid_loader))
        data_source_iter = iter(train_loader)
        data_target_iter = iter(valid_loader)
        # 1，training loop-------------------------------------------------
        step = 0
        err_s_label = 0
        err_s_domain = 0
        err_t_domain = 0
        err = 0
        for step in range(len_loader):
            model.train()
            p = float(step + epoch * len_loader) / epochs / len_loader
            alpha = 0.2 / (1. + np.exp(-10 * p)) - 1
            
            # training model using source data
            s_img, s_label = data_source_iter.next()

            model.zero_grad()
            batch_size = len(s_label)
            domain_label = torch.zeros(batch_size).long()

            s_img = s_img.to(device)
            s_label = s_label.to(device)
            domain_label = domain_label.to(device)

            class_output, domain_output = model(img = s_img, alpha = alpha)
            err_s_label = model.criterion(class_output, s_label)
            err_s_domain = model.criterion(domain_output, domain_label)

            # training model using target data
            t_img, _ = data_target_iter.next()

            batch_size = len(t_img)

            domain_label = torch.ones(batch_size).long()

            t_img = t_img.to(device)
            domain_label = domain_label.to(device)

            _, domain_output = model(img = t_img, alpha = alpha)
            err_t_domain = model.criterion(domain_output, domain_label)
            err = err_t_domain + err_s_domain + err_s_label
            err.backward()
            model.optimizer.step()
            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f'               % (epoch, step + 1, len_loader, err_s_label.data.cpu().numpy(),
                 err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
            sys.stdout.flush()
            
        # 2，test-------------------------------------------------
        accu_s = test(model, train_loader)
        print('\nAccuracy of the source dataset: %f' % accu_s)
        accu_t = test(model, valid_loader)
        print('Accuracy of the target dataset: %f\n' % accu_t)
        if accu_t > best_accu_t:
            best_accu_s = accu_s
            best_accu_t = accu_t   

        # 3，logging-------------------------------------------------
        info = (epoch, err_s_label, err_s_domain, err_t_domain, err, accu_s, accu_t, best_accu_s, best_accu_t)
        dfhistory.loc[epoch - 1] = info
#         torch.save(
#             {
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': model.optimizer.state_dict(),
#                 'dfhistory': dfhistory
#             }, model_save_path)
#         if epoch > 1 and accu_t > dfhistory.loc[epoch - 2]["best_accu_t"]:
#             torch.save(
#                 {
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': model.optimizer.state_dict(),
#                     'dfhistory': dfhistory
#                 }, model_save_path + '_best') 
        
#         nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         print("\n"+"=========="*8 + "%s"%nowtime)

    print('Finished Training...')

    return dfhistory


# In[14]:


dfhistory = train_model(model, epochs, train_loader, valid_loader, log_step_freq, model_save_path, epochs_to_save)


# ## Evaluate Model

# In[21]:
'''

dfhistory


# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')
def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()


# In[23]:


plot_metric(dfhistory, "loss")


# In[24]:


plot_metric(dfhistory, "accuracy")


# ## Use Model

# In[28]:


def predict(model, test_loader):
    model.eval()
    result = torch.cat([model.forward(t[0].to(device)) for t in test_loader])
    return(result.data)


# In[29]:


y_pred_probs = predict(model, valid_loader)
y_pred_probs


# In[30]:


y_pred = torch.where(y_pred_probs > 0.5,
        torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
y_pred
'''
