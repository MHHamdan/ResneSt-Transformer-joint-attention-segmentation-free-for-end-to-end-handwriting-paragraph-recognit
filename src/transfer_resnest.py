from pathlib import Path
import numpy as np
import math
from itertools import groupby
import h5py
import numpy as np
import unicodedata
import cv2
import torch
from torch import nn
from torchvision.models import resnet101
from torch.autograd import Variable
import torchvision
from data import preproc as pp
from data import evaluation
from torch.utils.data import Dataset
import time
import timm
import random

import wandb

# default `log_dir` is "runs" - we'll be more specific here
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="iam", type=str)
parser.add_argument("--local_rank", default="default", type=int)
parser.add_argument("--name", default="default", type=str)
parser.add_argument("--pretrained", default=True, type=str2bool)
parser.add_argument("--augmentation", default=True, type=str2bool)
parser.add_argument("--initialisation", default=True, type=str2bool)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--freeze", default="None", type=str)


args = parser.parse_args()
local_rank = args.local_rank

if args.pretrained:
    project_name = "Pretrained"
else:
    project_name = "Scratch"
    
def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

set_random_seeds(random_seed=args.seed)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class OCR(nn.Module):

    def __init__(self, vocab_len, hidden_dim, nheads,
                 num_encoder_layers, num_decoder_layers):
        super().__init__()
    
#         self.backbone = resnet101(pretrained=args.pretrained)
        self.backbone = timm.create_model('resnest101e', pretrained=args.pretrained)
        del self.backbone.fc
        _ = self.backbone.to(local_rank)
                
#         for name,p in self.backbone.named_parameters():
#             if "bn" not in name or "attnpool" in name:
#                 p.requires_grad =  False

        # create a default PyTorch transformer
        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads with length of vocab
        # DETR used basic 3 layer MLP for output
        self.vocab = nn.Linear(hidden_dim,vocab_len)

        # output positional encodings (object queries)
        self.decoder = nn.Embedding(vocab_len, hidden_dim)
        self.query_pos = PositionalEncoding(hidden_dim, .2)

        # spatial positional encodings, sine positional encoding can be used.
        # Detr baseline uses sine positional encoding.
        self.row_embed = nn.Parameter(torch.rand(90, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(90, hidden_dim // 2))
#         self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
#         self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

        self.trg_mask = None
  
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask
    
    def get_feature(self,x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)   
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x


    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)


    def forward(self, inputs, trg):
        # propagate inputs through ResNet-101 up to avg-pool layer
        x = self.get_feature(inputs)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        bs,_,H, W = h.shape
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # generating subsequent mask for target
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(trg.shape[1]).to(trg.device)

        # Padding mask
        trg_pad_mask = self.make_len_mask(trg)

        # Getting postional encoding for target
        trg = self.decoder(trg)
        trg = self.query_pos(trg)
        
        output = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1), trg.permute(1,0,2), tgt_mask=self.trg_mask, 
                                  tgt_key_padding_mask=trg_pad_mask.permute(1,0))

        return self.vocab(output.transpose(0,1))


def make_model(vocab_len, hidden_dim=256, nheads=4,
                 num_encoder_layers=4, num_decoder_layers=4):
    
    return OCR(vocab_len, hidden_dim, nheads,
                 num_encoder_layers, num_decoder_layers)

"""
Uses generator functions to supply train/test with data.
Image renderings and text are created on the fly each time.
"""

class DataGenerator(Dataset):
    """Generator class with data streaming"""

    def __init__(self, source, split, transform, tokenizer):
        self.tokenizer = tokenizer
        self.transform = transform
        
        self.split = split
        self.dataset = dict()

        with h5py.File(source, "r") as f:
            self.dataset[self.split] = dict()

            self.dataset[self.split]['dt'] = np.array(f[self.split]['dt'])
            self.dataset[self.split]['gt'] = np.array(f[self.split]['gt'])
          
            randomize = np.arange(len(self.dataset[self.split]['gt']))
            np.random.seed(42)
            np.random.shuffle(randomize)

            self.dataset[self.split]['dt'] = self.dataset[self.split]['dt'][randomize]
            self.dataset[self.split]['gt'] = self.dataset[self.split]['gt'][randomize]

            # decode sentences from byte
            self.dataset[self.split]['gt'] = [x.decode() for x in self.dataset[self.split]['gt']]
            
        self.size = len(self.dataset[self.split]['gt'])


    def __getitem__(self, i):
        img = self.dataset[self.split]['dt'][i]
        
        #making image compatible with resnet
#         img = cv2.transpose(img)
        img = np.repeat(img[..., np.newaxis],3, -1).astype("float32")   
#         img = pp.normalization(img).astype("float32")

        if self.transform is not None:
            aug = self.transform(image=img)
            img = aug['image']
            
#             img = self.transform(img)
            
        y_train = self.tokenizer.encode(self.dataset[self.split]['gt'][i]) 
        
        #padding till max length
        y_train = np.pad(y_train, (0, self.tokenizer.maxlen - len(y_train)))

        gt = torch.Tensor(y_train)

        return img, gt          

    def __len__(self):
      return self.size

class Tokenizer():
    """Manager tokens functions and charset/dictionary properties"""

    def __init__(self, chars, max_text_length=128):
        self.PAD_TK, self.UNK_TK,self.SOS,self.EOS = "¶", "¤", "SOS", "EOS"
        self.chars = [self.PAD_TK] + [self.UNK_TK ]+ [self.SOS] + [self.EOS] +list(chars)
        self.PAD = self.chars.index(self.PAD_TK)
        self.UNK = self.chars.index(self.UNK_TK)

        self.vocab_size = len(self.chars)
        self.maxlen = max_text_length

    def encode(self, text):
        """Encode text to vector"""
        text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
        text = " ".join(text.split())

        groups = ["".join(group) for _, group in groupby(text)]
        text = "".join([self.UNK_TK.join(list(x)) if len(x) > 1 else x for x in groups])
        encoded = []

        text = ['SOS'] + list(text) + ['EOS']
        for item in text:
            index = self.chars.index(item)
            index = self.UNK if index == -1 else index
            encoded.append(index)

        return np.asarray(encoded)

    def decode(self, text):
        """Decode vector to text"""
        
        decoded = "".join([self.chars[int(x)] for x in text if x > -1])
        decoded = self.remove_tokens(decoded)
        decoded = pp.text_standardize(decoded)

        return decoded

    def remove_tokens(self, text):
        """Remove tokens (PAD) from text"""

        return text.replace(self.PAD_TK, "").replace(self.UNK_TK, "")

import os
import datetime
import string

batch_size = 16
epochs = 200

# define paths
#change paths accordingly
source = args.source
source_path = '../data/{}.hdf5'.format(source)

# define input size, number max of chars per line and list of valid chars
input_size = (1024, 128, 1)
max_text_length = 128
charset_base = string.printable[:95]
# charset_base = string.printable[:36].lower() + string.printable[36+26:95].lower() 

print("source:", source_path)
print("charset:", charset_base)


import torchvision.transforms as T
device = torch.device("cuda:{}".format(local_rank))

# transform = T.Compose([
#     T.ToTensor()])
tokenizer = Tokenizer(charset_base)
import albumentations
import albumentations.pytorch

if args.augmentation:

    transform_train = albumentations.Compose([
        albumentations.OneOf(
            [
                albumentations.MotionBlur(p=1, blur_limit=8),
                albumentations.OpticalDistortion(p=1, distort_limit=0.05),
                albumentations.GaussNoise(p=1, var_limit=(10.0, 100.0)),
                albumentations.RandomBrightnessContrast(p=1, brightness_limit=0.2),
                albumentations.Downscale(p=1, scale_min=0.3, scale_max=0.5),
            ],
            p=.5,
        ),
#         albumentations.Resize(224,224),
        albumentations.Normalize(),
        albumentations.pytorch.ToTensorV2()

    ])

    transform_valid = albumentations.Compose(
        [
#         albumentations.Resize(224,224),            
            albumentations.Normalize(),
            albumentations.pytorch.ToTensorV2()
        ]
    )

else:
    transform_train = albumentations.Compose(
        [
#         albumentations.Resize(224,224),            
            albumentations.Normalize(),
            albumentations.pytorch.ToTensorV2()
        ]
    )
    
    transform_valid = albumentations.Compose(
        [
#         albumentations.Resize(224,224),
            
            albumentations.Normalize(),
            albumentations.pytorch.ToTensorV2()
        ]
    )
    
    

train_loader = torch.utils.data.DataLoader(DataGenerator(source_path,'train',transform_train, tokenizer), batch_size=batch_size, num_workers=6)
# val_loader = torch.utils.data.DataLoader(DataGenerator(source_path,'valid',transform_valid, tokenizer), batch_size=batch_size, shuffle=False, num_workers=6)
val_loader = torch.utils.data.DataLoader(DataGenerator(source_path,'valid',transform_valid, tokenizer), batch_size=batch_size, num_workers=6)

num_encoder_layers = 4
num_decoder_layers = 4


ddp_model = make_model( vocab_len=tokenizer.vocab_size,hidden_dim=256, nheads=4,
                 num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)

if args.freeze != "None":
    layers = args.freeze.split(",")
    l = []
    for i in layers:
        l.append("layer" + str(i))

    for name, p in ddp_model.backbone.named_parameters():
        if name.split(".")[0] in l:
            p.requires_grad = False


if args.initialisation:
    for name, parameter in ddp_model.named_parameters():
        if "backbone" not in name and (parameter.dim()>1):
            nn.init.xavier_uniform_(parameter)
            
elif args.initialisation and (not args.pretrained):
    init_funcs = {
    1: lambda x: torch.nn.init.normal_(x, mean=0., std=1.), # can be bias
    2: lambda x: torch.nn.init.xavier_normal_(x, gain=1.), # can be weight
    3: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.), # can be conv1D filter
    4: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.), # can be conv2D filter
    "default": lambda x: torch.nn.init.constant(x, 1.), # everything else
    }
    for p in ddp_model.parameters():
        init_func = init_funcs.get(len(p.shape), init_funcs["default"])
        init_func(p)
    


ddp_model = ddp_model.to(device)
# d = torch.load(
#     "/home/mhamdan/seq2seqAttenHTR/Transformer_ocr/output/cool-morning-52/bentham_firstbest_loss.pt.pt",
#     map_location="cuda:{}".format(args.local_rank),
# )
d = torch.load(
    "/home/mhamdan/seq2seqAttenHTR/Transformer_ocr/output/beloved-carnation-204/bentham_only_cursive_firstbest_loss.pt",
    map_location="cuda:{}".format(args.local_rank),
)

# d = torch.load(
#     "../output/vocal-moon-17/washington_firstbest.pt",
#     map_location={"cuda:0": "cuda:{}".format(args.local_rank)},
# )

ddp_model.load_state_dict(d)


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

smoothing = .4
criterion = LabelSmoothing(size=tokenizer.vocab_size, padding_idx=0, smoothing=smoothing)
criterion.to(device)
lr = .00006 # learnig rte
backbone_lr = .00006
# if not args.pretrained:
#     backbone_lr = backbone_lr*10
model = ddp_model
param_dicts = [
    {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
    {
        "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
        "lr": backbone_lr,
    },
]

scheduler_factor = .8

config = {"batch_size": batch_size,  # try log-spaced values from 1 to 50,000
          "optimizer": "adam",  # try optim.Adadelta and optim.SGD
          "database_name": source,
          "epochs": epochs,
          "lr": lr,
          "backbone_lr": backbone_lr,
          "num_decoder_layers":num_decoder_layers,
          "num_encoder_layers":num_encoder_layers,
          "augmentation":args.augmentation,
          "pretrained":args.pretrained,
          "scheduler_factor":scheduler_factor,
         "initialisation":args.initialisation,
         "seed":args.seed,
         "smoothing":smoothing,
         "sceduler":"loss",
            "freeze":args.freeze,
          "base_model":"resnest_aug",          
         "base_database":"bentham_only_cursive"}

optimizer = torch.optim.AdamW(param_dicts, lr=lr,weight_decay=.0004)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=scheduler_factor)



# Initialize wandb run
run = wandb.init(
    config=config,
    project=project_name,
)

if args.pretrained:
    output_path = os.path.join("..", "output", run.name)
else:
    output_path = os.path.join("..", "output", run.name)

count = 0
while True:
    if os.path.exists(output_path):
        if os.listdir(output_path):
            output_path = output_path+"_{}".format(count)
            count+=1
        else:
            break
    else:
        os.makedirs(output_path, exist_ok=False)
        break

target_path = os.path.join(output_path, "{}_{}".format(source,args.name))
print("output", output_path)
print("target", target_path)

with open(output_path+"/config.json","w") as f:
    f.write(str(config))

    

def train(model, criterion, optimiser,dataloader):
 
    model.train()
    total_loss = 0
    for batch, (imgs, labels_y,) in enumerate(dataloader):
          imgs = imgs.to(device)
          labels_y = labels_y.to(device)
    
          optimiser.zero_grad()
          output = model(imgs.float(),labels_y.long()[:,:-1])
 
          loss = criterion(output.log_softmax(-1).contiguous().view(-1, tokenizer.vocab_size), labels_y[:,1:].contiguous().view(-1).long()) 
 
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
          optimizer.step()
          total_loss += loss.item()
 
    return total_loss / len(dataloader)
 
def evaluate(model, criterion, dataloader,):
 
    model.eval()
    epoch_loss = 0
    cer = 0
    with torch.no_grad():
      for batch, (imgs, labels_y,) in enumerate(dataloader):
            imgs = imgs.to(device)
            labels_y = labels_y.to(device)
 
            output = model(imgs.float(),labels_y.long()[:,:-1])
            o = output.argmax(-1)
            predicts = list(map(lambda x : tokenizer.decode(x).replace('SOS','').replace('EOS',''),o))
            gt = list(map(lambda x : tokenizer.decode(x).replace('SOS','').replace('EOS',''),labels_y))
            cer += evaluation.ocr_metrics(predicts=predicts,
                                   ground_truth=gt)[0]
            
            loss = criterion(output.log_softmax(-1).contiguous().view(-1, tokenizer.vocab_size), labels_y[:,1:].contiguous().view(-1).long())
  
            epoch_loss += loss.item()
    
 
    return epoch_loss / len(dataloader), cer



#train model
 
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


best_CER = np.inf
best_valid_loss = np.inf
import torch.distributed as dist
 
c = 0
for epoch in range(epochs):
      
    start_time = time.time()
     
    train_loss = train(model,  criterion, optimizer, train_loader)
    valid_loss,cer = evaluate(model, criterion, val_loader)
    run.log({"epoch": epoch, "train_loss": train_loss, "valid_loss":valid_loss, "lr":lr_scheduler.get_last_lr()[0], "lr_backbone":lr_scheduler.get_last_lr()[1], "cer":cer})

    epoch_mins, epoch_secs = epoch_time(start_time, time.time())
    c+=1
    save = 0
    if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), target_path+"best_loss.pt") 
            save=1
            c=0
    
    if cer < best_CER:
            best_CER = cer
            torch.save(model.state_dict(), target_path+"best_cer.pt") 
#             save=1
#             c=0

    if epoch%10==0 and epoch!=0 and save==0:
        torch.save(model.state_dict(), target_path+"_{}.pt".format(epoch))                

    if c>4:
        #decrease lr if loss does not deacrease after 5 steps
        lr_scheduler.step()
        c=0
    
    print(f'Epoch: {epoch+1:02}','learning rate{}'.format(lr_scheduler.get_last_lr()))
    print(f'Time: {epoch_mins}m {epoch_secs}s') 
    print(f'Train Loss: {train_loss:.3f}')
    print(f'Val   Loss: {valid_loss:.3f}')
    print("CER: ", cer)            
    
run.finish()

    
