from pathlib import Path
import numpy as np
import math
import h5py
import numpy as np
import unicodedata
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset
import time
import timm
import random
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import itertools
import string
from torch.autograd import Variable
from tqdm.autonotebook import tqdm

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

set_random_seeds(random_seed=13)


class CFG:
    debug = False
    batch_size = 200
    num_workers = 6
    head_lr = 0.0006
    image_encoder_lr = 0.0001
    text_encoder_lr = 0.0001
    weight_decay = 1e-3
    patience = 5
    factor = 0.8
    epochs = 200
    device = torch.device("cuda:2")

    image_embedding = 2048
    text_embedding = 300
    max_length = 30

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1
    
class Tokenizer():
    """Manager tokens functions and charset/dictionary properties"""

    def __init__(self, chars, max_text_length=CFG.max_length):
        self.PAD_TK, self.UNK_TK,self.SOS,self.EOS = "¶", "¤", "SOS", "EOS"
        self.chars = [self.PAD_TK] + [self.UNK_TK ]+ [self.SOS] + [self.EOS] +list(chars)
        self.PAD = self.chars.index(self.PAD_TK)
        self.UNK = self.chars.index(self.UNK_TK)

        self.vocab_size = len(self.chars)
        self.maxlen = max_text_length

    def encode(self, text):
        """Encode text to vector"""
        text = text.decode("utf-8") 
        text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
#         text = " ".join(text.split())

#         groups = ["".join(group) for _, group in groupby(text)]
#         text = "".join([self.UNK_TK.join(list(x)) if len(x) > 1 else x for x in groups])
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

class DataGenerator(Dataset):
    """Generator class with data streaming"""

    def __init__(self, source, split, transform, tokenizer):
        self.tokenizer = tokenizer
        self.transform = transform
        
        self.split = split
        self.dataset = dict()

#         self.dataset = h5py.File(source, "r")
        with h5py.File(source, "r") as f:
            self.dataset[self.split] = dict()

            self.dataset[self.split]['dt'] = np.array(f[self.split]['dt'])
            self.dataset[self.split]['gt'] = np.array(f[self.split]['gt'])
#             self.dataset[self.split]['label'] = np.array(f[self.split]['label'])            
          
#             randomize = np.arange(len(self.dataset[self.split]['gt']))
#             np.random.seed(42)
#             np.random.shuffle(randomize)

#             self.dataset[self.split]['dt'] = self.dataset[self.split]['dt'][randomize]
#             self.dataset[self.split]['gt'] = self.dataset[self.split]['gt'][randomize]
#         print(self.dataset[self.split]['gt'].shape)
    
        self.size = len(self.dataset[self.split]['gt'])

        

    def __getitem__(self, i):
        img = self.dataset[self.split]['dt'][i]
        #making image compatible with resnet
#         img = cv2.transpose(img)
#         img = np.repeat(img[..., np.newaxis],3, -1).astype("float32")   
#         img = pp.normalization(img).astype("float32")

        if self.transform is not None:
            aug = self.transform(image=img)
            img = aug['image']
            
            
#             img = self.transform(img)
        y_train = self.tokenizer.encode(self.dataset[self.split]['gt'][i].lower()) 
#         print(self.dataset[self.split]['gt'][i])
#         print(len(self.dataset[self.split]['gt'][i]))
#         if len(y_train)==0:
#             asdas
#         print(y_train)
#         print()
        #padding till max length
        y_train = np.pad(y_train, (0, self.tokenizer.maxlen - len(y_train)))
#         if all(y_train==0):
#             print(self.dataset[self.split]['gt'][i])
#             print("afdas")
#             ssa
        gt = torch.Tensor(y_train)
#         label = self.dataset[self.split]['label'][i]
        label = 1        
        if label==0:
            label = -1
            
        return img, gt,label         

    def __len__(self):
      return self.size

charset_base = string.printable[:95]
tokenizer = Tokenizer(charset_base)
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

criterion = LabelSmoothing(size=tokenizer.vocab_size, padding_idx=0, smoothing=.1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=CFG.max_length):
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


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    
cos_loss = nn.CosineEmbeddingLoss(reduction="mean", margin=.5)
class Clip(nn.Module):

    def __init__(self, 
                 tokenizer,
                 temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
):
        super().__init__()
    
#         self.backbone = resnet101(pretrained=args.pretrained)
        self.backbone = timm.create_model(
                    "resnest26d", True, num_classes=0, global_pool="avg"
                )
#         for p in self.backbone.parameters():
#             p.requires_grad = True
            
        self.tokenizer = tokenizer
        self.embeding = nn.Embedding(self.tokenizer.vocab_size,CFG.text_embedding)
        self.conv1 = nn.Conv1d(CFG.text_embedding,32,8)        
        self.gelu = nn.GELU()
        
#         self.pos_encoding = PositionalEncoding(CFG.text_embedding, .2)
            
#         encoder_layer = nn.TransformerEncoderLayer(d_model=CFG.text_embedding, nhead=4, dropout=.2)
#         self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0
        self.temperature = temperature
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=736)
        
    def make_len_mask(self, inp):
        return (inp == 0)
            
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask
        
    def forward(self, img, input_ids, check):        
        
        image_features = self.backbone.forward(img)        
#         with open("das.sa","a") as f:
#             f.write(str(pad_mask.sum(0)))
                
        input_ids = self.embeding(input_ids)
        text_features = self.gelu(self.conv1(input_ids.permute(0,2,1)))
        text_features = text_features.flatten(1)
#         print(text_features)
#         input_ids = self.pos_encoding(input_ids.permute(1,0,2))        
#         input_ids = input_ids.permute(1,0,2)
#         last_hidden_state = self.text_encoder(input_ids)
#         print()
#         print(last_hidden_state[:, self.target_token_idx, :])        
#         text_features = last_hidden_state[:, self.target_token_idx, :]
        
        # Getting Image and Text Embeddings (with same dimension)
        
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features) 
            
# #         print(image_features,text_features)
#         loss = cos_loss(image_embeddings, text_embeddings, check.to(CFG.device))
#         return loss
#         loss = 1-F.cosine_similarity(image_embeddings, text_embeddings)
#         Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()        


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

    
source = "iam_aug"
source_path = '../data/{}.hdf5'.format(source)

import albumentations
import albumentations.pytorch

transform_train = albumentations.Compose([
    albumentations.OneOf(
        [
            albumentations.MotionBlur(p=1, blur_limit=8),
            albumentations.OpticalDistortion(p=1, distort_limit=0.05),
            albumentations.GaussNoise(p=1, var_limit=(10.0, 100.0)),
            albumentations.RandomBrightnessContrast(p=1, brightness_limit=0.2),
            albumentations.Downscale(p=1, scale_min=0.8, scale_max=.9),
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



train_loader = torch.utils.data.DataLoader(DataGenerator(source_path,'train',transform_valid, tokenizer), batch_size=CFG.batch_size, shuffle=True, num_workers=2)
# valid_loader = torch.utils.data.DataLoader(DataGenerator(source_path,'test',transform_valid, tokenizer), batch_size=300, shuffle=False, num_workers=2)

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
#     tqdm_object = tqdm(train_loader, total=len(train_loader))
    for img, input_ids,check in train_loader:
        input_ids = input_ids.to(CFG.device).long()
        img = img.to(CFG.device)
        loss  = model(img, input_ids, check)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)        
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = img.size(0)
        loss_meter.update(loss.item(), count)
#         tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

#     tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for img, input_ids,check in valid_loader:
        input_ids = input_ids.to(CFG.device).long()
        img = img.to(CFG.device)
        loss = model(img, input_ids, check)
        
        count = img.size(0)
        loss_meter.update(loss.item(), count)
        
#         tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

model = Clip(tokenizer).to(CFG.device)

# params = [
#     {"params": model.backbone.parameters(), "lr": CFG.image_encoder_lr},
#     {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
#     {"params": itertools.chain(
#         model.image_projection.parameters(), model.text_projection.parameters()
#     ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
# ]
optimizer = torch.optim.AdamW(model.parameters(),lr=.0001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
)
step = "epoch"

best_loss = float('inf')
print(CFG.__dict__)
save = "../output/clip_entropy_26{}.pt".format(source)
print(save)
for epoch in range(CFG.epochs):
    print(f"Epoch: {epoch + 1}")
    model.train()
    train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
    print("Epoch ", epoch)
#     print("validation loss", valid_loss)
    print("training loss", train_loss)
    if train_loss.avg < best_loss:
        best_loss = train_loss.avg            
        torch.save(model.state_dict(), save)
        print("Saved Best Model!")
    
    print("learning rate", get_lr(optimizer))
    if epoch%10==0:
        save1 = "../output/clip_entropy_26{}_epoch{}.pt".format(source,epoch)
        torch.save(model.state_dict(), save1)
        print("Saved epoch Model!")

    lr_scheduler.step(train_loss.avg)
print("best_loss", best_loss)


# for epoch in range(CFG.epochs):
#     model.train()
#     train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
# #     model.eval()
# #     with torch.no_grad():
# #         valid_loss = valid_epoch(model, valid_loader)

#     print("Epoch ", epoch)
#     print("validation loss", valid_loss)
#     print("training loss", train_loss)
#     print("learning rate", get_lr(optimizer))
#     if valid_loss.avg < best_loss:
#         best_loss = valid_loss.avg            
#         torch.save(model.state_dict(), save)
#         print("Saved Best Model!")

#     lr_scheduler.step(valid_loss.avg)
# print("best_loss", best_loss)
