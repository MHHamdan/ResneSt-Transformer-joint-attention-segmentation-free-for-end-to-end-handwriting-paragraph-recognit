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
from torch.autograd import Variable
from torch.utils.data import Dataset
import time
import albumentations
import albumentations.pytorch
import timm
import argparse
import string
import json
torch.backends.cudnn.benchmark = True
torch.manual_seed(13)
np.random.seed(13)
def parse_args():
    """[This is a function used to parse command line arguments]
    Returns:
        args ([object]): [Parse parameter object to get parse object]
    """
    parse = argparse.ArgumentParser(description='ocr')
    parse.add_argument('--target_path', type=str, default="model/", help='target folder')
    parse.add_argument('--name_file', type=str, default="resnest", help='name of the state dict')
    parse.add_argument('--file_path', type=str, default="", help='path of the hdf5 file')
    parse.add_argument('--epochs', type=int, default=200, help='Number of total epochs')
    parse.add_argument('--batch_size', type=int, default=36, help='Size of one batch')
    parse.add_argument('--lr', type=float, default=0.00006, help='Initial learning rate')
    parse.add_argument('--charset_base', type=str, default=string.printable[:95], help='path to vocab')
    parse.add_argument('--device', type=int, default=0, help='cuda device')
    parse.add_argument('--finetune', type=str, default='', help='pretrain model path')

    args = parse.parse_args()
    return args


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=46):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class OCR(nn.Module):
    def __init__(
        self,
        vocab_len,
        max_len,
        hidden_dim,
        nheads,
        num_encoder_layers,
        num_decoder_layers,
    ):
        super().__init__()

        #         self.backbone = resnet101(pretrained=args.pretrained)
        self.backbone = timm.create_model("resnest101e", pretrained=True)
        del self.backbone.fc

        #         self.backbone = swin_b(weights=weights)
        #         del self.backbone.head

        #         for name,p in self.backbone.named_parameters():
        #             if "bn" not in name or "attnpool" in name:
        #                 p.requires_grad =  False

        # create a default PyTorch transformer
        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        #         self.transformer = nn.Transformer(
        #             hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads with length of vocab
        # DETR used basic 3 layer MLP for output
        self.vocab = nn.Linear(hidden_dim, vocab_len)

        # output positional encodings (object queries)
        self.decoder = nn.Embedding(vocab_len, hidden_dim)
        self.query_pos = PositionalEncoding(hidden_dim, 0.2, max_len)
        self.row_embed = nn.Parameter(torch.rand(15, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(15, hidden_dim // 2))

        # spatial positional encodings, sine positional encoding can be used.

        self.trg_mask = None

    def get_feature(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, inputs, trg):
        # propagate inputs through ResNet-101 up to avg-pool layer
        x = self.get_feature(inputs)
        # convert from 1024 to 256 feature planes for the transformer
        h = self.conv(x)
        # construct positional encodings
        bs, _, H, W = h.shape
        pos = (
            torch.cat(
                [
                    self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                    self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(1)
        )
        h = pos + 0.1 * h.flatten(2).permute(2, 0, 1)

        # construct positional encodings

        # generating subsequent mask for target
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(trg.shape[1]).to(
                trg.device
            )

        # Padding mask
        trg_pad_mask = self.make_len_mask(trg)

        # Getting postional encoding for target
        trg = self.decoder(trg)
        trg = self.query_pos(trg.permute(1, 0, 2))
        #         trg = trg.permute(1,0,2)
        output = self.transformer_decoder(
            trg,
            h,
            tgt_mask=self.trg_mask,
            tgt_key_padding_mask=trg_pad_mask.permute(1, 0),
        )

        #         output = self.transformer(h.permute(2, 0, 1), trg.permute(1,0,2), tgt_mask=self.trg_mask,
        #                                   tgt_key_padding_mask=trg_pad_mask.permute(1,0))

        return self.vocab(output.transpose(0, 1))


def make_model(
    vocab_len,
    maxlen,
    hidden_dim=256,
    nheads=6,
    num_encoder_layers=2,
    num_decoder_layers=6,
):

    return OCR(
        vocab_len, maxlen, hidden_dim, nheads, num_encoder_layers, num_decoder_layers
    )


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

            self.dataset[self.split]["dt"] = np.array(f[self.split]["dt"])
            self.dataset[self.split]["gt"] = np.array(f[self.split]["gt"])

            #             randomize = np.arange(len(self.dataset[self.split]['gt']))
            #             np.random.seed(42)
            #             np.random.shuffle(randomize)

            #             self.dataset[self.split]['dt'] = self.dataset[self.split]['dt'][randomize]
            #             self.dataset[self.split]['gt'] = self.dataset[self.split]['gt'][randomize]

            # decode sentences from byte
            self.dataset[self.split]["gt"] = [
                x.decode() for x in self.dataset[self.split]["gt"]
            ]

        self.size = len(self.dataset[self.split]["gt"])

    def __getitem__(self, i):
        img = self.dataset[self.split]["dt"][i]

        # making image compatible with resnet
        #         img = cv2.transpose(img)
        #         img = np.repeat(img[..., np.newaxis],3, -1).astype("float32")
        #         img = pp.normalization(img).astype("float32")

        if self.transform is not None:
            aug = self.transform(image=img)
            img = aug["image"]

        #             img = self.transform(img)

        #         print(self.dataset[self.split]['gt'][i])
        y_train = self.tokenizer.encode(self.dataset[self.split]["gt"][i])

        # padding till max length
#         y_train = np.pad(y_train, (0, self.tokenizer.maxlen - len(y_train)))

        gt = torch.Tensor(y_train)

        return img, gt

    def __len__(self):
        return self.size


class Tokenizer:
    """Manager tokens functions and charset/dictionary properties"""

    def __init__(self, chars, max_text_length=630):
        self.PAD_TK, self.UNK_TK, self.SOS, self.EOS = "¶", "¤", "SOS", "EOS"
        self.chars = (
            [self.PAD_TK] + [self.UNK_TK] + [self.SOS] + [self.EOS] + list(chars)
        )
        self.PAD = self.chars.index(self.PAD_TK)
        self.UNK = self.chars.index(self.UNK_TK)

        self.vocab_size = len(self.chars)
        self.maxlen = max_text_length

    def encode(self, text):
        """Encode text to vector"""
        #         text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
        #         text = " ".join(text.split())

        #         groups = ["".join(group) for _, group in groupby(text)]
        #         text = "".join([self.UNK_TK.join(list(x)) if len(x) > 1 else x for x in groups])
        text = str(text)
        encoded = []

        text = ["SOS"] + list(text.strip()) + ["EOS"]
        for item in text:
            index = self.chars.index(item)
            index = self.UNK if index == -1 else index
            encoded.append(index)

        return np.asarray(encoded)

    def decode(self, text):
        """Decode vector to text"""

        decoded = "".join([self.chars[int(x)] for x in text if x > -1])
        decoded = self.remove_tokens(decoded)
        #         decoded = pp.text_standardize(decoded)

        return decoded

    def remove_tokens(self, text):
        """Remove tokens (PAD) from text"""

        return text.replace(self.PAD_TK, "").replace(self.UNK_TK, "")


import os
import datetime
import string
args = parse_args()
print(args)

batch_size = args.batch_size
epochs = args.epochs


with open(args.charset_base, 'r') as f:
    data = json.load(f)    
    
max_text_length = data['max_length']
charset_base = data['vocab']
# charset_base = ":îÿm<Ö5a,wjRĪLq#f-C1ùÀ8S7âēcûêD0X²Eȳ\"9ū!œā.iyëGk(pō vhztTß̈çHr\nV°lN)à%ôx€n3—bJU¤_Psoè¬{*g;}öQZ?äO/üÉKW&dA6¾BFY>2=eu+'4Mé"
# charset_base = string.printable[:36].lower() + string.printable[36+26:95].lower()
print("charset:", charset_base)

device = torch.device("cuda:{}".format(args.device))
tokenizer = Tokenizer(charset_base, max_text_length)

transform_train = albumentations.Compose(
    [
        albumentations.OneOf(
            [
                albumentations.MotionBlur(p=1, blur_limit=7),
                albumentations.OpticalDistortion(p=1, distort_limit=0.05),
                albumentations.GaussNoise(p=1, var_limit=(10.0, 100.0)),
                albumentations.Equalize(p=1),
                albumentations.Solarize(p=1, threshold=50),
                albumentations.RandomBrightnessContrast(p=1, brightness_limit=0.2),
                albumentations.Downscale(p=1, scale_min=0.8, scale_max=0.9),
            ],
            p=0.5,
        ),
        #         albumentations.Resize(224,224),
        albumentations.Normalize(),
        albumentations.pytorch.ToTensorV2(),
    ]
)


transform_valid = albumentations.Compose(
    [
        #         albumentations.Resize(224,224),
        albumentations.Normalize(),
        albumentations.pytorch.ToTensorV2(),
    ]
)
num_encoder_layers = 2
num_decoder_layers = 6

ddp_model = make_model(
    vocab_len=tokenizer.vocab_size,
    maxlen=tokenizer.maxlen,
    hidden_dim=384,
    nheads=6,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
)
def collate_fn(batch):
#     print(batch[0][0].shape,batch[1][0].shape)
    imgs, labels = zip(*batch)
    
    # Convert labels to 2D tensors
    labels = [label for label in labels]
    imgs = [img for img in imgs]
    
    # Find max length of labels
    max_len = max(len(label) for label in labels)
    
    # Pad labels to max length
    labels = [torch.nn.functional.pad(label, (0, max_len - len(label)), 'constant', 0) for label in labels]
    
    # Stack labels into a single tensor
    labels = torch.stack(labels)
    imgs = torch.stack(imgs)
    return imgs, labels


file_path = args.file_path
val_loader = torch.utils.data.DataLoader(
    DataGenerator("{}".format(file_path), "valid", transform_valid, tokenizer),
    batch_size=batch_size * 4,
    collate_fn=collate_fn,
    num_workers=1,
)
train_loader = torch.utils.data.DataLoader(
    DataGenerator("{}".format(file_path), "train", transform_train, tokenizer),
    batch_size=batch_size,
    num_workers=1,
    collate_fn=collate_fn,
    shuffle=True,
)


# train_loader = torch.utils.data.DataLoader(DataGenerator("data/augmented_images.hdf5",'Wednesday15June2022121821AM_000000_1.jpg',transform_train, tokenizer), batch_size=batch_size, num_workers=1,shuffle=True)


# init_funcs = {
# 1: lambda x: torch.nn.init.normal_(x, mean=0., std=1.), # can be bias
# 2: lambda x: torch.nn.init.xavier_normal_(x, gain=1.), # can be weight
# 3: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.), # can be conv1D filter
# 4: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.), # can be conv2D filter
# "default": lambda x: torch.nn.init.constant(x, 1.), # everything else
# }
# for p in ddp_model.parameters():
#     init_func = init_funcs.get(len(p.shape), init_funcs["default"])
#     init_func(p)


model = ddp_model.to(device)


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


smoothing = 0.1
criterion = LabelSmoothing(
    size=tokenizer.vocab_size, padding_idx=0, smoothing=smoothing
)
criterion.to(device)
lr = args.lr  # learnig rte
# backbone_lr = .0003


scheduler_factor = 0.8


if args.finetune:
    checkpoint = torch.load("saved_models/{}".format(args.finetune), map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0004)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=scheduler_factor)

def train(model, criterion, optimiser, dataloader, scaler):

    model.train()
    total_loss = 0
    for batch, (
        imgs,
        labels_y,
    ) in enumerate(dataloader):        
        imgs = imgs.to(device).float()
        labels_y = labels_y.to(device).long()
        optimiser.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(imgs, labels_y[:, :-1])
            loss = criterion(
                output.log_softmax(-1).contiguous().view(-1, tokenizer.vocab_size),
                labels_y[:, 1:].contiguous().view(-1).long(),
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #             loss.backward()
        #             torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
        #             optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model,
    criterion,
    dataloader,
):

    model.eval()
    epoch_loss = 0
    cer = 0
    
    with torch.no_grad():
        for batch, (
            imgs,
            labels_y,
        ) in enumerate(dataloader):            
            imgs = imgs.to(device)
            labels_y = labels_y.to(device)
            output = model(imgs.float(), labels_y.long()[:, :-1])
            #             o = output.argmax(-1)
            #             predicts = list(map(lambda x : tokenizer.decode(x).replace('SOS','').replace('EOS',''),o))
            #             gt = list(map(lambda x : tokenizer.decode(x).replace('SOS','').replace('EOS',''),labels_y))
            #             cer += evaluation.ocr_metrics(predicts=predicts,
            #                                    ground_truth=gt)[0]

            loss = criterion(
                output.log_softmax(-1).contiguous().view(-1, tokenizer.vocab_size),
                labels_y[:, 1:].contiguous().view(-1).long(),
            )

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


# train model


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


target_path = args.target_path
name_file = args.name_file + args.file_path.replace(".hdf5","")
best_CER = np.inf
best_valid_loss = np.inf
scaler = torch.cuda.amp.GradScaler()

print("Started training")
print(name_file)
c = 0
for epoch in range(epochs):

    start_time = time.time()

    train_loss = train(model, criterion, optimizer, train_loader, scaler)
    valid_loss = evaluate(model, criterion, val_loader)
    epoch_mins, epoch_secs = epoch_time(start_time, time.time())
    print(f"Epoch: {epoch+1:02}", "learning rate{}".format(lr_scheduler.get_last_lr()))
    print(f"Time: {epoch_mins}m {epoch_secs}s")
    print(f"Train Loss: {train_loss:.3f}")
    print(f"Val   Loss: {valid_loss:.3f}")

    c += 1
    save = 1
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        print("saving it", best_valid_loss)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": valid_loss,
                "best_loss": best_valid_loss,
            },
            target_path + name_file + "best_loss.pt",
        )

        save=0
        c = 0
        
    if save and not args.finetune:        
        print("saving for last loss")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": valid_loss,
                "best_loss": best_valid_loss,
            },
            target_path + name_file + "last.pt",
        )
        
#     if epoch % 20 == 0 and epoch != 0:
#         torch.save(
#             {
#                 "epoch": epoch,
#                 "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": optimizer.state_dict(),
#                 "loss": valid_loss,
#                 "best_loss": best_valid_loss,
#                 "c": c,
#             },
#             target_path + name_file + "_{}.pt".format(epoch + 1),
#         )

    if c > 4:
        # decrease lr if loss does not deacrease after 5 steps
        lr_scheduler.step()
        c = 0
