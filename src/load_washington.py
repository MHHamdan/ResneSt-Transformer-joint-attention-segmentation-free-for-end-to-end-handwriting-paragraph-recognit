import os
import glob
import json
import string

import scipy as sp
import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage.transform import resize
import h5py
from scipy.io import loadmat

def replace_tokens(text, tokens):
    for t in tokens:
        text = text.replace(t, '')
        
    return text
    
def lens_by_split(data):
    ntrain, nval, ntest = 0, 0, 0
    for datum in data:
        if datum['split'] == 'train':
            ntrain += 1
        elif datum['split'] == 'val':
            nval += 1
        elif datum['split'] == 'test':
            ntest += 1
        else:
            raise ValueError
            
    return (ntrain, nval, ntest)
    
def _prepare_img(f):
    img = rgb2gray(imread(f))
    try: 
        img = img_as_ubyte(resize(img, (60, 160)))
        img = np.invert(img)
        
    except ValueError:
        img = img_as_ubyte(resize(img, (60, 160)))
        print("failed to preprocess", f.split('/')[-1])
            
    img = img[np.newaxis, :, :]
    return img

def load_washington(fold=1, root=""):
    """
    Loads the washington dataset
    
    fold: which fold of the four washington dataset folds to load
    root: path to directory where data is stored and where some caching is done
    """
    output_json = 'washington_fold_%d.json' % fold
    output_json = os.path.join(root, output_json)

    if not os.path.exists(output_json):
        files = sorted(glob.glob(os.path.join(root, 'gw_20p_wannot/*.tif')))
    
        #Directory to store segmented washington
        word_dir = os.path.join(root, 'words/')
        if not os.path.exists(word_dir):
            os.makedirs(word_dir)
        
        img_index = 0
        word_files = []
        for f in files:
            p, fi = os.path.split(f)
            bf = os.path.join(p, fi[:-4] + '_boxes.txt')

            #Segment washington and save to word_dir
            img = imread(f)
            h, w = img.shape
            with open(bf) as ff:
                boxes = ff.readlines()
            
            for b in boxes[1:]: #First line is not a box annotation
                b = [float(b) for b in b.split()]
                x1 = int(np.round(b[0] * (w-1) + 1))
                x2 = int(np.round(b[1] * (w-1) + 1))
                y1 = int(np.round(b[2] * (h-1) + 1))
                y2 = int(np.round(b[3] * (h-1) + 1))
                word = img[y1:y2, x1:x2]
                
                word_file = os.path.join(word_dir, 'word_%d.png' % img_index)
                word_files.append(word_file)
                
                imsave(word_file, word)
                img_index += 1

        #Load annotations            
        with open(os.path.join(root, 'gw_20p_wannot/annotations.txt')) as ff: #Use our own, correct annotations
            lines = ff.readlines()
        
        # Normalize texts to [a-z0-9]
        alphabet = string.ascii_lowercase + string.digits
        texts = [l[:-1] for l in lines]        
        ntexts = [replace_tokens(text.lower(), [t for t in text if t not in alphabet]) for text in texts]
        
        #Load folds supplied by almazan et al
        dic = loadmat(os.path.join(root, "GW_words_indexes_sets_fold%d.mat" % fold))
        splits = {}
        splits['train'] = np.squeeze(dic['idxTrain'])
        splits['val'] = np.squeeze(dic['idxValidation'])
        splits['test'] = np.squeeze(dic['idxTest'])
        
        data = []
        for i, (f, text) in enumerate(zip(word_files, ntexts)):
            datum = {}
            datum['file'] = f
            datum['text'] = text
            datum['split'] = [k for k, v in splits.items() if v[i] == 1][0]
            if datum['split'] == 'val': # Use all available data to train
                datum['split'] = 'train'
            data.append(datum)
        
        with open(output_json, 'w') as f:
            json.dump(data, f)
    
    else: #otherwise load the json
        with open(output_json) as f:
            data = json.load(f)
    
    return data  


load_washington(root = "../raw/")