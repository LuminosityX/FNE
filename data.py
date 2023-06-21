import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import re
import os
import json
from PIL import Image

from randaugment import RandomAugment

def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]

        ann_split = ann['image'].split('/')[-1]
        image_path = os.path.join(self.image_root,ann_split)     
        
        #image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']]
    
    

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    

        ann_split = self.ann[index]['image'].split('/')[-1]
        image_path = os.path.join(self.image_root,ann_split)     
        
        #image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index

def get_transform(config):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    train_transform = transforms.Compose([                        
        transforms.RandomResizedCrop(config.image_res,scale=(0.5, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                           'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
        transforms.ToTensor(),
        normalize,
    ])  

    test_transform = transforms.Compose([
        transforms.Resize((config.image_res,config.image_res),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ]) 

    return train_transform, test_transform


def get_loaders(config):

    train_transform, test_transform = get_transform(config)

    train_dataset = re_train_dataset([config.train_file], train_transform, config.image_root)
    val_dataset = re_eval_dataset(config.val_file, test_transform, config.image_root)  
    test_dataset = re_eval_dataset(config.test_file, test_transform, config.image_root)                
    
    datasets = [train_dataset, val_dataset, test_dataset]
    samplers = [None, None, None]
    batch_size=[config.batch_size_train]+[config.batch_size_test]*2
    num_workers=[4,4,4]
    is_trains=[True, False, False]
    collate_fns=[None,None,None]

    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    