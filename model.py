import torch
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_

import torch.nn.functional as F

from transformers import BertModel, ViTModel, BertTokenizer

from loss import HardTripletLoss, TripletLoss



class AverageSims(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count)


class VarianceSims(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.var = 0
        self.sum = 0
        self.count = 0

    def update(self, val, avg, n=1):
        self.val = val
        self.sum += (val * n - avg) ** 2
        self.count += n
        self.var = self.sum / (self.count - 1)



class Momentum_model(nn.Module):

    def __init__(self, opt):
        super(Momentum_model, self).__init__()

        ##########################
        self.opt = opt
        self.avg_sims_tool_pos = AverageSims()
        self.var_sims_tool_pos = VarianceSims()

        self.avg_sims_tool_neg = AverageSims()
        self.var_sims_tool_neg = VarianceSims()
        ##########################

        self.queue_size = opt.queue_size                                                                  # 65536
        self.momentum = opt.momentum    
        self.margin = opt.margin                                                                          # 0.995

        self.visual_encoder = ViTModel.from_pretrained("google/vit-base-patch16-384")
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")

        # create momentum models
        self.visual_encoder_m = ViTModel.from_pretrained("google/vit-base-patch16-384")
        self.text_encoder_m = BertModel.from_pretrained("bert-base-uncased")        

        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.text_encoder,self.text_encoder_m],
                           ]
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(self.visual_encoder.config.hidden_size, opt.queue_size))
        self.register_buffer("text_queue", torch.randn(self.text_encoder.config.hidden_size, opt.queue_size))
        self.register_buffer("idx_queue", torch.full((1,opt.queue_size),-100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)


    def forward(self, images, targets, targets_attention, idx, epoch, logger):

        # Forward
        img_emb = self.visual_encoder(pixel_values=images)
        cap_emb = self.text_encoder(input_ids=targets, attention_mask=targets_attention)

        #img_output = img_emb.pooler_output
        img_output = img_emb.last_hidden_state.mean(1)
        #cap_output = cap_emb.pooler_output
        cap_output = cap_emb.last_hidden_state.mean(1)

        image_feat = F.normalize(img_output, dim=-1)
        text_feat = F.normalize(cap_output, dim=-1)

        idx = idx.view(-1,1)                                                                                     # [b, 1]
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()],dim=1)  
        pos_idx = torch.eq(idx, idx_all)                                                                # [b, 65536+b] 找出动量模型中img相同的，因为flickr中是一个图像对5个文本
        #sim_targets = pos_idx / pos_idx.sum(1,keepdim=True) 

        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(pixel_values=images) 
            image_feat_m = F.normalize(image_embeds_m.last_hidden_state.mean(1), dim=-1)  
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)        

            text_output_m = self.text_encoder_m(input_ids=targets, attention_mask=targets_attention)    
            text_feat_m = F.normalize(text_output_m.last_hidden_state.mean(1), dim=-1) 
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

        sim_i2t = image_feat @ text_feat_all
        sim_t2i = text_feat @ image_feat_all          

        if epoch >= 1:

            loss_hardtripletloss = HardTripletLoss(image_feat, text_feat, sim_i2t, sim_t2i, self.margin, pos_idx, epoch, self.avg_sims_tool_pos, self.var_sims_tool_pos, self.avg_sims_tool_neg, self.var_sims_tool_neg, self.opt.nums_right_sims)

            loss_ita = (loss_hardtripletloss) / image_feat.shape[0]

            logger.update('Le', loss_ita.item(), images.size(0))

            self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)

        else:
            loss_tripletloss = TripletLoss(image_feat, text_feat, self.margin)

            loss_ita = (loss_tripletloss) / image_feat.shape[0]

            logger.update('Le', loss_ita.item(), images.size(0))

        return loss_ita

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feats, text_feats, idxs):
        # gather keys before updating queue
        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr  



class FNE(object):

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip

        self.model = Momentum_model(opt)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if torch.cuda.is_available():
            self.model.cuda()
            cudnn.benchmark = True
        
        params = list(self.model.parameters())
        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.model.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict[0])

    def train_start(self):
        """switch to train mode
        """
        self.model.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.model.eval()

    def train_emb(self, images, texts, ids, epoch, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        text_input = self.tokenizer(texts, padding='longest', max_length=30, return_tensors="pt")

        targets = text_input.input_ids
        targets_attention = text_input.attention_mask

        if torch.cuda.is_available():
            images = images.cuda()
            targets = targets.cuda()
            targets_attention = targets_attention.cuda()

            ids = ids.cuda()  

        # compute the embeddings
        loss = self.model(images, targets, targets_attention, ids, epoch, self.logger)

        # measure accuracy and record loss
        self.optimizer.zero_grad()

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
