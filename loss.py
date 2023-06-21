import torch

def TripletLoss(im, s, margin):
    scores = im @ s.T
    diagonal = scores.diag().view(im.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    # compare every diagonal score to scores in its column
    # caption retrieval
    cost_s = (margin + scores - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    # image retrieval
    cost_im = (margin + scores - d2).clamp(min=0)

    # clear diagonals
    mask = torch.eye(scores.size(0)) > .5
    if torch.cuda.is_available():
        mask = mask.cuda()
    cost_s = cost_s.masked_fill_(mask, 0)
    cost_im = cost_im.masked_fill_(mask, 0)

    # keep the maximum violating negative for each query
    
    cost_s = cost_s.max(1)[0]
    cost_im = cost_im.max(0)[0]

    return cost_s.sum() + cost_im.sum()


def HardTripletLoss(im, s, sims_i2t, sims_t2i, margin, mask, epoch, avg_sims_tool_pos, var_sims_tool_pos, avg_sims_tool_neg, var_sims_tool_neg, nums_right_sims):
    diagonal_1 = sims_i2t.diag().view(sims_i2t.size(0), 1)
    d1 = diagonal_1.expand_as(sims_i2t)

    diagonal_2 = sims_t2i.diag().view(sims_t2i.size(0), 1)
    d2 = diagonal_2.expand_as(sims_t2i)

    # compare every diagonal score to scores in its column
    # caption retrieval
    cost_s = (margin + sims_i2t - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    # image retrieval
    cost_im = (margin + sims_t2i - d2).clamp(min=0)

    # clear diagonals
        
    cost_s = cost_s.masked_fill_(mask, 0)
    cost_im = cost_im.masked_fill_(mask, 0)

    # keep the maximum violating negative for each query

    if epoch >=1:

        ###
        cost_s_index = cost_s.max(1)[1]
        cost_im_index = cost_im.max(1)[1]
        ###

        average_sims_scores(im, s, avg_sims_tool_pos, var_sims_tool_pos, avg_sims_tool_neg, var_sims_tool_neg)
    
        index_s = choice_negative(d1.detach(), sims_i2t.detach(), mask.detach(), avg_sims_tool_pos, var_sims_tool_pos, avg_sims_tool_neg, var_sims_tool_neg, nums_right_sims, cost_s_index, margin)
        index_im = choice_negative(d2.detach(), sims_t2i.detach(), mask.detach(), avg_sims_tool_pos, var_sims_tool_pos, avg_sims_tool_neg, var_sims_tool_neg, nums_right_sims, cost_im_index, margin)

        cost_s = cost_s.gather(1, index_s)
        cost_im = cost_im.gather(1, index_im)
    else:
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(1)[0]


    return cost_s.sum() + cost_im.sum()


def average_sims_scores(img, txt, avgsims_pos, varsims_pos, avgsims_neg, varsims_neg):
    scores = img @ txt.T
    scores_t = scores.T

    match_index_i2t = scores.sort(1, descending=True)[1]
    match_index_t2i = scores_t.sort(1, descending=True)[1]

    # clear diagonals
    mask = torch.eye(scores.size(0)) > .5
    I = mask.cuda()
    cost_s = scores.masked_fill(I, 0)
    cost_im = scores_t.masked_fill(I, 0)

    cost_s = cost_s.max(1)[0]
    cost_im = cost_im.max(1)[0]

    for i in range(scores.size(0)):
        if match_index_i2t[i][0].item() == i and match_index_t2i[i][0].item() == i:
            avgsims_pos.update(scores[i, i])

            ###
            avgsims_neg.update(cost_s[i])
            avgsims_neg.update(cost_im[i])
            ###


    for i in range(scores.size(0)):
        if match_index_i2t[i][0].item() == i and match_index_t2i[i][0].item() == i:
            varsims_pos.update(scores[i, i], avgsims_pos.avg)

            ###
            varsims_neg.update(cost_s[i], avgsims_neg.avg)
            varsims_neg.update(cost_im[i], avgsims_neg.avg)
            ###

def normal_scores(mean, var, x):

    normal_dist = torch.distributions.Normal(mean, torch.sqrt(var))
    output_y = normal_dist.log_prob(x).exp()

    return output_y

def choice_negative(pos_sims, sims, mask, avg_sims_tool_pos, var_sims_tool_pos, avg_sims_tool_neg, var_sims_tool_neg, nums_right_sims, index, margin):  
    # pos_sims [batch_size, 1]
    # sims [batch_size, memory_size]
    tau = 0.5
    match_para = 10000
    false_margin = 0.999

    neg_scores = torch.exp(-(sims - pos_sims) ** 2 * tau)

    if avg_sims_tool_pos.count > nums_right_sims:


        mean_data_pos = avg_sims_tool_pos.avg
        var_data_pos = var_sims_tool_pos.var

        mean_data_neg = avg_sims_tool_neg.avg
        var_data_neg = var_sims_tool_neg.var

        pos_p = normal_scores(mean_data_pos, var_data_pos, sims)
        neg_p = normal_scores(mean_data_neg, var_data_neg, sims)

        match_p =  (pos_p / (pos_p + (match_para - 1) * neg_p))

        no_match_p = 1 - match_p

        match_p = torch.exp(-(match_p) * tau)


        ###
        match_mask = no_match_p < false_margin

        neg_scores[match_mask] = match_p[match_mask]
        ###

    neg_scores = neg_scores.masked_fill_(mask, 0)

    #margin_mask = pos_sims - margin
    margin_mask = (sims <= (avg_sims_tool_pos.avg-margin))
    neg_scores = neg_scores.masked_fill_(margin_mask, 0)
    neg_scores_sum = neg_scores.sum(1)
    for i, sum_i in enumerate(neg_scores_sum):
        if sum_i == 0:
            neg_scores[i, index[i]] = 1.0


    negative = torch.multinomial(neg_scores, 1)

    return negative