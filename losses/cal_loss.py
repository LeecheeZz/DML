import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn

def cal_loss(outputs,labels,loss_func): # outputs: [[8, 701], [8, 701], [8, 701]]   label : [1, 8]
    loss = 0
    if isinstance(outputs,list):
        for i in outputs:
            loss += loss_func(i,labels)
        loss = loss/len(outputs)
    else:
        loss = loss_func(outputs,labels)
    return loss

def cal_kl_loss(outputs,outputs2,loss_func): # outputs: [[8, 701], [8, 701], [8, 701]]
    loss = 0
    if isinstance(outputs,list):
        for i in range(len(outputs)):
            loss += loss_func(F.log_softmax(outputs[i], dim=1),
                               F.softmax(Variable(outputs2[i]), dim=1))
        loss = loss/len(outputs)
    else:
        loss = loss_func(F.log_softmax(outputs, dim=1),
                          F.softmax(Variable(outputs2), dim=1))
    return loss

def cal_triplet_loss(outputs,outputs2,labels,loss_func,split_num=8):  # loss_func = Tripletloss(margin=opt.triplet_loss)
    if isinstance(outputs,list): # outputs : [[8, 512], [8, 512], [8, 512]]  outputs2 : [[8, 512], [8, 512], [8, 512]]
        loss = 0
        # print(len(outputs))
        # print("********************************")
        for i in range(len(outputs)): # 3
            # print(outputs[i].shape)
            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            out_concat = torch.cat((outputs[i], outputs2[i]), dim=0) # [16, 512]
            labels_concat = torch.cat((labels,labels),dim=0) # [16]
            loss += loss_func(out_concat,labels_concat)
        loss = loss/len(outputs)
    else:
        out_concat = torch.cat((outputs, outputs2), dim=0)
        labels_concat = torch.cat((labels,labels),dim=0)
        loss = loss_func(out_concat,labels_concat)
    return loss

def cal_infoNCE_loss(outputs, outputs2, loss_func, logit_scale):
    if isinstance(outputs,list): # outputs : [[8, 512], [8, 512], [8, 512]]  outputs2 : [[8, 512], [8, 512], [8, 512]]
        loss = 0
        for i in range(len(outputs)): # 3
            outputs[i] = F.normalize(outputs[i], dim=-1)
            outputs2[i] = F.normalize(outputs2[i], dim=-1)
            logits_per_image1 = logit_scale * outputs[i] @ outputs2[i].T # torch.Size([8, 8]) ABT
            logits_per_image1 = logits_per_image1.cuda()
            logits_per_image2 = logits_per_image1.T  # BAT
            logits_per_image2 = logits_per_image2.cuda()
            labels = torch.arange(len(logits_per_image1), dtype=torch.long, device='cuda')
        
            loss += (loss_func(logits_per_image1, labels) + loss_func(logits_per_image2, labels))/2
        loss = loss/len(outputs)
    else:
        outputs = F.normalize(outputs, dim=-1)
        outputs2 = F.normalize(outputs2, dim=-1)
        logits_per_image1 = logit_scale * outputs @ outputs2.T # torch.Size([8, 8]) ABT
        
        logits_per_image2 = logits_per_image1.T  # BAT
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long)
        
        loss = (loss_func(logits_per_image1, labels) + loss_func(logits_per_image2, labels))/2
    return loss

def cal_dkd_loss(outputs, outputs2, labels, loss_func):
    loss = torch.tensor(0, dtype=torch.float).cuda()
    for i in range(len(outputs)): # 6
        outputs_i = F.log_softmax(outputs[i], dim=1)
        outputs2_i = F.log_softmax(outputs2[i], dim=1)
        loss += loss_func(outputs_i, outputs2_i, labels)
        loss += loss_func(outputs2_i, outputs_i, labels) 
    loss = loss / (len(outputs) * 2)
    return loss
    
def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher,  reduction='sum')
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

class DKD(nn.Module):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, alpha, beta, temperature):
        super(DKD, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(self, logits_student, logits_teacher, target):
        # losses
        loss_dkd = dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )
        
        return loss_dkd