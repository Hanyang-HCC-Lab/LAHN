import numpy as np
import json
import random
import os
from easydict import EasyDict as edict
import time

import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F

import train_config as train_config
from dataset_impcon import get_dataloader
from util import iter_product
from sklearn.metrics import f1_score
import loss_impcon as loss
from model import primary_encoder_v2_no_pooler_for_con, weighting_network

from transformers import AdamW,get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup 

from tqdm import tqdm
import pandas as pd
import copy

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


# Credits https://github.com/varsha33/LCL_loss
def train(epoch,train_loader,model_main,loss_function,optimizer,lr_scheduler,log, model_momentum, queue_features, queue_labels):

    model_main.cuda()
    model_main.train()
    if log.param.momentum > 0.0:
        model_momentum.cuda()
        model_momentum.eval()

    total_true,total_pred_1,acc_curve_1 = [],[],[]
    loss_curve = []
    train_loss_1 = 0
    total_epoch_acc_1 = 0
    steps = 0
    start_train_time = time.time()

    sim = Similarity(log.param.temperature)
    
    if log.param.w_aug:
        if log.param.w_double:
            train_batch_size = log.param.train_batch_size*3
        else:
            train_batch_size = log.param.train_batch_size*2 # only for w_aug
    else:
        train_batch_size = log.param.train_batch_size
    print("train with aug:", log.param.w_aug)
    print("train with double aug:", log.param.w_double)
    print("train with separate double aug:", log.param.w_separate)
    print("loss with sup(using label info):", log.param.w_sup)
    print("len(train_loader):", len(train_loader))
    print("train_batch_size including augmented posts/implications:", train_batch_size)
    if log.param.w_separate:
        assert log.param.w_double, "w_double should be set to True for w_separate=True option"
    

    cls_tokens = []

    # for idx,batch in tqdm(enumerate(tqdm(train_loader))):
    for idx,batch in enumerate(train_loader):

        if "ihc" in log.param.dataset or "sbic" in log.param.dataset or 'dynahate' in log.param.dataset or 'toxigen' in log.param.dataset:
            text_name = "post"
            label_name = "label"
        else:
            raise NotImplementedError

        text = batch[text_name]
        attn = batch[text_name+"_attn_mask"]
        label = batch[label_name]
        label = torch.tensor(label)
        label = torch.autograd.Variable(label).long()


        if (label.size()[0] is not train_batch_size):
            continue

        if torch.cuda.is_available():
            text = text.cuda()
            attn = attn.cuda()
            label = label.cuda()
            # print(label)

        #####################################################################################
        if log.param.w_aug: # text split
            if log.param.w_double:
                pass

            else:
                if log.param.loss_type == "SemiCon":
                    pass

                else:
                    # W_AUG, SCL

                    assert log.param.train_batch_size == label.shape[0] // 2
                    assert label.shape[0] % 2 == 0
                    original_label, augmented_label = torch.split(label, [log.param.train_batch_size, log.param.train_batch_size], dim=0)
                    only_original_labels = original_label

                    original_text, augmented_text = torch.split(text, [log.param.train_batch_size, log.param.train_batch_size], dim=0)
                    original_attn, augmented_attn = torch.split(attn, [log.param.train_batch_size, log.param.train_batch_size], dim=0)


                    original_last_layer_hidden_states, original_supcon_feature_1 = model_main.get_cls_features_ptrnsp(original_text, original_attn) # #v2
                    # _, augmented_supcon_feature_1 = model_main.get_cls_features_ptrnsp(augmented_text,augmented_attn) # #v2


                    if log.param.loss_type in ["CE", "UCL", "SCL", "SupConLoss_Original"]:
                        _, augmented_supcon_feature_1 = model_main.get_cls_features_ptrnsp(original_text, original_attn)

                    elif log.param.loss_type=="ImpCon":    
                        _, augmented_supcon_feature_1 = model_main.get_cls_features_ptrnsp(augmented_text,augmented_attn)

                    elif log.param.aug_type =="Dropout": 
                        _, augmented_supcon_feature_1 = model_momentum.get_cls_features_ptrnsp(original_text, original_attn)

                    elif log.param.aug_type =="Augmentation": 
                        # _, augmented_supcon_feature_1 = model_main.get_cls_features_ptrnsp(augmented_text,augmented_attn) # #v2
                        _, augmented_supcon_feature_1 = model_momentum.get_cls_features_ptrnsp(augmented_text,augmented_attn) # #v2

                    # Momentum Features
                    if log.param.loss_type not in ["CE", "UCL", "SCL", "SupConLoss_Original", "ImpCon", "AugCon"]:
                        with torch.no_grad():
                            _, original_supcon_feature_momentum = model_momentum.get_cls_features_ptrnsp(original_text, original_attn) # #v2
                            # original_supcon_feature_momentum = original_supcon_feature_momentum.detach()
                            _, augmented_supcon_feature_momentum = model_momentum.get_cls_features_ptrnsp(augmented_text,augmented_attn) # #v2
                            # momentum_feature = torch.cat([original_supcon_feature_momentum, augmented_supcon_feature_momentum], dim=0)

                            queue_features = torch.cat((queue_features, original_supcon_feature_momentum.cpu()), 0)
                            queue_labels = torch.cat((queue_labels, original_label.view([-1, 1]).cpu()), 0)
                            if queue_features.shape[0] > log.param.queue_size:
                                queue_features = queue_features[log.param.train_batch_size: , :]
                                queue_labels = queue_labels[log.param.train_batch_size: , :]

                            moco_features, moco_labels = queue_features, queue_labels


                                
                    k = log.param.hard_neg_k  

                    anchor_labels = torch.zeros((log.param.train_batch_size, log.param.train_batch_size), device='cuda:0')
                    for anchor_idx, i in enumerate(only_original_labels):
                        if i==1:
                            anchor_labels[anchor_idx] = only_original_labels
                        elif i==0:
                            anchor_labels[anchor_idx] = 1 - only_original_labels
                    
                    if log.param.loss_type not in ["CE", "UCL", "SCL", "SupConLoss_Original", "ImpCon"]:
                        with torch.no_grad():
                            if moco_features.shape[0] > int(log.param.queue_size) * 1/4: 

                                moco_features = moco_features.cuda()
                                moco_labels = moco_labels.squeeze().cuda()
                                moco_original_labels = moco_labels.squeeze().cuda()
                                # Anchors * MoCo features Cos_Sim Matrix
                                moco_sim = sim(original_supcon_feature_1.unsqueeze(1), moco_features.unsqueeze(0))
                                labels_concat_moco = torch.cat((only_original_labels, moco_labels), dim=0) # batch label + moco label 

                                target = torch.zeros((only_original_labels.size(0), labels_concat_moco.size(0)), device='cuda:0')

                                for m_idx, i in enumerate(only_original_labels):
                                    if i==1:
                                        target[m_idx] = labels_concat_moco.view(labels_concat_moco.shape[0])
                                    elif i==0:
                                        target[m_idx] = 1 - labels_concat_moco.view(labels_concat_moco.shape[0])

                                ### Hard Neg Sampling 
                                anchor_labels = target[:, :only_original_labels.shape[0]]
                                moco_labels = target[:, only_original_labels.shape[0]:] 
                                ### Pos sample 
                                cos_sim_moco_hard_neg = moco_sim * (1 - moco_labels) 

                                # Weight 
                                if log.param.moco_weight == "True":    
                                    neg_weight = model_momentum(moco_features)
                                    # neg_weight = model_main(moco_features)
                                    neg_weight = F.softmax(neg_weight,dim=1)
                                    neg_weight = neg_weight[:, only_original_labels].T 
                                    # weighting to Similarity matrix
                                    cos_sim_moco_hard_neg = cos_sim_moco_hard_neg * neg_weight
                                elif log.param.moco_weight == "False":
                                    pass    

                                cos_sim_moco_hard_neg[(cos_sim_moco_hard_neg == -0.0) | (cos_sim_moco_hard_neg == 0.0)] = -999
                                
                                cos_sim_moco_topk_hard_neg = torch.topk(cos_sim_moco_hard_neg, k, dim=1) #indices, values
                                hard_neg_idx = cos_sim_moco_topk_hard_neg.indices
                                
                                hard_neg_features = torch.zeros((hard_neg_idx.size(0), hard_neg_idx.size(1), moco_features.size(1)))
                                for batch_idx in range(hard_neg_idx.size(0)):
                                        for k_idx in range(hard_neg_idx.size(1)):
                                            hard_neg_features[batch_idx, k_idx] = moco_features[hard_neg_idx[batch_idx][k_idx]]
                                
                               
                    
                    supcon_feature_1 = torch.cat([original_supcon_feature_1, augmented_supcon_feature_1], dim=0)
                    assert original_last_layer_hidden_states.shape[0] == log.param.train_batch_size

                    pred_1 = model_main(original_last_layer_hidden_states)
                    



        else:
            assert log.param.train_batch_size == label.shape[0]
            only_original_labels = label
            last_layer_hidden_states, supcon_feature_1 = model_main.get_cls_features_ptrnsp(text,attn) # #v2
            pred_1 = model_main(last_layer_hidden_states)


        if log.param.w_aug and log.param.w_sup:
            if log.param.w_double:
                if log.param.w_separate:
                    raise NotImplementedError
                else:
                    loss_1 = (loss_function["lambda_loss"]*loss_function["ce_loss"](pred_1,only_original_labels)) + ((1-loss_function["lambda_loss"])*loss_function["contrastive_for_double"](supcon_feature_1,label)) 
            else:
                # W_AUG, SCL
                if log.param.loss_type == "CE":
                    loss_1 = (loss_function["lambda_loss"]*loss_function["ce_loss"](pred_1,only_original_labels))

                elif log.param.loss_type == "UCL":
                    loss_1 = (loss_function["lambda_loss"]*loss_function["ce_loss"](pred_1,only_original_labels)) + ((1-loss_function["lambda_loss"])*loss_function["ucl"](supcon_feature_1))

                elif log.param.loss_type == "ImpCon":
                    loss_1 = (loss_function["lambda_loss"]*loss_function["ce_loss"](pred_1,only_original_labels)) + ((1-loss_function["lambda_loss"])*loss_function["ImpCon"](supcon_feature_1))
                
                elif log.param.loss_type == "SCL": 
                    loss_1 = (loss_function["lambda_loss"]*loss_function["ce_loss"](pred_1,only_original_labels)) + ((1-loss_function["lambda_loss"])*loss_function["SupConLoss"](supcon_feature_1, label))
                elif log.param.loss_type == "SupConLoss_Original": 
                    loss_1 = (loss_function["lambda_loss"]*loss_function["ce_loss"](pred_1,only_original_labels)) + ((1-loss_function["lambda_loss"])*loss_function["SupConLoss_Original"](supcon_feature_1, label))

                elif log.param.loss_type == "Ours": 

                    if moco_features.shape[0] > int(log.param.queue_size) * 1/4:
                        loss_1 = (loss_function["lambda_loss"]*loss_function["ce_loss"](pred_1,only_original_labels)) + ((1-loss_function["lambda_loss"])*loss_function["Ours"](supcon_feature_1, label, None, hard_neg_features, None, moco_labels, anchor_labels, moco_features, moco_original_labels))
                    else:
                        loss_1 = (loss_function["lambda_loss"]*loss_function["ce_loss"](pred_1,only_original_labels)) + ((1-loss_function["lambda_loss"])*loss_function["Ours"](supcon_feature_1, label, None, None, None, None, anchor_labels))
                

        elif log.param.w_aug and not log.param.w_sup:
            if log.param.w_double:
                if log.param.w_separate:
                    loss_1 = (loss_function["lambda_loss"]*loss_function["ce_loss"](pred_1,only_original_labels)) + ((0.5*(1-loss_function["lambda_loss"]))*loss_function["contrastive"](supcon_feature_1)) + ((0.5*(1-loss_function["lambda_loss"]))*loss_function["contrastive"](supcon_feature_2)) 
                else:
                    loss_1 = (loss_function["lambda_loss"]*loss_function["ce_loss"](pred_1,only_original_labels)) + ((1-loss_function["lambda_loss"])*loss_function["contrastive_for_double"](supcon_feature_1))
            else:
                loss_1 = (loss_function["lambda_loss"]*loss_function["ce_loss"](pred_1,only_original_labels)) + ((1-loss_function["lambda_loss"])*loss_function["contrastive"](supcon_feature_1)) 
        else: # log.param.w_aug == False
            loss_1 = loss_function["ce_loss"](pred_1,only_original_labels)


        loss = loss_1
        train_loss_1  += loss_1.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model_main.parameters(), max_norm=1.0)

        optimizer.step()
        model_main.zero_grad()

        lr_scheduler.step()
        optimizer.zero_grad()

        steps += 1


        if steps % 100 == 0:
            print(f'Epoch: {epoch:02}, Idx: {idx+1}, Training Loss_1: {loss_1.item():.6f}, Time taken: {((time.time()-start_train_time)/60): .2f} min')
            start_train_time = time.time()

        true_list = only_original_labels.data.detach().cpu().tolist()
        total_true.extend(true_list)

        num_corrects_1 = (torch.max(pred_1, 1)[1].view(only_original_labels.size()).data == only_original_labels.data).float().sum()
        pred_list_1 = torch.max(pred_1, 1)[1].view(only_original_labels.size()).data.detach().cpu().tolist()

        total_pred_1.extend(pred_list_1)

        acc_1 = 100.0 * (num_corrects_1/log.param.train_batch_size)
        acc_curve_1.append(acc_1.item())

        loss_curve.append(loss_1.item())

        total_epoch_acc_1 += acc_1.item()
        
        # momentum update
        if log.param.momentum > 0.:
            with torch.no_grad():
                for param, param_m in zip(model_main.parameters(), model_momentum.parameters()):
                    param_m.data = param_m.data * log.param.momentum + param.data * (1. - log.param.momentum)
                    param_m.requires_grad = False


    print(train_loss_1/len(train_loader))
    print(total_epoch_acc_1/len(train_loader))

    return train_loss_1/len(train_loader), total_epoch_acc_1/len(train_loader), acc_curve_1, cls_tokens, loss_curve, queue_features, queue_labels


def test(test_loader, model_main, log):
    model_main.eval()
    
    total_pred_1,total_true,total_pred_prob_1 = [],[],[]
    save_pred = {"true":[],"pred_1":[],"pred_prob_1":[],"feature":[]}

    total_feature = []
    total_num_corrects = 0
    total_num = 0
    print(len(test_loader))

    with torch.no_grad():
        for idx,batch in enumerate(test_loader):
            if "ihc" in log.param.dataset:
                text_name = "post"
                label_name = "label"
            elif "dynahate" in log.param.dataset:
                text_name = "post"
                label_name = "label"
            elif "sbic" in log.param.dataset:
                text_name = "post"
                label_name = "label"
            elif "toxigen" in log.param.dataset:
                text_name = "post"
                label_name = "label"
            else:
                text_name = "cause"
                label_name = "emotion"
                raise NotImplementedError

            text = batch[text_name]
            attn = batch[text_name+"_attn_mask"]
            label = batch[label_name]
            label = torch.tensor(label)
            label = torch.autograd.Variable(label).long()

            if torch.cuda.is_available():
                text = text.cuda()
                attn = attn.cuda()
                label = label.cuda()

            last_layer_hidden_states, supcon_feature_1 = model_main.get_cls_features_ptrnsp(text,attn) # #v2
            pred_1 = model_main(last_layer_hidden_states)
            softmaxed_tensor = F.softmax(pred_1, dim=1)
            
            num_corrects_1 = (torch.max(pred_1, 1)[1].view(label.size()).data == label.data).float().sum()

            pred_list_1 = torch.max(pred_1, 1)[1].view(label.size()).data.detach().cpu().tolist()
            true_list = label.data.detach().cpu().tolist()

            total_num_corrects += num_corrects_1.item()
            total_num += text.shape[0]


            temp_result = pd.DataFrame({"post":text.data.detach().cpu().tolist(),
                                        "label":true_list,
                                        "pred":pred_list_1,
                                        "softmax": softmaxed_tensor.detach().cpu().tolist()})

            total_pred_1.extend(pred_list_1)
            total_true.extend(true_list)
            total_feature.extend(supcon_feature_1.data.detach().cpu().tolist())
            total_pred_prob_1.extend(pred_1.data.detach().cpu().tolist())

    f1_score_1 = f1_score(total_true,total_pred_1, average="macro")
    f1_score_1_w = f1_score(total_true,total_pred_1, average="weighted")
    f1_score_1 = {"macro":f1_score_1,"weighted":f1_score_1_w}

    total_acc = 100 * total_num_corrects / total_num

    save_pred["true"] = total_true
    save_pred["pred_1"] = total_pred_1

    save_pred["feature"] = total_feature
    save_pred["pred_prob_1"] = total_pred_prob_1

    return total_acc,f1_score_1,save_pred

def cl_train(log):

    np.random.seed(log.param.SEED)
    random.seed(log.param.SEED)
    torch.manual_seed(log.param.SEED)
    torch.cuda.manual_seed(log.param.SEED)
    torch.cuda.manual_seed_all(log.param.SEED)

    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 


    os.environ["PYTHONHASHSEED"] = str(log.param.SEED)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"



    print("#######################start run#######################")
    print("log:", log)
    train_data,valid_data,test_data = get_dataloader(log.param.train_batch_size,log.param.eval_batch_size,log.param.dataset,w_aug=log.param.w_aug,w_double=log.param.w_double,label_list=None, cls_tokens=None,
                                                     model_type=log.param.model_type)
    print("len(train_data):", len(train_data)) 

    losses = {
            "ucl":loss.UnSupConLoss(temperature=log.param.temperature),
            "Ours":loss.Ours(temperature=log.param.temperature),
            "ImpCon":loss.ImpCon(temperature=log.param.temperature),
            "SupConLoss":loss.SupConLoss(temperature=log.param.temperature),
            "SupConLoss_Original":loss.SupConLoss_Original(temperature=log.param.temperature),
            "ce_loss":nn.CrossEntropyLoss(),
            "bce_loss":nn.BCEWithLogitsLoss(),
            "lambda_loss":log.param.lambda_loss,
            }

    model_run_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    model_main = primary_encoder_v2_no_pooler_for_con(log.param.hidden_size,log.param.label_size,log.param.model_type)


    if log.param.momentum > 0.:
        model_momentum = primary_encoder_v2_no_pooler_for_con(log.param.hidden_size,log.param.label_size,log.param.model_type)
        for param, param_m in zip(model_main.parameters(), model_momentum.parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient 
   
    queue_features = torch.zeros((0, 768))
    queue_labels = torch.zeros((0, 1))

    total_params = list(model_main.named_parameters())
    num_training_steps = int(len(train_data)*log.param.nepoch)
    optimizer_grouped_parameters = [
    {'params': [p for n, p in total_params], 'weight_decay': log.param.decay}]


    optimizer = AdamW(optimizer_grouped_parameters, lr=log.param.main_learning_rate, eps=1e-8)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


    if log.param.run_name != "":
        save_home = "./save/"+str(log.param.SEED)+"/"+log.param.dataset+"/"+log.param.run_name+"/"+log.param.loss_type+log.param.dir_name+"/"+model_run_time+"/"
    else:
        save_home = "./save/"+str(log.param.SEED)+"/"+log.param.dataset+"/"+log.param.loss_type+log.param.dir_name+"/"+model_run_time+"/"

    total_train_acc_curve_1, total_val_acc_curve_1 = [],[]
    total_train_loss_curve = []

    for epoch in range(1, log.param.nepoch + 1):


        if log.param.momentum > 0.0:
            # Queue initialize by epoch
#             train_loss_1,train_acc_1,train_acc_curve_1, cls_tokens, train_loss_curve, None, None = train(epoch,train_data,model_main, losses,optimizer,lr_scheduler, log, model_momentum, queue_features, queue_labels)
            # Queue keep by epoch
            train_loss_1,train_acc_1,train_acc_curve_1, cls_tokens, train_loss_curve, queue_features, queue_labels = train(epoch,train_data,model_main, losses,optimizer,lr_scheduler, log, model_momentum, queue_features, queue_labels)
            val_acc_1,val_f1_1,val_save_pred = test(valid_data, model_main, log)
            test_acc_1,test_f1_1,test_save_pred = test(test_data, model_main, log)
        else:
            train_loss_1,train_acc_1,train_acc_curve_1, cls_tokens, train_loss_curve, _, _ = train(epoch,train_data,model_main, losses,optimizer,lr_scheduler, log, None, None, None)
            val_acc_1,val_f1_1,val_save_pred = test(valid_data, model_main, log)
            test_acc_1,test_f1_1,test_save_pred = test(test_data, model_main, log)


        total_train_acc_curve_1.extend(train_acc_curve_1)
        total_train_loss_curve.extend(train_loss_curve)

        print('====> Epoch: {} Train loss_1: {:.4f}'.format(epoch, train_loss_1))

        os.makedirs(save_home,exist_ok=True)
        with open(save_home+"/acc_curve.json", 'w') as fp:
            json.dump({"train_acc_curve_1":total_train_acc_curve_1}, fp,indent=4)
        with open(save_home+"/loss_curve.json", 'w') as fp:
            json.dump({"train_loss_epoch":epoch}, fp,indent=4)
            json.dump({"train_loss_curve":total_train_loss_curve}, fp,indent=4)

        if epoch == 1:
             best_criterion = 0.0

        ########### best model by val_f1_1["macro"]
        is_best = val_f1_1["macro"] > best_criterion
        best_criterion = max(val_f1_1["macro"],best_criterion)

        print("Best model evaluated by macro f1")
        print(f'Valid Accuracy: {val_acc_1:.3f} Valid F1: {val_f1_1["macro"]:.4f}')
        print(f'Test Accuracy: {test_acc_1:.3f} Test F1: {test_f1_1["macro"]:.4f}')


        if is_best:
            print("======> Best epoch <======")
            log.train_loss_1 = train_loss_1
            log.stop_epoch = epoch
            log.valid_f1_score_1 = val_f1_1
            log.test_f1_score_1 = test_f1_1
            log.valid_accuracy_1 = val_acc_1
            log.test_accuracy_1 = test_acc_1
            log.train_accuracy_1 = train_acc_1


            ## load the model
            with open(save_home+"/log.json", 'w') as fp:
                json.dump(dict(log), fp,indent=4)
            fp.close()

            ###############################################################################
            # save model
            if log.param.save:
                torch.save(model_main.state_dict(), os.path.join(save_home, 'model.pt'))
                print(f"best model is saved at {os.path.join(save_home, 'model.pt')}")
        


##################################################################################################

if __name__ == '__main__':

    tuning_param = train_config.tuning_param

    param_list = [train_config.param[i] for i in tuning_param]
    param_list = [tuple(tuning_param)] + list(iter_product(*param_list)) ## [(param_name),(param combinations)]

    for param_com in param_list[1:]: # as first element is just name

        log = edict()
        log.param = train_config.param

        for num,val in enumerate(param_com):
            log.param[param_list[0][num]] = val

        log.param.label_size = 2
        
        cl_train(log)

