dataset = ["ihc_pure_imp", "sbic_imp", "toxigen"]

tuning_param  = ["lambda_loss", "main_learning_rate","train_batch_size","eval_batch_size","nepoch","temperature","SEED","dataset", "decay",
                 "loss_type", "model_type", "momentum", "queue_size", "hard_neg_k" , "aug_type", "aug_enque", "moco_weight"] ## list of possible paramters to be tuned


temperature = [0.05, 0.07, 0.1]
lambda_loss = [0.1]
momentum = [0.999]

aug_type = ["Augmentation", "Dropout"]


queue_size = [512, 1024, 2048]
hard_neg_k = [16, 32, 64]


aug_enque = ["False"]
moco_weight = ["True"]

train_batch_size = [16]
eval_batch_size = [64]

decay = [0.0] # default value of AdamW
main_learning_rate = [2e-05]

hidden_size = 768
nepoch = [3]
run_name = "best"

model_type = ["roberta-base", "bert-base-uncased"]


SEED = [0]

loss_type = ["Ours"]

# loss_type = ["CE", "UCL", "SCL"]

dir_name = "_0401_additional_seed"

w_aug = True
w_double = False
w_separate = False
w_sup = True

save = True
param = {"temperature":temperature,"run_name":run_name,"dataset":dataset,"main_learning_rate":main_learning_rate,"train_batch_size":train_batch_size,"eval_batch_size":eval_batch_size,"hidden_size":hidden_size,"nepoch":nepoch,"dataset":dataset,"lambda_loss":lambda_loss,"loss_type":loss_type,"decay":decay,"SEED":SEED,"model_type":model_type,"w_aug":w_aug, "w_sup":w_sup, "save":save,"w_double":w_double, "w_separate":w_separate,
         "loss_type":loss_type, "dir_name":dir_name, "momentum":momentum, "queue_size":queue_size, "hard_neg_k":hard_neg_k,
         "aug_type":aug_type, "aug_enque":aug_enque, "moco_weight":moco_weight,}
