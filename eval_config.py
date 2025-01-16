tuning_param  = ["dataset", "load_dir"]
dataset = [
    "ihc_pure", "sbic", "dynahate",
    #  "toxigen",
    ] # dataset for evaluation

root_path = "./save/0/ihc_pure_imp/best/"
load_dir = [
            root_path+"2024_00_00_00",
    ]

train_batch_size = 16
eval_batch_size = 256
hidden_size = 768
# model_type = "bert-base-uncased"
model_type = "roberta-base"
SEED = 0

param = {"dataset":dataset,"train_batch_size":train_batch_size,"eval_batch_size":eval_batch_size,"hidden_size":hidden_size,"dataset":dataset,"SEED":SEED,"model_type":model_type, "load_dir":load_dir}


