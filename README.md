# LAHN

This is the official implementation of `LAHN: Label-aware Hard Negative Sampling Strategies with Momentum Contrastive Learning for Implicit Hate Speech Detection` (Accepted at ACL Findings 2024).

- Paper: https://aclanthology.org/2024.findings-acl.957/

## Dataset
Data preparation and preprocessing follow the contents of the ImpCon repo. See sections 'Prepare Dataset' and 'Data Preprocess' in the following repository: https://github.com/youngwook06/ImpCon

## Model Training
You can train a model by:
```
python train.py
```
### Train Config
- To train a model with an augmentation method (It means using external knowledge in the paper):
```
aug_type = ["Augmentation"]
```
- To train a model with a dropout method 
```
aug_type = ["Dropout"]
```

## Evaluation
You can evaluate the saved model by:
```
python eval.py
```

## Acknowledgement
This code is based on the code from https://github.com/youngwook06/impcon (Kim et al., 2022) and https://github.com/varsha33/LCL_loss (Suresh et al., 2021).

## Citation
```
@inproceedings{kim-etal-2024-label,
    title = "Label-aware Hard Negative Sampling Strategies with Momentum Contrastive Learning for Implicit Hate Speech Detection",
    author = "Kim, Jaehoon  and
      Jin, Seungwan  and
      Park, Sohyun  and
      Park, Someen  and
      Han, Kyungsik",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.957/",
    doi = "10.18653/v1/2024.findings-acl.957",
    pages = "16177--16188",
}
```
