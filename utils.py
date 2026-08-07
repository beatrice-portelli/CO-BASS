import os
import torch
import torch.nn as nn
import random
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd
from glob import glob

OUTPUT_FORMAT = "png"

DISEASE_COLS = ["_vibrio", "_photobact", "_tenacibac", "_aeromonas", "_lactococcus"]
SYMPTOM_COLS = [
    "emaciated",
    "damaged_fins",
    "missing_skin_skin_lesions",
    "ulcers",
    "exophthalmos",
    "hemorrhages",
    "swollen_abdomen",
    "abnormal_coloration"
]



def generate_co_occurrence_mask(y_diseases, y_symptoms, threshold=0.05, soft=False):
    """
    Computes an empirical mapping mask based on co-occurrence in the training split.
    y_diseases: numpy array
    y_symptoms: numpy array
    """
    num_dis = y_diseases.shape[1]
    num_sym = y_symptoms.shape[1]
    
    # Initialize an empirical mapping matrix: shape (num_symptoms, num_diseases)
    empirical_matrix = np.zeros((num_sym, num_dis))
    
    for d in range(num_dis):
        # Isolate all samples that are positive for this specific disease
        disease_indices = np.where(y_diseases[:, d] == 1)[0]
        
        if len(disease_indices) > 0:
            # Calculate how frequently each symptom appears alongside this disease
            symptom_frequencies = y_symptoms[disease_indices].mean(axis=0)
            empirical_matrix[:, d] = symptom_frequencies
            
    # Convert to a binary mask based on a minimum co-occurrence threshold
    # If a symptom appears with a disease less than threshold (%) of the time, treat it as spurious/forbidden (0)
    binary_mask = (empirical_matrix >= threshold).astype(np.float32)
    soft_mask = np.where((empirical_matrix >= threshold), empirical_matrix.astype(np.float32), 0)

    return soft_mask if soft else binary_mask



def precompute_co_occurrence():
    df = pd.read_csv("train_data.csv")
    df["num_diseases"] = df[DISEASE_COLS].sum(1)
    print("OVERALL DISTRIBUTION")
    print(df[(df.num_diseases==1)|(df._aeromonas==1)][DISEASE_COLS].sum())
    
    df.index = df["photo_id"]
    
    FOLDS = pd.read_json("split_info.json")

    matrices = []
    
    for fold in range(len(FOLDS.columns)):    
        train_idx = FOLDS[fold]["train_ids"]
        val_idx = FOLDS[fold]["val_ids"]
        
        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]
    
        small_df = train_df[(train_df.num_diseases==1)|(train_df._aeromonas==1)]
        mask = generate_co_occurrence_mask(small_df[DISEASE_COLS].values, small_df[SYMPTOM_COLS].values, threshold=0.2, soft=True)
        mask_df = pd.DataFrame(mask, index=SYMPTOM_COLS, columns=DISEASE_COLS)
    
        mask_df.to_csv(f"fold_{fold+1}_co-occurrence.csv")

        matrices.append(mask)
    return matrices


def collect_metrics(TARGET):

    prev_metrics = f"performance_{TARGET}.csv"
    if os.path.exists(prev_metrics):
        prev_df = pd.read_csv(prev_metrics)
    else:
        prev_df = pd.DataFrame(index=[], columns=[])

    prev_paths = prev_df.path+"/preds_test_ALL.pkl" if len(prev_df)>0 else []

    data = []
    
    if TARGET == "dis":
        TARGET_COLS = DISEASE_COLS
    else:
        TARGET_COLS = SYMPTOM_COLS
    
    paths = glob(f"runs/*_{TARGET}_*/preds_test_ALL.pkl")

    paths = set(paths)
    new_paths = paths - set(prev_paths)
    print("Processing", len(new_paths), "new files")
    

    for path in new_paths:

        base_path = path.replace("/preds_test_ALL.pkl","")

        MODEL = path.split("/")[1].split("_")[-1]
        
        df = pd.read_pickle(path)
        
        y_true = df.real
        y_prob = df.pred
        
        y_pred_05 = (df.pred>0.5).astype(int)
        y_pred_03 = (df.pred>0.3).astype(int)


        wf1 = f1_score(y_true, y_pred_05, zero_division=0, average="weighted")
        sf1 = f1_score(y_true, y_pred_05, zero_division=0, average="samples")

        wf1_03 = f1_score(y_true, y_pred_03, zero_division=0, average="weighted")
        sf1_03 = f1_score(y_true, y_pred_03, zero_division=0, average="samples")
        

        class_auc = dict()
        for i, disease in enumerate(TARGET_COLS):
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
            class_auc[disease] = auc
        auc = roc_auc_score(y_true, y_prob)
        
        item = {
            "path": base_path,
            "target": TARGET,
            "model": MODEL,
            "wF1": wf1,
            "sF1": sf1,
            "wF1_03": wf1_03,
            "sF1_03": sf1_03,
            "AUC": auc
        }
        for k,v in class_auc.items():
            item["AUC_"+k] = v
        data.append(item)
    data = pd.DataFrame(data)

    full_data = pd.concat((prev_df, data))
    full_data.to_csv(prev_metrics, index=False)
    
    return full_data


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # safe for multi-GPU setups
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"All random seeds locked to: {seed}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def freeze_params(model):
    for p in model.parameters():
        p.requires_grad = False

