import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import argparse
from torchvision import transforms
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from datetime import datetime

from utils import DISEASE_COLS, SYMPTOM_COLS, OUTPUT_FORMAT
from utils import seed_everything, seed_worker
from models import MultiTaskVisionModel
from datasets import MTLSeabassDataset


DIS_COLS = DISEASE_COLS
SYM_COLS = SYMPTOM_COLS


# Training Loop
def train_one_epoch(model, dataloader, criterion, optimizer, device, alpha=0.5):
    model.train()
    running_loss = 0.0
    for images, disease_targets, symptom_targets in dataloader:
        images = images.to(device)
        disease_targets = disease_targets.to(device)
        symptom_targets = symptom_targets.to(device)
        
        optimizer.zero_grad()
        
        disease_outputs, symptom_outputs = model(images)

        loss_disease = criterion(disease_outputs, disease_targets)
        loss_symptom = criterion(symptom_outputs, symptom_targets)
        
        joint_loss = (alpha * loss_disease) + ((1.0 - alpha) * loss_symptom)
        joint_loss.backward()
        optimizer.step()
        
        running_loss += joint_loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

# Test / Evaluation function
def evaluate_model(model, dataloader, criterion, device, alpha=0.5):
    model.eval()
    all_dis_targets, all_dis_probs = [], []
    all_sym_targets, all_sym_probs = [], []
    running_loss = 0.0
    
    with torch.no_grad():
        for images, disease_targets, symptom_targets in dataloader:
            images = images.to(device)
            disease_targets = disease_targets.to(device)
            symptom_targets = symptom_targets.to(device)
            
            disease_outputs, symptom_outputs = model(images)
            loss_disease = criterion(disease_outputs, disease_targets)
            loss_symptom = criterion(symptom_outputs, symptom_targets)
            
            joint_loss = (alpha * loss_disease) + ((1.0 - alpha) * loss_symptom)
            
            running_loss += joint_loss.item() * images.size(0)
                        
            all_dis_targets.append(disease_targets.cpu().numpy())
            all_sym_targets.append(symptom_targets.cpu().numpy())
            
            # Apply sigmoid because BCEWithLogitsLoss uses raw logits
            all_dis_probs.append(torch.sigmoid(disease_outputs).cpu().numpy())
            all_sym_probs.append(torch.sigmoid(symptom_outputs).cpu().numpy())
            
    y_true_dis = np.vstack(all_dis_targets)
    y_prob_dis = np.vstack(all_dis_probs)
    y_true_sym = np.vstack(all_sym_targets)
    y_prob_sym = np.vstack(all_sym_probs)
    
    # Binary thresholding at 0.5 for F1 calculation
    binary_preds_dis = (y_prob_dis > 0.5).astype(int)
    binary_preds_sym = (y_prob_sym > 0.5).astype(int)

    loss = running_loss / len(dataloader.dataset)
    
    dis_report = classification_report(y_true_dis, binary_preds_dis, zero_division=0.0, target_names=DIS_COLS, output_dict=True) 
    sym_report = classification_report(y_true_sym, binary_preds_sym, zero_division=0.0, target_names=SYM_COLS, output_dict=True) 
    
    return loss, dis_report, y_true_dis, y_prob_dis, sym_report, y_true_sym, y_prob_sym




def evaluate_ensemble_on_test(test_df, out_dir_sym, out_dir_dis, model_type, IMG_DIR, BATCH_SIZE, DEVICE, val_transforms):
    # Initialize DataLoader for Test Set
    test_dataset = MTLSeabassDataset(test_df, IMG_DIR, transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Re-instantiate the 5 models and load weights
    ensemble_models = []
    for fold in range(5):
        checkpoint_path = f"{out_dir_dis}/best_fold_{fold+1}.pth"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Missing weight file: {checkpoint_path}. Run training first.")
            
        model = MultiTaskVisionModel(model_type, len(DIS_COLS), len(SYM_COLS), args.freeze)
        
        # Load state dict
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        ensemble_models.append(model)
        
    print(f"Successfully loaded {len(ensemble_models)} fold checkpoints into memory.")

    # Step-by-step Inference Loop
    sym_all_targets = []
    dis_all_targets = []
    # This matrix accumulates probabilities across models: shape (num_samples, num_diseases)
    sym_ensemble_probs = np.zeros((len(test_df), len(SYM_COLS)))
    dis_ensemble_probs = np.zeros((len(test_df), len(DIS_COLS)))
    
    with torch.no_grad():
        for batch_idx, (images, dis_labels, sym_labels) in enumerate(test_loader):
            
            images = images.to(DEVICE)
            start_idx = batch_idx * BATCH_SIZE
            end_idx = start_idx + images.size(0)
            
            # Store ground truth labels
            sym_all_targets.append(sym_labels.numpy())
            dis_all_targets.append(dis_labels.numpy())
            
            # Run inference through each fold model independently
            sym_batch_ensemble_probs = np.zeros((images.size(0), len(SYM_COLS)))
            dis_batch_ensemble_probs = np.zeros((images.size(0), len(DIS_COLS)))
            
            for model in ensemble_models:
                dis_outputs, sym_outputs = model(images)
                dis_probs = torch.sigmoid(dis_outputs).cpu().numpy()
                sym_probs = torch.sigmoid(sym_outputs).cpu().numpy()
                sym_batch_ensemble_probs += sym_probs
                dis_batch_ensemble_probs += dis_probs
                
            # Average out probabilities across the 5 models
            sym_batch_ensemble_probs /= len(ensemble_models)
            dis_batch_ensemble_probs /= len(ensemble_models)
            sym_ensemble_probs[start_idx:end_idx] = sym_batch_ensemble_probs
            dis_ensemble_probs[start_idx:end_idx] = dis_batch_ensemble_probs
            
    sym_all_targets = np.vstack(sym_all_targets)
    dis_all_targets = np.vstack(dis_all_targets)
    
    # Binary Thresholding
    sym_binary_preds = (sym_ensemble_probs > 0.5).astype(int)
    dis_binary_preds = (dis_ensemble_probs > 0.5).astype(int)
    
    print("\nPERFORMANCE MATRIX:")
    print("----------------------------------------------------------------")
    
    # Per-class breakdowns
    for i, disease in enumerate(DIS_COLS):
        try:
            auc = roc_auc_score(dis_all_targets[:, i], dis_ensemble_probs[:, i])
            auc_str = f"{auc:.4f}"
        except ValueError:
            auc_str = "N/A"
        print(f"Disease: {disease:<12} | AUC: {auc_str}")

    for i, disease in enumerate(SYM_COLS):
        try:
            auc = roc_auc_score(sym_all_targets[:, i], sym_ensemble_probs[:, i])
            auc_str = f"{auc:.4f}"
        except ValueError:
            auc_str = "N/A"
        print(f"Disease: {disease:<12} | AUC: {auc_str}")
        
    print("----------------------------------------------------------------")
    macro_f1 = f1_score(sym_all_targets, sym_binary_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(sym_all_targets, sym_binary_preds, average="micro", zero_division=0)
    print(f"SYM Ensemble Global Macro-F1 Score: {macro_f1:.4f}")
    print(f"SYM Ensemble Global Micro-F1 Score: {micro_f1:.4f}")

    macro_f1 = f1_score(dis_all_targets, dis_binary_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(dis_all_targets, dis_binary_preds, average="micro", zero_division=0)
    print(f"DIS Ensemble Global Macro-F1 Score: {macro_f1:.4f}")
    print(f"DIS Ensemble Global Micro-F1 Score: {micro_f1:.4f}")
    
    report = classification_report(sym_all_targets, sym_binary_preds, zero_division=0.0, target_names=SYM_COLS, output_dict=False)
    print(report)
    report = classification_report(sym_all_targets, sym_binary_preds, zero_division=0.0, target_names=SYM_COLS, output_dict=True)
    val_f1 = report['weighted avg']['f1-score']
    samp_f1 = report['samples avg']['f1-score']
    print(
        f"SYM Val Weighted-F1: {val_f1:.4f} |"\
        f" Val Sample-F1: {samp_f1:.4f} |"
    )

    report = classification_report(dis_all_targets, dis_binary_preds, zero_division=0.0, target_names=DIS_COLS, output_dict=False)
    print(report)
    report = classification_report(dis_all_targets, dis_binary_preds, zero_division=0.0, target_names=DIS_COLS, output_dict=True)
    val_f1 = report['weighted avg']['f1-score']
    samp_f1 = report['samples avg']['f1-score']
    print(
        f"DIS Val Weighted-F1: {val_f1:.4f} |"\
        f" Val Sample-F1: {samp_f1:.4f} |"
    )
    
    return macro_f1, sym_ensemble_probs, sym_all_targets, dis_ensemble_probs, dis_all_targets



def train_model(out_dir_sym, out_dir_dis, args):

    seed_everything(seed=42)
    
    # Hyperparameters & Configuration
    IMG_DIR = "data/" 
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 2e-4 if not args.freeze else 2e-4
    ALPHA = args.alpha
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    MODEL = args.model
    TARGET = "both"
    

    # Image Augmentations
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args.test_only is None:

        # ===============================================================================
    
        df = pd.read_csv("train_data.csv")
        df.index = df["photo_id"]
    
        FOLDS = pd.read_json("split_info.json")
    
        fold_f1_scores = []
        
        print(f"Starting Multi-Label {MODEL} Baseline for {TARGET} on {DEVICE}...")
        
        for fold in range(len(FOLDS.columns)):
            print(f"\n--- Training Fold {fold + 1}/5 ---")
    
            train_idx = FOLDS[fold]["train_ids"]
            val_idx = FOLDS[fold]["val_ids"]
            
            train_df = df.loc[train_idx]
            val_df = df.loc[val_idx]
            
            train_dataset = MTLSeabassDataset(train_df, IMG_DIR, transform=train_transforms)
            val_dataset = MTLSeabassDataset(val_df, IMG_DIR, transform=val_transforms)
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker)
    
            model = MultiTaskVisionModel(MODEL, len(DIS_COLS), len(SYM_COLS), args.freeze)
            model = model.to(DEVICE)
            
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            
            best_val_f1 = 0.0
            best_val_samp_f1 = 0.0
            best_epoch = 0
            
            for epoch in range(EPOCHS):
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, ALPHA)
                test_loss, dis_report, dis_test_real, dis_test_preds, sym_report, sym_test_real, sym_test_preds = evaluate_model(model, val_loader, criterion, DEVICE, ALPHA)
    
                sym_val_f1 = sym_report['weighted avg']['f1-score']
                sym_samp_f1 = sym_report['samples avg']['f1-score']

                dis_val_f1 = dis_report['weighted avg']['f1-score']
                dis_samp_f1 = dis_report['samples avg']['f1-score']

                val_f1 = sym_val_f1 + dis_val_f1
                samp_f1 = sym_samp_f1 + dis_samp_f1
                
                if val_f1 > best_val_f1:
                    best_epoch = epoch
                    best_val_f1 = val_f1
                    best_val_samp_f1 = samp_f1
                    
                    # Save the best model weights for this fold iteration
                    torch.save(model.state_dict(), f"{out_dir_dis}/best_fold_{fold+1}.pth")

                    _, _, dis_train_real, dis_train_preds, _, sym_train_real, sym_train_preds = evaluate_model(model, train_loader, criterion, DEVICE, ALPHA)
                    
                    
                    sym_val_results = {
                        "real": sym_test_real,
                        "pred": sym_test_preds,
                    }
                    sym_train_results = {
                        "real": sym_train_real,
                        "pred": sym_train_preds,
                    }
                    pd.Series(sym_train_results).to_pickle(f"{out_dir_sym}/preds_train_{fold+1}.pkl")
                    pd.Series(sym_val_results).to_pickle(f"{out_dir_sym}/preds_val_{fold+1}.pkl")

                    dis_val_results = {
                        "real": dis_test_real,
                        "pred": dis_test_preds,
                    }
                    dis_train_results = {
                        "real": dis_train_real,
                        "pred": dis_train_preds,
                    }
                    pd.Series(dis_train_results).to_pickle(f"{out_dir_dis}/preds_train_{fold+1}.pkl")
                    pd.Series(dis_val_results).to_pickle(f"{out_dir_dis}/preds_val_{fold+1}.pkl")
                    
                    
                print(
                    f"Epoch {epoch+1:02d}/{EPOCHS:02d} |"\
                    f" Train Loss: {train_loss:.4f} |"\
                    f" Test Loss: {test_loss:.4f} |"\
                    f" SYM w_F1: {sym_val_f1:.4f} |"\
                    f" SYM s_F1: {sym_samp_f1:.4f} |"\
                    f" DIS w_F1: {dis_val_f1:.4f} |"\
                    f" DIS s_F1: {dis_samp_f1:.4f} |"
                )
                
            print(f"Fold {fold+1} Finished. Best epoch {best_epoch+1}. Best Validation Weighted-F1: {best_val_f1:.4f}. Best Validation Sample-F1: {best_val_samp_f1:.4f}.")
            fold_f1_scores.append(best_val_f1)
    
        print("\nFinal Baseline Results Across 5 Folds:")
        print(f"Mean Validation Macro-F1 Score: {np.mean(fold_f1_scores):.4f} +/- {np.std(fold_f1_scores):.4f}")

    # ===============================================================================

    test_df = pd.read_csv("test_data.csv")

    _, sym_ensemble_probs, sym_all_targets, dis_ensemble_probs, dis_all_targets = evaluate_ensemble_on_test(test_df, out_dir_sym, out_dir_dis, MODEL, IMG_DIR, BATCH_SIZE, DEVICE, val_transforms)

    test_results = {
        "real": sym_all_targets,
        "pred": sym_ensemble_probs,
    }
    pd.Series(test_results).to_pickle(f"{out_dir_sym}/preds_test_ALL.pkl")

    test_results = {
        "real": dis_all_targets,
        "pred": dis_ensemble_probs,
    }
    pd.Series(test_results).to_pickle(f"{out_dir_dis}/preds_test_ALL.pkl")
    
    
    for ensemble_probs, all_targets in [
        # (ensemble_probs_trainfold, all_targets_trainfold),
        # (ensemble_probs_valfold, all_targets_valfold),
        (sym_ensemble_probs, sym_all_targets),
        (dis_ensemble_probs, dis_all_targets),
    ]:
    
        y_true = all_targets
        y_pred = (ensemble_probs>0.5).astype(int)
        print(MODEL, TARGET)
        print(classification_report(y_true, y_pred, zero_division=0))






if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="resnet", 
        help="""Must be one of: [ resnet, mobilenet, efficientnet, convnext, swint] (default=resnet)"""
    )
    parser.add_argument(
        "--freeze", 
        action="store_true",
        help="Freeze all layers except the new linear one"
    )
    parser.add_argument(
        "--test_only",
        type=str,
        default=None,
        help="Path to folder to recover and perform test only"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Parameter to balance the MTL loss. 0 = symptom only loss, 1 = disease only loss. Default 0.5"
    )

    args = parser.parse_args()

    frozen = "[freeze]" if args.freeze else ""
    alpha = f"[alpha={args.alpha}]"

    # create output directory
    ts = str(datetime.now().timestamp()).split(".")[0]
    out_dir_sym = f"runs/{ts}_sym_[MTL]{alpha}{frozen}{args.model}" if args.test_only is None else f"runs/{args.test_only}".replace("_dis_", "_sym_")
    out_dir_dis = f"runs/{ts}_dis_[MTL]{alpha}{frozen}{args.model}" if args.test_only is None else f"runs/{args.test_only}"
    print(out_dir_sym)
    print(out_dir_dis)
    os.makedirs(out_dir_sym, exist_ok=True)
    os.makedirs(out_dir_dis, exist_ok=True)

    train_model(out_dir_sym, out_dir_dis, args)