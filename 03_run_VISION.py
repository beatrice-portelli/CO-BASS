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
from models import create_vision_model
from datasets import SeabassDataset

# Training Loop
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

# Test / Evaluation function
def evaluate_model(model, dataloader, criterion, TARGET_COLS, device):
    model.eval()
    all_targets = []
    all_preds = []
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.to(device))
            running_loss += loss.item() * images.size(0)
            
            # Apply sigmoid because BCEWithLogitsLoss uses raw logits
            preds = torch.sigmoid(outputs).cpu().numpy()
            
            all_targets.append(labels.numpy())
            all_preds.append(preds)
            
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    
    # Binary thresholding at 0.5 for F1 calculation
    binary_preds = (all_preds > 0.5).astype(int)

    loss = running_loss / len(dataloader.dataset)
    
    report = classification_report(all_targets, binary_preds, zero_division=0.0, target_names=TARGET_COLS, output_dict=True) 
    
    return loss, report, all_targets, all_preds



def evaluate_ensemble_on_test(test_df, out_dir, model_type, TARGET_COLS, IMG_DIR, BATCH_SIZE, DEVICE, val_transforms):
    # Initialize DataLoader for Test Set
    test_dataset = SeabassDataset(test_df, IMG_DIR, TARGET_COLS, transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Re-instantiate the 5 models and load weights
    ensemble_models = []
    for fold in range(5):
        checkpoint_path = f"{out_dir}/best_fold_{fold+1}.pth"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Missing weight file: {checkpoint_path}. Run training first.")
            
        model = create_vision_model(model_type, len(TARGET_COLS), True)
        
        # Load state dict
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        ensemble_models.append(model)
        
    print(f"Successfully loaded {len(ensemble_models)} fold checkpoints into memory.")

    all_targets = []
    # This matrix accumulates probabilities across models: shape (num_samples, num_diseases)
    ensemble_probs = np.zeros((len(test_df), len(TARGET_COLS)))
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(DEVICE)
            start_idx = batch_idx * BATCH_SIZE
            end_idx = start_idx + images.size(0)
            
            # Store ground truth labels
            all_targets.append(labels.numpy())
            
            # Run inference through each fold model independently
            batch_ensemble_probs = np.zeros((images.size(0), len(TARGET_COLS)))
            for model in ensemble_models:
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                batch_ensemble_probs += probs
                
            # Average out probabilities across the 5 models
            batch_ensemble_probs /= len(ensemble_models)
            ensemble_probs[start_idx:end_idx] = batch_ensemble_probs
            
    all_targets = np.vstack(all_targets)
    
    binary_preds = (ensemble_probs > 0.5).astype(int)
    
    print("\nPERFORMANCE MATRIX:")
    print("----------------------------------------------------------------")
    
    for i, disease in enumerate(TARGET_COLS):
        try:
            auc = roc_auc_score(all_targets[:, i], ensemble_probs[:, i])
            auc_str = f"{auc:.4f}"
        except ValueError:
            auc_str = "N/A" # Safe handling if a fold split accidentally completely omits a rare pathogen type
            
        print(f"Disease: {disease:<12} | AUC: {auc_str}")
        
    print("----------------------------------------------------------------")
    macro_f1 = f1_score(all_targets, binary_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_targets, binary_preds, average="micro", zero_division=0)
    print(f"Ensemble Global Macro-F1 Score: {macro_f1:.4f}")
    print(f"Ensemble Global Micro-F1 Score: {micro_f1:.4f}")
    
    report = classification_report(all_targets, binary_preds, zero_division=0.0, target_names=TARGET_COLS, output_dict=False)

    print(report)
    report = classification_report(all_targets, binary_preds, zero_division=0.0, target_names=TARGET_COLS, output_dict=True)
    val_f1 = report['weighted avg']['f1-score']
    samp_f1 = report['samples avg']['f1-score']

    print(
        f" Val Weighted-F1: {val_f1:.4f} |"\
        f" Val Sample-F1: {samp_f1:.4f} |"
    )
    
    return macro_f1, ensemble_probs, all_targets



def train_model(out_dir, args):

    seed_everything(seed=42)
    
    # Hyperparameters & Configuration
    IMG_DIR = "data/" 
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 2e-4 if not args.freeze else 2e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    MODEL = args.model
    TARGET = args.target

    if TARGET == "dis":
        TARGET_COLS = DISEASE_COLS
    else:
        TARGET_COLS = SYMPTOM_COLS

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
            
            train_dataset = SeabassDataset(train_df, IMG_DIR, TARGET_COLS, transform=train_transforms)
            val_dataset = SeabassDataset(val_df, IMG_DIR, TARGET_COLS, transform=val_transforms)
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker)
    
            model = create_vision_model(MODEL, len(TARGET_COLS), args.freeze)
            model = model.to(DEVICE)
            
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            
            best_val_f1 = 0.0
            best_val_samp_f1 = 0.0
            
            for epoch in range(EPOCHS):
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
                test_loss, report, test_real, test_preds = evaluate_model(model, val_loader, criterion, TARGET_COLS, DEVICE)
    
                val_f1 = report['weighted avg']['f1-score']
                samp_f1 = report['samples avg']['f1-score']
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_val_samp_f1 = samp_f1
                    
                    # Save the best model weights for this fold iteration
                    torch.save(model.state_dict(), f"{out_dir}/best_fold_{fold+1}.pth")
                    val_results = {
                        "real": test_real,
                        "pred": test_preds,
                    }
                    _, _, train_real, train_preds = evaluate_model(model, train_loader, criterion, TARGET_COLS, DEVICE)
                    train_results = {
                        "real": train_real,
                        "pred": train_preds,
                    }
                    pd.Series(train_results).to_pickle(f"{out_dir}/preds_train_{fold+1}.pkl")
                    pd.Series(val_results).to_pickle(f"{out_dir}/preds_val_{fold+1}.pkl")
                    
                    
                print(
                    f"Epoch {epoch+1:02d}/{EPOCHS:02d} |"\
                    f" Train Loss: {train_loss:.4f} |"\
                    f" Test Loss: {test_loss:.4f} |"\
                    f" Val Weighted-F1: {val_f1:.4f} |"\
                    f" Val Sample-F1: {samp_f1:.4f} |"
                )
                
            print(f"Fold {fold+1} Finished. Best Validation Weighted-F1: {best_val_f1:.4f}. Best Validation Sample-F1: {best_val_samp_f1:.4f}.")
            fold_f1_scores.append(best_val_f1)
    
        print("\nFinal Baseline Results Across 5 Folds:")
        print(f"Mean Validation Macro-F1 Score: {np.mean(fold_f1_scores):.4f} +/- {np.std(fold_f1_scores):.4f}")

    # ===============================================================================

    test_df = pd.read_csv("test_data.csv")

    # _, ensemble_probs_trainfold, all_targets_trainfold = evaluate_ensemble_on_test(train_df, out_dir, MODEL, len(TARGET_COLS), val_transforms)
    # _, ensemble_probs_valfold, all_targets_valfold = evaluate_ensemble_on_test(val_df, out_dir, MODEL, len(TARGET_COLS), val_transforms)
    _, ensemble_probs_testfold, all_targets_testfold = evaluate_ensemble_on_test(test_df, out_dir, MODEL, TARGET_COLS, IMG_DIR, BATCH_SIZE, DEVICE, val_transforms)

    test_results = {
        "real": all_targets_testfold,
        "pred": ensemble_probs_testfold,
    }
    pd.Series(test_results).to_pickle(f"{out_dir}/preds_test_ALL.pkl")
    
    
    for ensemble_probs, all_targets in [
        # (ensemble_probs_trainfold, all_targets_trainfold),
        # (ensemble_probs_valfold, all_targets_valfold),
        (ensemble_probs_testfold, all_targets_testfold),
    ]:

    
        y_true = all_targets
        y_pred = (ensemble_probs>[0.5]*len(TARGET_COLS)).astype(int)
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
        "--target", 
        type=str, 
        default="dis", 
        help="Must be one of: [dis, sym] (default=dis)"
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

    args = parser.parse_args()

    frozen = "[freeze]" if args.freeze else ""

    # create output directory
    ts = str(datetime.now().timestamp()).split(".")[0]
    out_dir = f"runs/{ts}_{args.target}_{frozen}{args.model}" if args.test_only is None else f"runs/{args.test_only}"
    print(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    train_model(out_dir, args)