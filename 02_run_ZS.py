import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import f1_score, classification_report
from datetime import datetime
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

import open_clip

from utils import DISEASE_COLS, SYMPTOM_COLS, OUTPUT_FORMAT
from utils import seed_everything, seed_worker
from datasets import SeabassDataset, OpenClipSeabassDataset


# Custom collate function for clip and openclip
def clip_collate_fn(batch):
    images = [item[0] for batch_item in batch for item in [batch_item]]
    labels = torch.stack([item[1] for item in batch])
    return images, labels

def openclip_collate_fn(batch):
    images = torch.vstack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return images, labels

DISEASE_PROMPTS = {
    "_vibrio": [
        "a healthy fish",
        "a sick fish with vibrio infection, skin hemorrhages and deep muscle ulcers"
    ],
    "_photobact": [
        "a healthy fish",
        "a sick fish with photobacterium infection, internal swelling and white pseudotubercles"
    ],
    "_tenacibac": [
        "a healthy fish",
        "a sick fish with tenacibaculum infection, rotten frayed fins and mouth mouth-rot sores"
    ],
    "_aeromonas": [
        "a healthy fish",
        "a sick fish with aeromonas bacterial infection, extensive bleeding skin ulcers and fin rot"
    ],
    "_lactococcus": [
        "a healthy fish",
        "a healthy fish & a sick fish with lactococcus infection, exophthalmos bulging pop-eyes and a distended abdomen"
    ]
}

SYMPTOM_PROMPTS = {
    "damaged_fins": [
        "a fish with healthy fins",
        "a fish with damaged, frayed, decaying, or splitting fins caused by rot"
    ],
    "ulcers": [
        "a fish with smooth undamaged skin",
        "a fish with skin ulcers, open and bloody"
    ],
    "exophthalmos": [
        "a fish with normal flat eyes",
        "a fish with severe exophthalmos bulging swollen eyes protruding"
    ],
    "hemorrhages": [
        "a fish with clean scales and skin",
        "a fish with red bloody hemorrhages, subcutaneous bleeding"
    ],
    "swollen_abdomen": [
        "a fish with a normal slender belly",
        "a fish with heavily swollen, fluid-filled distended bloated abdomen"
    ],
    "emaciated": [
        "a fish with healthy normal body weight and muscle", 
        "a fish with emaciated gaunt body and thin muscle mass"
    ],
    "missing_skin_skin_lesions": [
        "a fish with intact, healthy, and smooth skin surface", 
        "a fish with missing skin patches, open sores, and eroded skin lesions"
    ],
    "abnormal_coloration": [
        "a fish with natural and healthy skin color", 
        "a fish with abnormal skin discoloration, darkening, patchy skin"
    ],
}

# Zero-Shot Inference
def evaluate_clip_zero_shot(model, processor, dataloader, target_cols, prompt_dict, device):
    model.eval()
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            all_targets.append(labels.numpy())
            batch_probs = []
            
            # Run independent text-image alignment scoring loops for each binary column
            for col in target_cols:
                prompts = prompt_dict[col] # [Healthy phrase, Diseased phrase]
                
                # Preprocess image and text inputs via HF CLIP processor
                inputs = processor(
                    text=prompts, 
                    images=images, 
                    return_tensors="pt", 
                    padding=True
                ).to(device)
                
                outputs = model(**inputs)
                
                # Extract image-to-text assignment probabilities via softmax over the prompt options
                # index [:, 1] captures the absolute risk score of matching the "Diseased" caption description
                probs = outputs.logits_per_image.softmax(dim=1)[:, 1].cpu().numpy()

                batch_probs.append(probs)
                
            # Stack outputs: shape (batch_size, num_classes)
            all_probs.append(np.column_stack(batch_probs))
            
    return np.vstack(all_targets), np.vstack(all_probs)


def evaluate_openclip_zero_shot(model, preprocess, tokenizer, dataloader, target_cols, prompt_dict, device):
    model.eval()
    all_targets = []
    all_probs = []
    
    with torch.no_grad(), torch.autocast("cuda"):
        for images, labels in tqdm(dataloader):

            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            all_targets.append(labels.numpy())
            batch_probs = []
            
            # Run independent text-image alignment scoring loops for each binary column
            for col in target_cols:
                prompts = prompt_dict[col] # [Healthy phrase, Diseased phrase]
                
                texts = tokenizer(prompts)
                
                text_features = model.encode_text(texts)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                probs = (100.0 * image_features @ text_features.T).softmax(dim=1)[:, 1].cpu().numpy()
                
                batch_probs.append(probs)
                
            # Stack outputs: shape (batch_size, num_classes)
            all_probs.append(np.column_stack(batch_probs))
            
    return np.vstack(all_targets), np.vstack(all_probs)



def run_model(out_dir, args):

    seed_everything(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_DIR = "data/"
    BATCH_SIZE=16

    model_name_dict = {
        "vit.base.32": "openai/clip-vit-base-patch32",
        "vit.h.laion2b": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "bioclip": "hf-hub:imageomics/bioclip",
        "bioclip2": "hf-hub:imageomics/bioclip-2",
    }

    TARGET = args.target
    MODEL = model_name_dict[args.model]

    OPENCLIP = "bioclip" in MODEL

    if TARGET == "dis":
        TARGET_COLS = DISEASE_COLS
        PROMPT_MATRIX = DISEASE_PROMPTS
    else:
        TARGET_COLS = SYMPTOM_COLS
        PROMPT_MATRIX = SYMPTOM_PROMPTS

    pd.DataFrame(PROMPT_MATRIX).to_csv(f"{out_dir}/prompts.csv", index=None)

    df = pd.read_csv("train_data.csv")
    df.index = df["photo_id"]
    
    FOLDS = pd.read_json("split_info.json")

    fold_f1_scores = []

    if OPENCLIP:
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
        tokenizer = open_clip.get_tokenizer(MODEL)
    else:
        model = CLIPModel.from_pretrained(MODEL).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(MODEL)

    print(f"Starting Multi-Label {MODEL} Baseline for {TARGET} on {DEVICE}...")
        
    for fold in range(len(FOLDS.columns)):
        print(f"\n--- Fold {fold + 1}/5 ---")

        train_idx = FOLDS[fold]["train_ids"]
        val_idx = FOLDS[fold]["val_ids"]
        
        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]

        

        if OPENCLIP:

            train_dataset = OpenClipSeabassDataset(train_df, IMG_DIR, TARGET_COLS, preprocess)
            val_dataset = OpenClipSeabassDataset(val_df, IMG_DIR, TARGET_COLS, preprocess)
    
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker, collate_fn=openclip_collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker, collate_fn=openclip_collate_fn)

            y_train, train_probs = evaluate_openclip_zero_shot(model, preprocess, tokenizer, train_loader, TARGET_COLS, PROMPT_MATRIX, DEVICE)
            y_val, val_probs = evaluate_openclip_zero_shot(model, preprocess, tokenizer, val_loader, TARGET_COLS, PROMPT_MATRIX, DEVICE)
            
        else:

            train_dataset = SeabassDataset(train_df, IMG_DIR, TARGET_COLS)
            val_dataset = SeabassDataset(val_df, IMG_DIR, TARGET_COLS)
    
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker, collate_fn=clip_collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker, collate_fn=clip_collate_fn)

            y_train, train_probs = evaluate_clip_zero_shot(model, processor, train_loader, TARGET_COLS, PROMPT_MATRIX, DEVICE)
            y_val, val_probs = evaluate_clip_zero_shot(model, processor, val_loader, TARGET_COLS, PROMPT_MATRIX, DEVICE)
        
        train_preds = (train_probs > 0.5).astype(int)
        val_preds = (val_probs > 0.5).astype(int)

        train_results = {
            "real": y_train,
            "pred": train_probs,
        }
        val_results = {
            "real": y_val,
            "pred": val_probs,
        }

        pd.Series(train_results).to_pickle(f"{out_dir}/preds_train_{fold+1}.pkl")
        pd.Series(val_results).to_pickle(f"{out_dir}/preds_val_{fold+1}.pkl")

        best_val_f1 = f1_score(y_val, val_preds, average="weighted", zero_division=0)
        best_val_samp_f1 = f1_score(y_val, val_preds, average="samples", zero_division=0)
        
        print(f"Fold {fold+1} Finished. Validation Weighted-F1: {best_val_f1:.4f}. Validation Sample-F1: {best_val_samp_f1:.4f}.")

        fold_f1_scores.append(best_val_f1)

    print("\nFinal Baseline Results Across 5 Folds:")
    print(f"Mean Validation Macro-F1 Score: {np.mean(fold_f1_scores):.4f} +/- {np.std(fold_f1_scores):.4f}")

    # ===============================================================================

    test_df = pd.read_csv("test_data.csv")

    if OPENCLIP:

        test_dataset = OpenClipSeabassDataset(test_df, IMG_DIR, TARGET_COLS, preprocess)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker, collate_fn=openclip_collate_fn)
    
        y_test, test_probs = evaluate_openclip_zero_shot(model, preprocess, tokenizer, test_loader, TARGET_COLS, PROMPT_MATRIX, DEVICE)
        
    else:
        
        test_dataset = SeabassDataset(test_df, IMG_DIR, TARGET_COLS)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker, collate_fn=clip_collate_fn)
    
        y_test, test_probs = evaluate_clip_zero_shot(model, processor, test_loader, TARGET_COLS, PROMPT_MATRIX, DEVICE)
    test_preds = (test_probs > 0.5).astype(int)

    test_results = {
        "real": y_test,
        "pred": test_probs,
    }
    pd.Series(test_results).to_pickle(f"{out_dir}/preds_test_ALL.pkl")

    y_pred = (test_probs>[0.5]*len(TARGET_COLS)).astype(int)
    
    print(MODEL, TARGET)
    print(classification_report(y_test, y_pred, zero_division=0))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        default="dis",
        help="Must be one of: [dis, sym] (default=dis)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit.base.32",
        help="Must be one of: [vit.base.32, vit.h.laion2b, bioclip, bioclip2] (default=vit.base.32)",
    )
    args = parser.parse_args()


    # create output directory
    ts = str(datetime.now().timestamp()).split(".")[0]
    out_dir = f"runs/{ts}_CLIP_{args.target}_{args.model}"
    print(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    run_model(out_dir, args)
