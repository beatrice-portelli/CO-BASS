import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import train_test_split
from utils import DISEASE_COLS, SYMPTOM_COLS, OUTPUT_FORMAT

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

# mpl.use("module://mpl_ascii")


def get_distribution(df, rounded=True):
    d = df.sum().to_frame()
    d.columns = ["n"]
    d["perc"] = (d.n / d.n.sum()).round(2) if rounded else (d.n / d.n.sum())
    d=d.sort_values("perc", ascending=False)
    return d

df = pd.read_csv("data/data.csv")

# Define multi-label target columns (the 5 pathogens)
target_cols = DISEASE_COLS
feature_cols = SYMPTOM_COLS
print("-"*100)
print("DISEASES:", target_cols)
print("SYMPTOMS:", feature_cols)

SPLIT = dict()

X = df[feature_cols].values
y = df[target_cols].values
image_ids = df["photo_id"].values

print("-"*100)
print(f"Total dataset shape: {df.shape}\n")
print(f"Target multi-label distribution:\n{get_distribution(df[target_cols])}\n")
print(f"Target multi-label distribution:\n{get_distribution(df[feature_cols])}\n")

# Extract test set (20%) using iterstrat splitter just for the initial train/test split
test_splitter = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# taking first fold as our 20% holdout test set
for train_cv_idx, test_idx in test_splitter.split(X, y):
    X_train_cv, X_test = X[train_cv_idx], X[test_idx]
    y_train_cv, y_test = y[train_cv_idx], y[test_idx]
    break

train_data = df.iloc[train_cv_idx].reset_index(drop=True)
test_data = df.iloc[test_idx].reset_index(drop=True)

SPLIT["test_ids"] = test_data.photo_id.tolist()

train_data.to_csv("train_data.csv", index=None)
test_data.to_csv("test_data.csv", index=None)

print("-"*100)
print(f"Holdout Test Set: {train_data.shape[0]} samples (~20%)")
print(f"Remaining Train+Val Set: {test_data.shape[0]} samples (~80%)")
print()
print(f"Train multi-label distribution:\n{get_distribution(train_data[target_cols])}\n")
print(f"Train multi-label distribution:\n{get_distribution(train_data[feature_cols])}\n")
print(f"Test multi-label distribution:\n{get_distribution(test_data[target_cols])}\n")
print(f"Test multi-label distribution:\n{get_distribution(test_data[feature_cols])}\n")

# Set up the Stratified 5-Fold Cross-Validation on the Train+Validation set
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

FOLDS = []

print("-"*100)
print("\nInitializing 5-Fold Cross-Validation Split:")
for fold, (train_idx, val_idx) in enumerate(mskf.split(train_data, train_data[target_cols].values)):

    train = train_data.iloc[train_idx]
    val = train_data.iloc[val_idx]

    
    print(f"--- FOLD {fold + 1} ---")
    print(f"    Train size: {train.shape[0]} samples")
    print(f"    Val size:   {val.shape[0]} samples")
    print()
    print(f"    Train multi-label distribution:\n{get_distribution(train[target_cols])}\n")
    print(f"    Val multi-label distribution:\n{get_distribution(val[target_cols])}\n")
    
    train_imgs = train["photo_id"]
    val_imgs = val["photo_id"]

    item = {
        "train_ids": train_imgs.tolist(),
        "val_ids": val_imgs.tolist(),
        "test_ids": SPLIT["test_ids"],
    }
    FOLDS.append(item)

SPLIT["folds"] = FOLDS

# save sample indices for all folds
split_df = pd.DataFrame(SPLIT["folds"]).T
split_df.to_json("split_info.json")

print("-"*100)
print("train, val, and test file names for 5 folds saved to split_info.json")
print(split_df)




for COLS in [DISEASE_COLS, SYMPTOM_COLS]:

    fig, axs = plt.subplots(nrows=1, ncols=2, width_ratios=[1.4,1], figsize=(2.4*3,2.4*2))
    
    # COLS = DISEASE_COLS
    
    dist_test = get_distribution(
        df.set_index("photo_id").loc[split_df.loc["test_ids",0]][COLS], False
    ).perc.rename("Test")
    dist_train = []
    for idx in range(5):
        dist_train_idx = get_distribution(
            df.set_index("photo_id").loc[split_df.loc["train_ids",idx]][COLS], False
        ).perc.rename(f"Train({idx+1})")
        dist_train.append(dist_train_idx)
    dist_val = []
    for idx in range(5):
        dist_val_idx = get_distribution(
            df.set_index("photo_id").loc[split_df.loc["val_ids",idx]][COLS], False
        ).perc.rename(f"Val({idx+1})")
        dist_val.append(dist_val_idx)
    dist_all = dist_train+dist_val+[dist_test]
    cmap = mpl.colormaps['rainbow']
    
    dist_df = pd.concat(dist_all, axis=1).T*100
    dist_df.plot(
        kind="bar",
        stacked=True,
        width=1,
        cmap=cmap,
        legend=False,
        lw=0.5,
        edgecolor="k",
        ax=axs[0]
    )
    
    legend_elements = [
        Patch(
            facecolor=cmap(idx/(len(dist_df.columns)-1)),
            ec="k",
            lw=0.5,
            label=label.replace("missing_skin_", "").replace("_", " ").title()
        )
        for idx, label in enumerate(dist_df.columns)
    ]
    
    axs[1].legend(
        handles=legend_elements,
        bbox_to_anchor=(0, 1),
        loc='upper left',
        frameon=False,
        ncols=1,
        reverse=True
    )

    kind = "Symptom" if len(legend_elements) > 5 else "Disease"
    axs[0].set_title(f"{kind} distribution across splits\n")

    axs[1].set_axis_off()
    
    axs[0].grid(axis="y", ls="--", c="w")
    axs[0].set_ylim(0,100)
    axs[0].set_xlim(-0.5,len(dist_all)-0.5)
    axs[0].yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
    plt.tight_layout()
    plt.savefig(f"fig/01_data_distribution_{kind}.{OUTPUT_FORMAT}")
    plt.show()