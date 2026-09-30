# CO-BASS

Official repository containing the code, data, and benchmark implementations for the paper [**"From Symptoms to Pathogens: Biologically Constrained Multi-Task Vision Models for Automated Fish Disease Diagnosis"**](https://openreview.net/forum?id=iMFCtuplMt) (ECCV 2026 - CV4Ecology).

---

## 📊 DATASET

**CO-BASS dataset: Co-occurring Bacterial Symptoms in Seabass**

320 standardized images of symptomatic European seabass annotated by veterinary experts for 8 physical symptoms and 5 pathogens.

⚠️ To request access to the data for research purposes, please write an email to the corresponding author mentioning the dataset name: portelli.beatrice[at]spes.uniud.it

**Symptoms**
1. emaciation
2. damaged fins
3. skin lesions (missing skin)
4. ulcers
5. exophthalmos
6. hemorrhages
7. swollen abdomen
8. abnormal coloration

**Diseases** (presence of bacterial pathogens belonging to genus)
1. *Vibrio*
2. *Photobacterium*
3. *Tenacibaculum*
4. *Aeromonas*
5. *Lactococcus*


Distribution of diseases in the dataset, calculated as percentage over all dataset samples (\%$_D$).

|                    | N   | %$_D$ |
|--------------------|----:|------:|
| Vibrio         (V) | 93  | 29%   |
| Tenacibaculum  (T) | 5   | 2%    |
| Photobacterium (P) | 20  | 6%    |
| Aeromonas      (A) | 0   | 0%    |
| Lactococcus    (L) | 3   | 1%    |
| V + T              | 61  | 19%   |
| V + P              | 86  | 27%   |
| V + A              | 3   | 1%    |
| T + P              | 1   | 0.3%  |
| P + A              | 7   | 2%    |
| V + T + P          | 40  | 12%   |
| V + T + A          | 1   | 0.3%  |

Distribution of symptoms in the dataset, calculated as percentage over all dataset samples (%$_D$) and over the total number of symptom instances (%$_S$).

|                     | N   | %$_D$ | %$_S$ |
|---------------------|----:|------:|------:|
| Hemorrhages         | 296 | 92% | 37% |
| Skin lesions        | 156 | 49% | 19% |
| Damaged fins        | 102 | 32% | 13% |
| Abnormal coloration | 62  | 19% | 8%  |
| Emaciated           | 61  | 19% | 8%  |
| Ulcers              | 52  | 16% | 6%  |
| Swollen Abdomen     | 49  | 15% | 6%  |
| Exophthalmos        | 30  | 9%  | 4%  |


---

## 🛠️ CODE - How to run

Code written on Python 3.10.18

1. Clone the repository: `git clone https://github.com/beatrice-portelli/CO-BASS.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Request access to the CO-BASS dataset (`data` folder)
4. Run `01_split_train_val_test.py` to split the dataset
5. To train all the models compared in the paper (with the best parameters), run `source experiment_01_best_models.sh`
6. To train the models needed to reproduce Figure 4 (effect of alpha in [0.1, 0.9]), run `source experiment_02_alpha.sh`
7. To train the models needed to reproduce Figure 5 (effect of beta in [0.1, 0.9]), run `source experiment_03_beta.sh`
8. To obtain a summary of the performance of all tested models, run `06_calculate_performances.py`
9. To obtain more in-depth results and reproduce Figures 3-5 of the paper, run `RESULTS.ipynb`

---

## 📂 Repository Structure

* **`data/`**: Contains images for the CO-BASS dataset. The labels for the images can be found in **`data/data.csv`**.
* * **`./datasets.py`**: Dataset classes to load single-task and multi-task dataset formats.
* * **`./models.py`**: Implementation of the models used in the paper.
* **`./utils.py`**: Helper functions for training and co-occurrence matrix generation. Contains the following useful constants used by other scripts:
  * OUTPUT_FORMAT: output format for the matplotlib images created by all scripts (eg, "png", "jpg", "pdf")
  * DISEASE_COLS: names of the disease columns used by all scripts
  * SYMPTOM_COLS: names of the symptom columns used by all scripts

## Details on the scripts

**`01_split_train_val_test.py`**
  
| inputs   | outputs | description |
| -------- | ------- | ----------- |
| data.csv | test_data.csv                        | Hold-out test dataset, 20% of data.csv |
|          | train_data.csv                       | Train and validation data, 80% of data.csv |
|          | split_info.json                      | Dictionary to map each train/val fold to a subset of samples. For each of the folds (0 to 4), contains the fields "train_ids", "val_ids", and "test_ids". Each field contains the list of image names. The test samples (hold-out data) are the same for all folds and are only reported for ease of parsing. |
|          | fig/01_data_distribution_Disease.png | Plot of the disease distribution across all folds (to verify if the folds are well stratified). |
|          | fig/01_data_distribution_Symptom.png | Plot of the symptom distribution across all folds (to verify if the folds are well stratified). |

**`02_run_ZS.py`**
```bash
python 02_run_ZS.py [--target TARGET] [--model MODEL]
  --target in [dis, sym]
  --model in [vit.base.32, vit.h.laion2b, bioclip, bioclip2]
```
| inputs   | outputs | description |
| -------- | ------- | ----------- |
| train_data.csv  | runs/{RUN_ID} | Base folder for all results of the run. {RUN_ID} is automatically created as {timestamp}\_CLIP\_{target}\_{model} |
| test_data.csv   | runs/{RUN_ID}/prompts.csv | File logging the positive and negative prompts used by the zero-shot model (logged in case they are changed between runs). |
| split_info.json | runs/{RUN_ID}/preds_train_{1,2,3,4,5}.pkl | For each training sample in the fold, real labels and predicted probability of the "symptomatic" prompt (compared to an "healthy" prompt).  |
|                 | runs/{RUN_ID}/preds_val_{1,2,3,4,5}.pkl | For each validation sample in the fold, real labels and predicted probability of the "symptomatic" prompt (compared to an "healthy" prompt).  |
|                 | runs/{RUN_ID}/preds_test_ALL.pkl | For each test sample, real labels and predicted probability of the "symptomatic" prompt (compared to an "healthy" prompt).  |

**`03_run_VISION.py`**
```bash
python 03_run_VISION.py [--target TARGET] [--model MODEL]
  --target in [dis, sym]
  --model in [resnet, mobilenet, efficientnet, convnext, swint]
```
| inputs   | outputs | description |
| -------- | ------- | ----------- |
| train_data.csv  | runs/{RUN_ID} | Base folder for all results of the run. {RUN_ID} is automatically created as {timestamp}\_{target}\_{model} or {timestamp}\_{target}\_[frozen]{model} if the flag --freeze is used. |
| test_data.csv   | runs/{RUN_ID}/preds_train_{1,2,3,4,5}.pkl | For each training sample in the fold, real labels and predicted probability distribution. |
| split_info.json | runs/{RUN_ID}/preds_val_{1,2,3,4,5}.pkl | For each validation sample in the fold, real labels and predicted probability distribution. |
|                 | runs/{RUN_ID}/preds_test_ALL.pkl | For each test sample, real labels and predicted probability distribution. |
|                 | runs/{RUN_ID}/best_fold_{1,2,3,4,5}.pth | Weights of the best-performing model for each split/fold.  |

**`04_run_MTL.py`**
```bash
python 04_run_MTL.py [--model MODEL] [--alpha ALPHA]
  --model in [resnet, mobilenet, efficientnet, convnext, swint]
  --alpha float number between 0 and 1. Defaults to 0.5.
          Parameter to balance the MTL loss.
          0 = symptom only loss, 1 = disease only loss.
```
| inputs   | outputs | description |
| -------- | ------- | ----------- |
| train_data.csv  | runs/{SYM_RUN_ID,DIS_RUN_ID} | The two base folders for results of the run. <span style="color:tomato;">⚠️ Predictions for the symptom and disease prediction task of the same model are saved in separate folders</span> for ease of processing. The two IDs are automatically created as {timestamp}\_sym\_[MTL][alpha={alpha}]{model} and {timestamp}\_dis\_[MTL][alpha={alpha}]{model}.|
| test_data.csv   | runs/{SYM_RUN_ID,DIS_RUN_ID}/preds_train_{1,2,3,4,5}.pkl | For each training sample in the fold, real labels and predicted probability distribution. |
| split_info.json | runs/{SYM_RUN_ID,DIS_RUN_ID}/preds_val_{1,2,3,4,5}.pkl | For each validation sample in the fold, real labels and predicted probability distribution. |
|                 | runs/{SYM_RUN_ID,DIS_RUN_ID}/preds_test_ALL.pkl | For each test sample, real labels and predicted probability distribution. |
|                 | runs/{DIS_RUN_ID}/best_fold_{1,2,3,4,5}.pth | Weights of the best-performing model for each split/fold. <span style="color:tomato;">⚠️ WARNING: the model weights are stored ONLY in the {DIS_RUN_ID} folder, not in {SYM_RUN_ID}, to save space.</span> |

**`05_run_S2D-MTL.py`**
```bash
python 05_run_S2D-MTL.py [--model MODEL] [--alpha ALPHA] [--beta BETA] [--gamma GAMMA]
  --model in [resnet, mobilenet, efficientnet, convnext, swint]
  --alpha float number between 0 and 1. Defaults to 0.5.
          Parameter to balance the MTL loss.
          0 = symptom only loss, 1 = disease only loss.
  --beta  float number between 0 and 1. Defaults to 0.5.
          Parameter to balance the S2D-MTL loss
          0 = only secondary predictions, 1 = only primary predictions (same as MTL only).
  --gamma float number between. Defaults to 0.
          Parameter to balance the causal Matrix loss.
          0 = only S2D-MTL loss, >0 = add L2 penalty for co-occurence loss.
```
| inputs   | outputs | description |
| -------- | ------- | ----------- |
| train_data.csv  | runs/{SYM_RUN_ID,DIS_RUN_ID} | The two base folders for results of the run. <span style="color:tomato;">⚠️ Predictions for the symptom and disease prediction task of the same model are saved in separate folders</span> for ease of processing. The two IDs are automatically created as {timestamp}\_sym\_[MTL][alpha={alpha}][beta={beta}][gamma={gamma}]{model} and {timestamp}\_dis\_[S2D_MTL][alpha={alpha}][beta={beta}][gamma={gamma}]{model}.|
| test_data.csv   | runs/{SYM_RUN_ID,DIS_RUN_ID}/preds_train_{1,2,3,4,5}.pkl | For each training sample in the fold, real labels and predicted probability distribution. |
| split_info.json | runs/{SYM_RUN_ID,DIS_RUN_ID}/preds_val_{1,2,3,4,5}.pkl | For each validation sample in the fold, real labels and predicted probability distribution. |
|                 | runs/{SYM_RUN_ID,DIS_RUN_ID}/preds_test_ALL.pkl | For each test sample, real labels and predicted probability distribution. |
|                 | runs/{DIS_RUN_ID}/best_fold_{1,2,3,4,5}.pth | Weights of the best-performing model for each split/fold. <span style="color:tomato;">⚠️ WARNING: the model weights are stored ONLY in the {DIS_RUN_ID} folder, not in {SYM_RUN_ID}, to save space.</span> |
|                 | ./fold_{1,2,3,4,5}_co-occurrence.csv | Pre-calculated co-occurrence matrix of symptoms and diseases in the training set of the five folds. Used when gamma>0 (causal matrix loss). |

**`06_calculate_performances.py`**

| inputs   | outputs | description |
| -------- | ------- | ----------- |
| runs/{...} | performance_dis.csv | Table summarizing the metrics of all disease classification models found in the runs folder. <br/> Calculates: weighted F1 and sample F1 (with decision threshold of 0.5 or 0.3), average AUC, and AUC for each class. |
|            | performance_sym.csv | Table summarizing the metrics of all symptom classification models found in the runs folder. <br/> Calculates: weighted F1 and sample F1 (with decision threshold of 0.5 or 0.3), average AUC, and AUC for each class. |

---

## 📜 Acknowledgements & Funding
This work was supported by INTERREG VI-A "MARINET - MARICULTURE NETWORK: Implementazione di nuove tecnologie per un'acquacoltura diversificata e sostenibile rivolta a una società sana e a regioni competitive" - ID: ITHR0200334 - CUP: G33C23000660007.
