Explainable Multimodal ML for Knee Osteoarthritis Severity & Surgery Risk

I built this project to answer a question that kept bugging me: when doctors decide whether a knee osteoarthritis patient needs surgery, they're looking at an X-ray and thinking about the patient's age, weight, pain history, and so on — so why do most ML projects only ever use one of those at a time?

This is my attempt at combining both. It takes a knee X-ray, predicts how severe the osteoarthritis is (using the standard Kellgren-Lawrence 0–4 grading scale), combines that with the patient's clinical data, and predicts whether surgery is a likely outcome — while also explaining why it made both predictions, instead of just spitting out a number.

Why I built it this way

I could have thrown a fancy joint neural network at this and tried to squeeze out an extra percentage point of accuracy. I didn't, on purpose. The goal here was a system I could actually explain — to a recruiter, to myself six months from now, to anyone who wants to know how it works, not just that it "works." So every choice below is one I can defend, not just one that happened to perform well on a leaderboard.

What it actually does
Knee X-ray ──► ResNet18 ──► Predicted KL Grade (0-4)
                                     │
                                     ▼
Clinical Data ──► preprocessing ──► combine features
                                     │
                                     ▼
                              Random Forest
                                     │
                                     ▼
                        Surgery: Yes / No + probability
Image branch — a pretrained ResNet18 fine-tuned to classify X-rays into KL grades 0 through 4.
Clinical branch — age, height, weight, BMI, and pain frequency, cleaned and encoded.
Late fusion — the predicted KL grade (never the ground-truth one — that would be cheating) gets combined with the clinical features.
Random Forest — takes that combined vector and predicts whether the patient ends up having surgery, plus a probability.
Explanations — Grad-CAM shows which parts of the X-ray the model was actually looking at, and SHAP shows which clinical features pushed the surgery prediction up or down, for both the model overall and for one individual patient at a time.
Datasets

Both come from Kaggle and are public research data from the Osteoarthritis Initiative (OAI):

X-rays — Knee Osteoarthritis Dataset with Severity, organized into train/val/test/auto_test folders, each split into subfolders 0 through 4 by KL grade.
Clinical data — OAI Clinical, a CSV with patient ID, side, age, height, weight, BMI, pain frequency, and whether they had surgery.

The tricky part was actually matching the two datasets up — the X-ray filenames and the clinical records don't share one obvious clean key straight out of the box. I wrote a mapping step that extracts the patient ID (and side, when it's parseable) from the filenames, joins it to the clinical table, and then prints out how many images matched, how many didn't, and why — so I could actually verify the mapping was right before trusting anything downstream. I'd rather see "600 unmatched images, go fix this" than silently train on garbage.

A few decisions worth knowing about
ResNet18, not something bigger. The dataset isn't huge, and a smaller pretrained backbone fine-tunes well without overfitting. Bigger models weren't going to buy much here.
Late fusion, not a joint network. Combining the two modalities after each one does its own job separately is simpler to build, easier to debug, and — importantly — easier to explain. You can point at the pipeline and say exactly what each part contributed.
Random Forest, not XGBoost. For a tabular dataset this size, Random Forest is plenty good, needs way less tuning, and plays nicely with SHAP's tree explainer.
Patient ID is deliberately excluded as a feature. It's just a label, not a medical signal — using it would let the model cheat by memorizing patients instead of learning anything generalizable.
Patient-level train/test split. Since most patients have both a left and right knee in the dataset, a random split could put one knee in training and the other in testing — which leaks information between the two. I split by patient ID instead, so a person is entirely in one split or the other.
Quadratic Weighted Kappa for the KL-grade model, because KL grades are ordinal — predicting grade 3 when the truth is grade 4 is a much smaller mistake than predicting grade 0, and plain accuracy doesn't know the difference.
Explainability, not just predictions

I didn't want a black box, so:

Grad-CAM overlays a heatmap on the X-ray showing which regions influenced the model's KL-grade prediction. It doesn't prove the model found a specific pathology — it just shows where its attention went — but it's a genuinely useful sanity check.
SHAP breaks down the surgery prediction feature by feature, both globally (what matters across all patients) and for a single patient (why this person's prediction came out the way it did, and in which direction each factor pushed it).
Repo structure
├── koa_multimodal_project.ipynb   # the full pipeline, section by section
├── clinical_info.csv              # clinical dataset (or your own copy)
└── archive/                       # unzipped X-ray dataset (train/val/test/auto_test)
Running it

The notebook is written for Google Colab but runs anywhere with the right packages installed.

Grab both datasets from Kaggle (links above) and unzip the X-ray archive.
Point the two config variables at your data:
python
   ARCHIVE_DIR = "/content/archive"
   CLINICAL_CSV = "/content/clinical_info.csv"
Run the notebook top to bottom. It walks through mapping validation, training, evaluation, and the explainability sections in order — nothing runs out of sequence.
Results

(Fill this in once you've run it on your machine — accuracy, F1, and QWK for the KL-grade model, and accuracy, recall, and ROC-AUC for the surgery model, straight from the notebook's evaluation cells.)

Honest limitations

I'd rather list these than have someone find them later:

This is trained and tested on one research cohort (OAI). It hasn't been validated on an independent population, and it isn't a diagnostic tool.
The X-ray-to-clinical-record mapping is inferred from filename patterns, not a guaranteed clean key — worth spot-checking if you're using a different copy of the data.
Errors in the KL-grade prediction can carry through into the surgery prediction, since the pipeline is two stages, not one joint model.
Grad-CAM and SHAP explain what the model learned, not clinical ground truth — they're a window into the model's reasoning, not a medical opinion.
Disclaimer

This is a machine learning prototype built on public research data, for learning and portfolio purposes. It is not a clinically validated tool and should never be used to make real medical decisions.

License

MIT — feel free to use, learn from, or build on this.
