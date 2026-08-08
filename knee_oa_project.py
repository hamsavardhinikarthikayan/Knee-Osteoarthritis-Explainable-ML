"""
Explainable Knee Osteoarthritis Grading
=======================================

Two-model academic/resume project:
1) Clinical data -> Random Forest -> KL Grade 0-4 -> SHAP
2) Knee X-ray -> ResNet18 -> KL Grade 0-4 -> Grad-CAM

Install:
pip install pandas numpy matplotlib seaborn scikit-learn shap joblib
pip install torch torchvision pillow grad-cam


"""

from pathlib import Path
import random
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

CLINICAL_FILE = DATA_DIR / "clinical_info.csv"
KL_FILE = DATA_DIR / "xr_kl_os_jsn.csv"
IMAGE_DIR = DATA_DIR / "images"

TRAIN_IMAGE_DIR = IMAGE_DIR / "train"
VAL_IMAGE_DIR = IMAGE_DIR / "val"
TEST_IMAGE_DIR = IMAGE_DIR / "test"

RANDOM_STATE = 42
NUM_CLASSES = 5
CLASS_NAMES = ["KL 0", "KL 1", "KL 2", "KL 3", "KL 4"]

NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# These are deliberately limited to direct clinical variables.
# ID/SIDE/FILENAME are identifiers, not predictive features.
# SURGERY/RISK/SXKOA/SYMPTOMATIC are excluded initially because
# they should be reviewed for possible leakage/downstream information.
CLINICAL_FEATURES = [
    "AGE",
    "HEIGHT",
    "WEIGHT",
    "MAX WEIGHT",
    "BMI",
    "FREQUENT PAIN",
    "SWELLING",
    "BENDING FULLY",
    "CREPITUS",
    "KOOS PAIN SCORE",
]


# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def set_seed(seed=RANDOM_STATE):
    """Make results as reproducible as practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_directories():
    """Create folders used for saved models and results."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def check_exists(path):
    """Raise a useful error when a required file/folder is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Not found: {path}\n"
            "Check the project folder structure."
        )


# ============================================================
# 3. CLINICAL DATA: LOAD + MERGE
# ============================================================

def load_and_merge_data():
    """
    clinical_info.csv supplies clinical features.
    xr_kl_os_jsn.csv supplies the KL Grade target.

    Records are matched using ID + SIDE.
    """

    check_exists(CLINICAL_FILE)
    check_exists(KL_FILE)

    clinical = pd.read_csv(CLINICAL_FILE)
    kl = pd.read_csv(KL_FILE)

    # Remove accidental spaces in column names.
    clinical.columns = clinical.columns.str.strip()
    kl.columns = kl.columns.str.strip()

    required_clinical = {"ID", "SIDE", "FILENAME"}
    required_kl = {"ID", "SIDE", "FILENAME", "KL Grade"}

    missing_1 = required_clinical - set(clinical.columns)
    missing_2 = required_kl - set(kl.columns)

    if missing_1:
        raise ValueError(f"Clinical file missing: {missing_1}")
    if missing_2:
        raise ValueError(f"KL file missing: {missing_2}")

    # Make IDs and side values consistent.
    clinical["ID"] = clinical["ID"].astype(str).str.strip()
    kl["ID"] = kl["ID"].astype(str).str.strip()

    clinical["SIDE"] = (
        clinical["SIDE"].astype(str).str.strip().str.upper()
    )
    kl["SIDE"] = (
        kl["SIDE"].astype(str).str.strip().str.upper()
    )

    merged = pd.merge(
        clinical,
        kl[["ID", "SIDE", "FILENAME", "KL Grade"]],
        on=["ID", "SIDE"],
        how="inner",
        suffixes=("_clinical", "_kl"),
    )

    # Validate that the corresponding filenames agree.
    filename_match = (
        merged["FILENAME_clinical"].astype(str).str.strip()
        == merged["FILENAME_kl"].astype(str).str.strip()
    )

    print("\n========== DATASET SUMMARY ==========")
    print("Clinical rows:", len(clinical))
    print("KL rows:", len(kl))
    print("Merged rows:", len(merged))
    print("Filename matches:", int(filename_match.sum()))
    print("Filename mismatches:", int((~filename_match).sum()))

    merged = merged[merged["KL Grade"].isin(range(NUM_CLASSES))].copy()
    merged["KL Grade"] = merged["KL Grade"].astype(int)

    print("\nKL distribution:")
    print(merged["KL Grade"].value_counts().sort_index())

    return merged


# ============================================================
# 4. CLINICAL MODEL: RANDOM FOREST
# ============================================================

def create_preprocessor(X):
    """Create safe preprocessing for numerical and categorical features."""

    numerical = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    return ColumnTransformer([
        ("num", numerical_pipeline, numerical),
        ("cat", categorical_pipeline, categorical)
    ])


def train_clinical_model(merged):
    """Train, evaluate and save the Random Forest clinical model."""

    missing = [
        c for c in CLINICAL_FEATURES
        if c not in merged.columns
    ]

    if missing:
        raise ValueError(
            f"Clinical features not found in CSV: {missing}"
        )

    X = merged[CLINICAL_FEATURES].copy()
    y = merged["KL Grade"].copy()

    # Stratification keeps the KL-grade proportions similar.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    preprocessor = create_preprocessor(X_train)

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", rf)
    ])

    print("\n========== TRAINING RANDOM FOREST ==========")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    print(f"Clinical accuracy: {accuracy:.4f}")
    print(f"Clinical macro-F1: {macro_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(NUM_CLASSES))
    )

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.xlabel("Predicted KL Grade")
    plt.ylabel("Actual KL Grade")
    plt.title("Random Forest - Clinical Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "clinical_confusion_matrix.png",
        dpi=200
    )
    plt.close()

    joblib.dump(
        pipeline,
        MODEL_DIR / "random_forest_clinical_pipeline.pkl"
    )

    return pipeline, X_test, y_test


# ============================================================
# 5. SHAP
# ============================================================

def generate_shap_explanation(pipeline, X_test):
    """
    Explain the trained Random Forest.

    SHAP is calculated after preprocessing because the actual
    Random Forest receives the transformed numeric matrix.
    """

    print("\n========== GENERATING SHAP EXPLANATION ==========")

    preprocessor = pipeline.named_steps["preprocessor"]
    rf = pipeline.named_steps["classifier"]

    X_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    # Limit the explanation size so it remains practical.
    X_sample = X_transformed[:min(500, len(X_transformed))]

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample)

    # SHAP has changed its multiclass return format across versions.
    if isinstance(shap_values, list):
        # list[class] -> samples x features
        importance = np.stack(
            [np.abs(v) for v in shap_values]
        ).mean(axis=(0, 1))

    else:
        arr = np.asarray(shap_values)

        if arr.ndim == 3:
            # Usually samples x features x classes in recent versions.
            if arr.shape[1] == len(feature_names):
                importance = np.abs(arr).mean(axis=(0, 2))
            else:
                importance = np.abs(arr).mean(axis=(0, 1))

        elif arr.ndim == 2:
            importance = np.abs(arr).mean(axis=0)

        else:
            raise ValueError(
                f"Unexpected SHAP shape: {arr.shape}"
            )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": importance
    }).sort_values(
        "mean_abs_shap",
        ascending=False
    )

    print("\nTop SHAP features:")
    print(importance_df.head(15).to_string(index=False))

    top = importance_df.head(15).sort_values(
        "mean_abs_shap"
    )

    plt.figure(figsize=(9, 7))
    plt.barh(
        top["feature"],
        top["mean_abs_shap"]
    )
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.title("Clinical Feature Importance using SHAP")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "clinical_shap_importance.png",
        dpi=200
    )
    plt.close()

    importance_df.to_csv(
        OUTPUT_DIR / "clinical_shap_importance.csv",
        index=False
    )


# ============================================================
# 6. IMAGE DATA
# ============================================================

def create_image_datasets():
    """Load train/validation/test X-rays using ImageFolder."""

    for folder in [
        TRAIN_IMAGE_DIR,
        VAL_IMAGE_DIR,
        TEST_IMAGE_DIR
    ]:
        check_exists(folder)

    # Training can use mild augmentation.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Validation/test must be deterministic.
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train = datasets.ImageFolder(
        TRAIN_IMAGE_DIR,
        transform=train_transform
    )

    val = datasets.ImageFolder(
        VAL_IMAGE_DIR,
        transform=test_transform
    )

    test = datasets.ImageFolder(
        TEST_IMAGE_DIR,
        transform=test_transform
    )

    print("\n========== IMAGE DATA ==========")
    print("Train:", len(train))
    print("Validation:", len(val))
    print("Test:", len(test))
    print("Class mapping:", train.class_to_idx)

    return train, val, test


# ============================================================
# 7. RESNET18
# ============================================================

def build_resnet18():
    """Load pretrained ResNet18 and change the output to five classes."""

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Original ResNet18 predicts 1000 ImageNet classes.
    # Our problem has five KL grades.
    input_features = model.fc.in_features

    model.fc = nn.Linear(
        input_features,
        NUM_CLASSES
    )

    return model.to(DEVICE)


def train_one_epoch(model, loader, loss_fn, optimizer):
    """One complete training pass over the training data."""

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear gradients from the previous batch.
        optimizer.zero_grad()

        # Forward pass: images -> predictions.
        outputs = model(images)

        # Measure prediction error.
        loss = loss_fn(outputs, labels)

        # Backpropagation: calculate gradients.
        loss.backward()

        # Update weights.
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate_image_model(model, loader, loss_fn):
    """Evaluate without changing model weights."""

    model.eval()

    total_loss = 0.0
    labels_all = []
    predictions_all = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = loss_fn(outputs, labels)

            total_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            predictions_all.extend(
                predictions.cpu().numpy()
            )
            labels_all.extend(
                labels.cpu().numpy()
            )

    loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        labels_all,
        predictions_all
    )

    macro_f1 = f1_score(
        labels_all,
        predictions_all,
        average="macro"
    )

    return (
        loss,
        accuracy,
        macro_f1,
        labels_all,
        predictions_all
    )


def train_resnet18(train_dataset, val_dataset):
    """Train ResNet18 using transfer learning."""

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    model = build_resnet18()

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_accuracy = -1.0

    print("\n========== TRAINING RESNET18 ==========")
    print("Device:", DEVICE)

    for epoch in range(NUM_EPOCHS):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer
        )

        val_loss, val_acc, val_f1, _, _ = (
            evaluate_image_model(
                model,
                val_loader,
                loss_fn
            )
        )

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f} | "
            f"Val Macro-F1 {val_f1:.4f}"
        )

        # Save the best validation model.
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(
                model.state_dict(),
                MODEL_DIR / "resnet18_best.pth"
            )

    # Load the best model, not merely the last epoch.
    model.load_state_dict(
        torch.load(
            MODEL_DIR / "resnet18_best.pth",
            map_location=DEVICE
        )
    )

    return model


# ============================================================
# 8. FINAL IMAGE EVALUATION
# ============================================================

def evaluate_resnet18(model, test_dataset):

    loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    loss_fn = nn.CrossEntropyLoss()

    (
        loss,
        accuracy,
        macro_f1,
        labels,
        predictions
    ) = evaluate_image_model(
        model,
        loader,
        loss_fn
    )

    print("\n========== FINAL IMAGE RESULTS ==========")
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro-F1: {macro_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            labels,
            predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES))
    )

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.xlabel("Predicted KL Grade")
    plt.ylabel("Actual KL Grade")
    plt.title("ResNet18 - X-ray Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "image_confusion_matrix.png",
        dpi=200
    )
    plt.close()


# ============================================================
# 9. GRAD-CAM
# ============================================================

def generate_gradcam(model, image_path):
    """
    Generate a Grad-CAM image for one X-ray.

    Grad-CAM highlights image regions associated with the model's
    prediction. It does not prove that those regions are clinically
    causal or diagnostically correct.
    """

    image_path = Path(image_path)
    check_exists(image_path)

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    original = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = transform(
        original
    ).unsqueeze(0).to(DEVICE)

    # Last convolutional block of ResNet18.
    target_layers = [model.layer4[-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    # Grad-CAM needs gradients even though this is inference.
    with torch.enable_grad():

        output = model(input_tensor)

        predicted_class = int(
            torch.argmax(
                output,
                dim=1
            ).item()
        )

        targets = [
            ClassifierOutputTarget(
                predicted_class
            )
        ]

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

    display_image = original.resize(
        (224, 224)
    )

    rgb_image = (
        np.asarray(display_image)
        .astype(np.float32)
        / 255.0
    )

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    output_file = (
        OUTPUT_DIR
        / f"gradcam_{image_path.stem}.jpg"
    )

    Image.fromarray(
        visualization
    ).save(output_file)

    print("\nGrad-CAM saved:", output_file)
    print(
        "Predicted KL Grade:",
        predicted_class
    )

    return output_file, predicted_class


# ============================================================
# 10. MAIN
# ============================================================

def main():

    warnings.filterwarnings("ignore")
    set_seed()
    make_directories()

    print("==========================================")
    print(" EXPLAINABLE KNEE OA GRADING PROJECT")
    print("==========================================")
    print("Device:", DEVICE)

    # -------- Clinical branch --------
    merged = load_and_merge_data()

    pipeline, X_test, y_test = train_clinical_model(
        merged
    )

    generate_shap_explanation(
        pipeline,
        X_test
    )

    # -------- Image branch --------
    train_data, val_data, test_data = (
        create_image_datasets()
    )

    image_model = train_resnet18(
        train_data,
        val_data
    )

    evaluate_resnet18(
        image_model,
        test_data
    )

    print("\n==========================================")
    print(" TRAINING COMPLETE")
    print("==========================================")
    print("Models saved in:", MODEL_DIR)
    print("Outputs saved in:", OUTPUT_DIR)
    print(
        "\nTo create Grad-CAM for an X-ray, use:"
    )
    print(
        "generate_gradcam("
        "image_model, "
        "'data/images/test/3/example.png'"
        ")"
    )


if __name__ == "__main__":
    main()
