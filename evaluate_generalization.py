import re
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import RobertaTokenizerFast, RobertaModel
from tqdm.auto import tqdm

# ── CONFIG ────────────────────────────────────────────────────────
DATA_PATH   = "data.csv"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH  = 48
BATCH_SIZE  = 64
HEAD_LR     = 1e-3
EPOCHS      = 3
DROPOUT_P   = 0.3
WEIGHT_DECAY= 0.05
LABEL_SMOOTHING = 0.1
PATIENCE    = 1  # early stopping patience

# Task configurations
task_configs = {
    "depression": {
        "items": [3,5,10,13,16,17,21,24,26,31,34,37,38,42],
        "checkpoint": "best_depression_severity_roberta.pt",
        "thresholds": [(9,0),(13,1),(20,2),(27,3),(float('inf'),4)]
    },
    "anxiety": {
        "items": [2,4,7,9,15,19,20,23,25,28,36,40,41],
        "checkpoint": "best_anxiety_severity_roberta.pt",
        "thresholds": [(7,0),(9,1),(14,2),(19,3),(float('inf'),4)]
    },
    "stress": {
        "items": [1,6,8,11,12,14,18,22,27,29,30,32,33,35,39],
        "checkpoint": "best_stress_severity_roberta.pt",
        "thresholds": [(14,0),(18,1),(25,2),(33,3),(float('inf'),4)]
    }
}

# Load and preprocess once
df = pd.read_csv(DATA_PATH, sep="\t", engine="python")
answer_cols = sorted([c for c in df.columns if re.fullmatch(r"Q\d+A", c)],
                     key=lambda x: int(x[1:-1]))
df["processed_text"] = df[answer_cols].astype(str).agg(" ".join, axis=1)
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# Dataset class
class SimpleDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], truncation=True, padding="max_length",
                        max_length=MAX_LENGTH, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Head-only model
class BaseHeadClassifier(nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        from transformers import RobertaModel
        # Load *base* roberta—no fine-tuned weights
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, 5)
        # Freeze the encoder
        for p in self.roberta.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = (
            out.pooler_output
            if hasattr(out, "pooler_output") and out.pooler_output is not None
            else out.last_hidden_state[:, 0, :]
        )
        return self.classifier(self.dropout(cls))

# Helper: map raw score to severity
def make_map(thresholds):
    def mapper(score):
        for thresh, label in thresholds:
            if score <= thresh:
                return label
    return mapper

# Function: train on hold-out, evaluate test, return test accuracy
def holdout_experiment(name, cfg):
    print(f"\n>> Hold-out for {name}")
    # compute raw and severity
    raw_col = f"{name}_raw"
    df[raw_col] = df[[f"Q{idx}A" for idx in cfg["items"]]].sum(axis=1)
    df[f"{name}_severity"] = df[raw_col].apply(make_map(cfg["thresholds"]))
    labels = df[f"{name}_severity"]
    # splits
    train_val, test_df = train_test_split(df, test_size=0.20, stratify=labels, random_state=42)
    train_df, val_df  = train_test_split(train_val, test_size=0.25, stratify=train_val[f"{name}_severity"], random_state=42)
    # dataloaders
    train_loader = DataLoader(SimpleDataset(train_df["processed_text"].tolist(), train_df[f"{name}_severity"].tolist()),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(SimpleDataset(val_df["processed_text"].tolist(),   val_df[f"{name}_severity"].tolist()),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(SimpleDataset(test_df["processed_text"].tolist(),  test_df[f"{name}_severity"].tolist()),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # model, optimizer, criterion with label smoothing
    model = BaseHeadClassifier(DROPOUT_P).to(DEVICE)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # early stopping
    best_val, patience = 0, 0
    for epoch in range(EPOCHS):
        model.train(); losses=[]
        for batch in train_loader:
            optimizer.zero_grad()
            ids=batch["input_ids"].to(DEVICE); mask=batch["attention_mask"].to(DEVICE)
            labs=batch["labels"].to(DEVICE)
            logits=model(ids,mask); loss=criterion(logits,labs)
            loss.backward(); optimizer.step()
            losses.append(loss.item())
        # eval val
        model.eval(); correct=0; total=0
        with torch.no_grad():
            for batch in val_loader:
                ids=batch["input_ids"].to(DEVICE); mask=batch["attention_mask"].to(DEVICE)
                labs=batch["labels"].to(DEVICE)
                preds=model(ids,mask).argmax(dim=1)
                correct+= (preds==labs).sum().item(); total+=labs.size(0)
        val_acc=correct/total
        print(f"Epoch {epoch+1} val_acc={val_acc:.4f}")
        if val_acc>best_val:
            best_val=val_acc; patience=0
        else:
            patience+=1
            if patience>PATIENCE:
                print("Early stopping")
                break

    # test evaluation
    model.eval(); correct=0; total=0
    with torch.no_grad():
        for batch in test_loader:
            ids=batch["input_ids"].to(DEVICE); mask=batch["attention_mask"].to(DEVICE)
            labs=batch["labels"].to(DEVICE)
            preds=model(ids,mask).argmax(dim=1)
            correct+= (preds==labs).sum().item(); total+=labs.size(0)
    test_acc = correct/total
    print(f"{name} test accuracy: {test_acc:.4f}")
    return test_acc

# Function: 5-fold cross-validation
def cross_val_experiment(name, cfg):
    print(f"\n>> 5-Fold CV for {name}")
    labels = df[f"{name}_severity"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores=[]
    for fold, (train_idx, test_idx) in enumerate(skf.split(df, labels),1):
        fold_train = df.iloc[train_idx]; fold_test=df.iloc[test_idx]
        train_loader = DataLoader(SimpleDataset(fold_train["processed_text"].tolist(),
                                                fold_train[f"{name}_severity"].tolist()),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        test_loader  = DataLoader(SimpleDataset(fold_test["processed_text"].tolist(),
                                                fold_test[f"{name}_severity"].tolist()),
                                  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        model = BaseHeadClassifier(DROPOUT_P).to(DEVICE)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        # train head-only quickly (no val split here, fixed epochs)
        for _ in range(EPOCHS):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                ids=batch["input_ids"].to(DEVICE); mask=batch["attention_mask"].to(DEVICE)
                labs=batch["labels"].to(DEVICE)
                logits=model(ids,mask); loss=criterion(logits,labs)
                loss.backward(); optimizer.step()
        # eval on fold_test
        model.eval(); correct=0; total=0
        with torch.no_grad():
            for batch in test_loader:
                ids=batch["input_ids"].to(DEVICE); mask=batch["attention_mask"].to(DEVICE)
                labs=batch["labels"].to(DEVICE)
                preds=model(ids,mask).argmax(dim=1)
                correct+= (preds==labs).sum().item(); total+=labs.size(0)
        acc=correct/total
        print(f"Fold {fold} accuracy: {acc:.4f}")
        scores.append(acc)
    mean, std = np.mean(scores), np.std(scores)
    print(f"{name} 5-fold CV mean ± std = {mean:.4f} ± {std:.4f}")
    return mean, std

if __name__ == "__main__":
    results = {}
    for name, cfg in task_configs.items():
        test_acc = holdout_experiment(name, cfg)
        cv_mean, cv_std = cross_val_experiment(name, cfg)
        results[name] = {"test_acc":test_acc, "cv_mean":cv_mean, "cv_std":cv_std}
    print("\nSummary:", results)
