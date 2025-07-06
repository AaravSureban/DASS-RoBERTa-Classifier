import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import RobertaTokenizerFast, RobertaModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm.auto import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# 1) Reproducibility: set random seeds everywhere
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# 2) Configuration
# ──────────────────────────────────────────────────────────────────────────────
DATA_CSV_PATH = "data.csv"  
# (replace with the actual filename you downloaded from Kaggle)

BATCH_SIZE = 32
NUM_EPOCHS = 4        # as per Table 6: 4 epochs per task
LEARNING_RATE = 2e-5   # AdamW with weight decay
WEIGHT_DECAY = 0.01
MAX_LENGTH = 48       # max tokens per example
DROPOUT_RATE = 0.1

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f">>> Using device: {DEVICE}")

# ──────────────────────────────────────────────────────────────────────────────
# 3) Load raw CSV and preprocess into three separate classification tasks:
#    – “depression_severity” (0–4)
#    – “anxiety_severity”    (0–4)
#    – “stress_severity”     (0–4)
#
#    The Kaggle file has individual DASS-42 item columns (e.g. Q1_answer, Q1_rt, …).
#    We need to:
#      1) Identify which of the 42 items belong to Depression, which to Anxiety, which to Stress.
#      2) Sum those item‐scores to get a total score for each subscale.
#      3) Map the total into severity levels (0 = Normal,1 = Mild,2 = Moderate,3 = Severe,4 = Extremely Severe).
#    4) Concatenate all 42 individual “answer” columns into one long space-separated string
#       (so that RoBERTa sees a “flattened” representation of every answer in textual form).
#       In effect, each answer (0–3) becomes a token that RoBERTa can attend over.
#    That matches “We consolidated the relevant columns…into a single processed text column.”
# ──────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(DATA_CSV_PATH, sep="\t", engine="python")
print(df.shape)
print(df.columns[:10])
print(df.head(3))

# (A) Identify the 42 DASS items and which subscale they belong to.
# From the official DASS-42 scoring key:
#  – Depression items: 3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42
#  – Anxiety items:    2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 36, 40, 41
#  – Stress items:     1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 30, 32, 33, 35, 39

depr_items = [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42]
anx_items  = [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 36, 40, 41]
stres_items= [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 30, 32, 33, 35, 39]

# In the Kaggle file, each question has 3 columns: e.g. “Q1_answer”, “Q1_ms”, “Q1_idx”
# We only need the “_answer” column for each DASS question.
# Confirm the naming convention:
answer_cols = [c for c in df.columns if c.endswith("A")]
# We expect 42 “Q#__answer” columns.
if len(answer_cols) != 42:
    raise ValueError(f"Expected 42 answer‐columns, found {len(answer_cols)}")

# Create three new columns: “depression_raw”, “anxiety_raw”, “stress_raw” (sums of item‐scores).
df["depression_raw"] = df[[f"Q{idx}A" for idx in depr_items]].sum(axis=1)
df["anxiety_raw"]    = df[[f"Q{idx}A" for idx in anx_items]].sum(axis=1)
df["stress_raw"]     = df[[f"Q{idx}A" for idx in stres_items]].sum(axis=1)

# Define severity‐level mapping based on DASS-42 scoring conventions:
def map_to_severity(score: int, subscale: str) -> int:
    """
    Returns 0–4 severity group for a given raw DASS subscale score.
    Thresholds (Depression, Anxiety, Stress) differ slightly:
     – Depression:   Normal 0–9,   Mild 10–13, Moderate 14–20, Severe 21–27, Extremely Severe 28+
     – Anxiety:      Normal 0–7,   Mild 8–9,   Moderate 10–14, Severe 15–19, Extremely Severe 20+
     – Stress:       Normal 0–14,  Mild 15–18, Moderate 19–25, Severe 26–33, Extremely Severe 34+
    """
    if subscale == "depression":
        if score <= 9:   return 0
        if score <= 13:  return 1
        if score <= 20:  return 2
        if score <= 27:  return 3
        return 4
    elif subscale == "anxiety":
        if score <= 7:   return 0
        if score <= 9:   return 1
        if score <= 14:  return 2
        if score <= 19:  return 3
        return 4
    elif subscale == "stress":
        if score <= 14:  return 0
        if score <= 18:  return 1
        if score <= 25:  return 2
        if score <= 33:  return 3
        return 4
    else:
        raise ValueError(f"Unknown subscale: {subscale}")

df["depression_severity"] = df["depression_raw"].apply(lambda x: map_to_severity(x, "depression"))
df["anxiety_severity"]    = df["anxiety_raw"].apply(lambda x: map_to_severity(x, "anxiety"))
df["stress_severity"]     = df["stress_raw"].apply(lambda x: map_to_severity(x, "stress"))

# (B) We need a SINGLE “processed_text” column for RoBERTa:
#     Concatenate all 42 “Q#__answer” into a single space-separated string.
#     Example: if Q1_answer=2, Q2_answer=1, …, Q42_answer=0, then:
#      processed_text = "2 1 0 … 3"
#
#     This way, RoBERTa sees each numeric answer as a token (“2” → <token “2”>).
cols_answer = sorted(answer_cols, key=lambda x: int(x[1:-1]))
df["processed_text"] = df[cols_answer].astype(str).agg(" ".join, axis=1)

# Finally, drop any rows with missing data (just in case)
df = df.dropna(subset=["processed_text", "depression_severity", "anxiety_severity", "stress_severity"]).reset_index(drop=True)
print(f">>> Total examples after cleaning: {len(df)}")


# ──────────────────────────────────────────────────────────────────────────────
# 4) We will train three separate RoBERTa‐based classifiers:
#    – one for depression (5 labels: 0–4)
#    – one for anxiety    (5 labels: 0–4)
#    – one for stress     (5 labels: 0–4)
#
#    Each uses the **same** “processed_text” as input, with a different set of labels.
# ──────────────────────────────────────────────────────────────────────────────

# (A) Split into train & test for each subtask (80/20 stratified)
train_df_d, test_df_d = train_test_split(
    df, test_size=0.20, stratify=df["depression_severity"], random_state=SEED
)
train_df_a, test_df_a = train_test_split(
    df, test_size=0.20, stratify=df["anxiety_severity"], random_state=SEED
)
train_df_s, test_df_s = train_test_split(
    df, test_size=0.20, stratify=df["stress_severity"], random_state=SEED
)


# (B) Define a PyTorch Dataset that tokenizes “processed_text” → input_ids, attention_mask
class DASSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer: RobertaTokenizerFast, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        txt = self.texts[idx]
        label = self.labels[idx]
        # tokenize + encode
        encoding = self.tokenizer(
            txt,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        # squeeze out the batch dimension
        input_ids = encoding["input_ids"].squeeze(0)        # torch.LongTensor of shape (max_len,)
        attention_mask = encoding["attention_mask"].squeeze(0)  # torch.LongTensor of shape (max_len,)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }


# (C) Instantiate one tokenizer (RoBERTa) and one base model
MODEL_NAME = "roberta-base"  # paper used RoBERTa‐Large if GPU allows; here we pick “base” for speed.
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

# ──────────────────────────────────────────────────────────────────────────────
# 5) Build a small classifier module that wraps RoBERTa + a dropout + linear
# ──────────────────────────────────────────────────────────────────────────────
class DASSTaskClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout_rate: float):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size  # typically 768 for “base”
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # “last_hidden_state” is (batch_size, seq_len, hidden_size)
        # We’ll take the [CLS] token representation → outputs.pooler_output (if available)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            cls_repr = outputs.pooler_output            # shape: (batch_size, hidden_size)
        else:
            # fall back to first token of last_hidden_state
            cls_repr = outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_size)
        x = self.dropout(cls_repr)
        logits = self.classifier(x)                     # shape: (batch_size, num_labels)
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# 6) A helper to evaluate accuracy / precision / recall / f1 on a DataLoader
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device):
    model.eval()
    preds = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(true_labels, preds, average="weighted")
    return acc, prec, rec, f1


# ──────────────────────────────────────────────────────────────────────────────
# 7) Training loop for a single task (given train/test DataFrames and label‐column name)
# ──────────────────────────────────────────────────────────────────────────────
def train_task(train_df: pd.DataFrame,
               test_df: pd.DataFrame,
               label_col: str,
               num_labels: int):
    """
    Trains a DASSTaskClassifier on the given (train_df, test_df).
    label_col is one of: “depression_severity”, “anxiety_severity”, “stress_severity”.
    Returns the trained model.
    """
    print(f"\n>>> Starting training for: {label_col}")

    # Build train/test datasets
    train_texts = train_df["processed_text"].tolist()
    train_labels = train_df[label_col].tolist()
    test_texts = test_df["processed_text"].tolist()
    test_labels = test_df[label_col].tolist()

    train_dataset = DASSDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    test_dataset  = DASSDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Instantiate model
    model = DASSTaskClassifier(MODEL_NAME, num_labels=num_labels, dropout_rate=DROPOUT_RATE)
    model.to(DEVICE)

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Track best validation accuracy
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} ({label_col})", leave=False)
        for batch in pbar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            epoch_losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            pbar.set_postfix_str(f"loss={np.mean(epoch_losses):.4f}")

        # After each epoch, evaluate on validation (test) set
        val_acc, val_prec, val_rec, val_f1 = evaluate(model, test_loader, DEVICE)
        print(f"Epoch {epoch+1} – {label_col}  Val Acc: {val_acc:.4f}  "
              f"Val Prec: {val_prec:.4f}  Val Rec: {val_rec:.4f}  Val F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best weights
            torch.save(model.state_dict(), f"best_{label_col}_roberta.pt")

    print(f">>> Finished training {label_col}. Best Val Acc: {best_val_acc:.4f}")
    # Load best weights back
    model.load_state_dict(torch.load(f"best_{label_col}_roberta.pt"))
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 8) Train all three tasks in sequence (you can also parallel‐ize if you have multiple GPUs)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Depression classifier (5 classes: 0–4)
    model_dep = train_task(train_df_d, test_df_d, label_col="depression_severity", num_labels=5)

    # Anxiety classifier (5 classes: 0–4)
    model_anx = train_task(train_df_a, test_df_a, label_col="anxiety_severity", num_labels=5)

    # Stress classifier (5 classes: 0–4)
    model_str = train_task(train_df_s, test_df_s, label_col="stress_severity", num_labels=5)

    # After training all three, evaluate final on test splits and print final metrics
    print("\n>>> Final evaluation on test sets:")
    for (model, test_df, label_col) in [
        (model_dep, test_df_d, "depression_severity"),
        (model_anx, test_df_a, "anxiety_severity"),
        (model_str, test_df_s, "stress_severity"),
    ]:
        # build a DataLoader just for final evaluation
        test_dataset = DASSDataset(
            test_df["processed_text"].tolist(),
            test_df[label_col].tolist(),
            tokenizer,
            MAX_LENGTH,
        )
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        acc, prec, rec, f1 = evaluate(model, test_loader, DEVICE)
        print(f"{label_col} –  Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")
