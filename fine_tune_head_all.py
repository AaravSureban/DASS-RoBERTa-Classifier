import re
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import RobertaTokenizerFast, RobertaModel
from tqdm.auto import tqdm

# ── CONFIG ─────────────────────────────────────────────────────────
DATA_PATH = "data.csv"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN   = 48
BATCH     = 64
HEAD_LR   = 1e-3
EPOCHS    = 3
DROPOUT   = 0.3
WEIGHT_DECAY = 0.05

# Define the three tasks
TASKS = {
  "depression": {
    "items":     [3,5,10,13,16,17,21,24,26,31,34,37,38,42],
    "checkpoint":"best_depression_severity_roberta.pt",
    "output":    "depression_head_finetuned.pt",
    "thresholds":[(9,0),(13,1),(20,2),(27,3),(float("inf"),4)]
  },
  "anxiety": {
    "items":     [2,4,7,9,15,19,20,23,25,28,36,40,41],
    "checkpoint":"best_anxiety_severity_roberta.pt",
    "output":    "anxiety_head_finetuned.pt",
    "thresholds":[(7,0),(9,1),(14,2),(19,3),(float("inf"),4)]
  },
  "stress": {
    "items":     [1,6,8,11,12,14,18,22,27,29,30,32,33,35,39],
    "checkpoint":"best_stress_severity_roberta.pt",
    "output":    "stress_head_finetuned.pt",
    "thresholds":[(14,0),(18,1),(25,2),(33,3),(float("inf"),4)]
  }
}

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
class SimpleDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts, self.labels = texts, labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        enc = tokenizer(self.texts[i],
                        truncation=True,
                        max_length=MAX_LEN,
                        padding="max_length",
                        return_tensors="pt")
        return {
          "input_ids":      enc["input_ids"].squeeze(0),
          "attention_mask": enc["attention_mask"].squeeze(0),
          "labels":         torch.tensor(self.labels[i],dtype=torch.long)
        }

class HeadOnlyClassifier(nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, 5)
        for p in self.roberta.parameters(): p.requires_grad = False
    def forward(self, ids, mask):
        out = self.roberta(input_ids=ids,attention_mask=mask)
        cls = out.pooler_output or out.last_hidden_state[:,0]
        return self.classifier(self.dropout(cls))

def map_to_sev(score, thresholds):
    for t,l in thresholds:
        if score <= t:
            return l

def fine_tune_head(cfg):
    print(f"--- Head‐only fine‐tuning {cfg['checkpoint']} → {cfg['output']} ---")
    df = pd.read_csv(DATA_PATH, sep="\t", engine="python")
    # build processed_text
    answer_cols = sorted([c for c in df if re.fullmatch(r"Q\\d+A",c)],
                         key=lambda x:int(x[1:-1]))
    df["processed_text"] = df[answer_cols].astype(str).agg(" ".join,axis=1)
    # raw + severity
    raw = df[[f"Q{i}A" for i in cfg["items"]]].sum(axis=1)
    df["label"] = raw.apply(lambda s: map_to_sev(s,cfg["thresholds"]))
    # split
    train, val = train_test_split(df, test_size=0.2,
                                  stratify=df["label"], random_state=42)
    # dataloaders
    train_loader = DataLoader(SimpleDataset(train["processed_text"].tolist(),
                                            train["label"].tolist()),
                              batch_size=BATCH,shuffle=True, num_workers=4)
    val_loader   = DataLoader(SimpleDataset(val["processed_text"].tolist(),
                                            val["label"].tolist()),
                              batch_size=BATCH,shuffle=False,num_workers=4)
    # model & optimizer
    model = HeadOnlyClassifier(DROPOUT).to(DEVICE)
    # load full-checkpoint weights into head-only model
    full_sd = torch.load(cfg["checkpoint"], map_location="cpu")
    head_sd = {k:v for k,v in full_sd.items() if k in model.state_dict()}
    model.load_state_dict({**model.state_dict(), **head_sd})
    opt = AdamW(filter(lambda p:p.requires_grad, model.parameters()),
                lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    # train head
    for epoch in range(EPOCHS):
        model.train()
        for b in train_loader:
            opt.zero_grad()
            logits = model(b["input_ids"].to(DEVICE),
                           b["attention_mask"].to(DEVICE))
            loss = loss_fn(logits, b["labels"].to(DEVICE))
            loss.backward(); opt.step()
        # val
        model.eval()
        correct,total=0,0
        with torch.no_grad():
            for b in val_loader:
                logits = model(b["input_ids"].to(DEVICE),
                               b["attention_mask"].to(DEVICE))
                preds = logits.argmax(dim=1)
                labels= b["labels"].to(DEVICE)
                correct+= (preds==labels).sum().item()
                total+= labels.size(0)
        print(f"Epoch {epoch+1} val_acc: {correct/total:.4f}")
    # save head-only
    torch.save(model.state_dict(), cfg["output"])
    print("Saved", cfg["output"], "\n")

if __name__ == "__main__":
    for cfg in TASKS.values():
        fine_tune_head(cfg)
