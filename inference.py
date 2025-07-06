import torch
from transformers import RobertaTokenizerFast, RobertaModel
import torch.nn as nn
import argparse

# ────────────────────────────────────────────────────────────────────────
# DASSTaskClassifier definition
# ────────────────────────────────────────────────────────────────────────
class DASSTaskClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout_rate: float):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            cls_repr = outputs.pooler_output
        else:
            cls_repr = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_repr)
        return self.classifier(x)

# ────────────────────────────────────────────────────────────────────────
# Main inference script
# ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Inference demo for DASS-42 severity classification"
    )
    parser.add_argument(
        "--task", type=str, required=True, choices=["depression", "anxiety", "stress"],
        help="Which subscale to run inference on"
    )
    parser.add_argument(
        "--text", type=str, required=True,
        help="42 space-separated numeric answers (each in {0,1,2,3})"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the .pt checkpoint file for the chosen task"
    )
    args = parser.parse_args()

    # Configuration
    MODEL_NAME = "roberta-base"
    NUM_LABELS = 5
    DROP_RATE = 0.1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer, model, and checkpoint
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
    model = DASSTaskClassifier(MODEL_NAME, num_labels=NUM_LABELS, dropout_rate=DROP_RATE)
    state_dict = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # Prepare input
    encoding = tokenizer(
        args.text,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    # Forward pass
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label = int(torch.argmax(logits, dim=1).cpu().numpy()[0])

    # Map label to severity
    severity_map = {
        0: "Normal",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Extremely Severe"
    }

    print(f"Task: {args.task.capitalize()}")
    print(f"Predicted severity label: {pred_label} ({severity_map[pred_label]})")
    print(f"Probability distribution: {probs}")

if __name__ == "__main__":
    main()
