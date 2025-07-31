import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vncorenlp import VnCoreNLP

# ---------- Load once at startup ----------
# Khởi tạo VnCoreNLP cho tách từ
vncorenlp = VnCoreNLP("/app/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

# Load tokenizer và model từ Hugging Face
tokenizer = AutoTokenizer.from_pretrained("Khoa/kompa-check-ads-0725", use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained("Khoa/kompa-check-ads-0725")

device = 0 if torch.cuda.is_available() else -1
text_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)


def preprocess_text(text: str) -> str:
    text = text.lower()
    sentences = vncorenlp.tokenize(text)
    return ' '.join([' '.join(sen) for sen in sentences])


def predict_ads(text: str) -> bool:
    if not text or not text.strip():
        raise ValueError("Input text must not be empty.")

    processed_text = preprocess_text(text)
    result = text_classifier(processed_text, truncation=True, max_length=100)[0]

    label_id = int(result['label'].split('_')[-1]) if "label" in result['label'].lower() else 0

    is_ads = True if label_id == 1 else False
    return is_ads


