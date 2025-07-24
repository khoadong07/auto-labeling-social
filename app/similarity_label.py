import os
import time
from pinecone import Pinecone, ServerlessSpec
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("PINECONE")
pc = Pinecone(api_key=API_KEY)
index_name = "semantic-label-search"
index = pc.Index(index_name)

model_name = "AITeamVN/Vietnamese_Embedding"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModel.from_pretrained(model_name)

def get_embedding(text: str) -> list[float]:
    """Sinh embedding rồi L2‑normalize, trả về list float."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        out = model(**inputs).last_hidden_state.mean(dim=1).squeeze()
    out = F.normalize(out, p=2, dim=0)
    return out.cpu().tolist()

def search_label_pinecone(query: str, top_k: int = 3) -> list[str]:
    vec  = get_embedding(query)
    resp = index.query(vector=vec, top_k=top_k, include_metadata=True)
    return [m["metadata"]["label"] for m in resp["matches"]]

def get_best_label_from_content(
    content: str,
    labels_input: list[str],
    top_k: int = 3
) -> list[str]:

    priority_map = {
        'tuyển dụng': 'Tuyển dụng',
        'livestream': 'Livestream',
        'minigame': 'Minigame'
    }
    for lab in labels_input:
        if lab.lower() in priority_map:
            return [priority_map[lab.lower()]]

    for lab in labels_input:
        if lab in content:
            res = search_label_pinecone(lab, top_k)
            if res:
                return res

    for lab in labels_input:
        res = search_label_pinecone(lab, top_k)
        if res:
            return res

    return []
