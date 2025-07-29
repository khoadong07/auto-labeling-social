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
index_name = "semantic-label"
index = pc.Index(index_name)

model_name = "AITeamVN/Vietnamese_Embedding"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModel.from_pretrained(model_name)

def get_embedding(text: str) -> list[float]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**inputs).last_hidden_state.mean(dim=1).squeeze()
    return F.normalize(output, p=2, dim=0).cpu().tolist()

def semantic_label_search(query_text: str, category: str, top_k: int = 5):
    query_vec = get_embedding(query_text)

    # Truy vấn Pinecone với bộ lọc category
    response = index.query(
        vector=query_vec,
        top_k=top_k,
        filter={"category": category},
        include_metadata=True
    )

    # Format lại kết quả
    results = []
    for match in response.get('matches', []):
        metadata = match.get('metadata', {})
        results.append(metadata.get("label"))

    return results

# def search_label_pinecone(query: str, top_k: int = 3) -> list[str]:
#     vec  = get_embedding(query)
#     resp = index.query(vector=vec, top_k=top_k, include_metadata=True)
#     return [m["metadata"]["label"] for m in resp["matches"]]

def get_best_label_from_content(
    content: str,
    category: str,
    labels_input: list[str],
    top_k: int = 3
) -> list[str]:

    priority_map = {
        'tuyển dụng': 'Tuyển dụng',
        'livestream': 'Livestream',
        'minigame': 'Minigame',
        'chứng khoán': 'Chứng khoán'
    }
    for lab in labels_input:
        if lab.lower() in priority_map:
            return [priority_map[lab.lower()]]

    for lab in labels_input:
        if lab in content:
            res = semantic_label_search(query_text=lab, category=category, top_k = top_k)
            if res:
                return res

    for lab in labels_input:
        res = semantic_label_search(query_text=lab, category=category, top_k = top_k)
        if res:
            return res

    return []
