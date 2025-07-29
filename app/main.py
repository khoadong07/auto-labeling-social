from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from label_inference import label_social_post
from similarity_label import get_best_label_from_content
import hashlib
import pandas as pd
import time

app = FastAPI(title="Social Listening Labeling API")

# ====================== Request/Response Models ======================
class InputItem(BaseModel):
    id: str
    topic_name: str
    type: str
    topic_id: str
    siteId: str
    siteName: str
    description: str
    title: str
    content: str

class LabelRequest(BaseModel):
    category: str
    data: List[InputItem]

class LabelResult(BaseModel):
    id: str
    topic_id: str
    siteId: str
    type: str
    label: str
    ref_label_map: List[str]
    ref_llm_label: List[str]
    process_time: float

class LabelResponse(BaseModel):
    results: List[LabelResult]

# ====================== Utilities ======================

def get_text_signature(title: str, content: str, description: str) -> str:
    combined_text = f"{title} {content} {description}".strip().lower()
    return hashlib.md5(combined_text.encode('utf-8')).hexdigest()

def merge_text(title: str, content: str, description: str) -> str:
    parts = [title.strip(), content.strip(), description.strip()]
    return " ".join(p for p in parts if p)

# ====================== API Endpoint ======================

@app.post("/label", response_model=LabelResponse)
def label_posts(request: LabelRequest):
    start_time = time.time()
    category = request.category
    data = request.data

    # Convert input to DataFrame
    records = [item.dict() for item in data]
    df = pd.DataFrame(records)

    # Prepare merged text and signature
    df["merged_text"] = df.apply(lambda row: merge_text(row["title"], row["content"], row["description"]), axis=1)
    df["text_signature"] = df.apply(lambda row: get_text_signature(row["title"], row["content"], row["description"]), axis=1)

    # Deduplication
    dedup_df = df.drop_duplicates(subset=["text_signature"])

    # Inference
    label_mapping = {}
    all_labels = {}

    for _, row in dedup_df.iterrows():
        text = row["merged_text"]
        post_type = row["type"]
        site_name = row["siteName"]
        result = label_social_post(text=text, category=category, type=post_type, site_name=site_name)
        labels = result.get("labels", [])
        best_label = get_best_label_from_content(text, category=category, labels_input=labels) if labels else ""
        label_mapping[row["text_signature"]] = best_label
        all_labels[row["text_signature"]] = labels

    # Construct result
    results = []
    for _, row in df.iterrows():
        sig = row["text_signature"]
        best_label = label_mapping.get(sig, "")
        full_labels = all_labels.get(sig, [])
        duration = time.time() - start_time

        results.append(LabelResult(
            id=row["id"],
            topic_id=row["topic_id"],
            siteId=row["siteId"],
            type=row["type"],
            ref_label_map=best_label,
            label=best_label[0] if best_label else None,
            ref_llm_label=full_labels,
            process_time=duration
        ))

    return LabelResponse(results=results)
