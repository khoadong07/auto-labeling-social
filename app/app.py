from similarity_label import get_best_label_from_content
import streamlit as st
import pandas as pd
import hashlib
from typing import Dict, List, Tuple
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from label_inference import label_social_post
from stqdm import stqdm

# ========================== Utilities ==========================
CATEGORIES = [
    'FMCG', 'Retail', 'Education Services', 'Banking',
    'Digital Payments', 'Insurance', 'Financial Services',
    'Investment Services', 'Real Estate Development', 'Healthcare',
    'Energy & Utilities', 'Software & IT Services',
    'Ride-Hailing & Delivery', 'Logistics & Delivery',
    'Telecommunications & Internet', 'Electronic Products',
    'Food & Beverage', 'Home & Living', 'Hospitality & Leisure',
    'Conglomerates', 'Beauty & Personal Care', 'Automotive',
    'Entertainment & Media', 'Industrial Parks & Zones',
    'Mobile Applications', 'E-commerce'
]

def get_text_signature(row) -> str:
    combined_text = f"{row['Title']} {row['Content']} {row['Description']}".strip().lower()
    return hashlib.md5(combined_text.encode('utf-8')).hexdigest()

def ensure_list_or_none(value):
    if isinstance(value, list):
        return value
    elif pd.isna(value):
        return None
    elif isinstance(value, str):
        try:
            val = ast.literal_eval(value)
            if isinstance(val, list):
                return val
        except:
            pass
    return None

def merge_text(row) -> str:
    parts = [str(row.get(col, "")).strip() for col in ['Title', 'Content', 'Description']]
    return " ".join(part for part in parts if part)

def parallel_labeling(dedup_df: pd.DataFrame, category: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    label_mapping = {}   
    
    all_labels = {}      # text_signature -> full list of labels

    def worker(row):
        text = row['merged_text']
        type = row['Type']
        site_name = row['SiteName']
        topic_name = row['Topic']
        result = label_social_post(text=text, category=category, type=type, site_name=site_name, topic_name=topic_name)
        labels = result.get("labels", [])
        if not labels:
            return row['text_signature'], "", []
        
        best_label = get_best_label_from_content(text, category=category, labels_input=labels)
        return row['text_signature'], best_label, labels

    st.info("🔄 Running parallel labeling on unique posts...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(worker, row): row for _, row in dedup_df.iterrows()}
        for future in stqdm(as_completed(futures), total=len(futures)):
            signature, best_label, full_labels = future.result()
            label_mapping[signature] = best_label
            all_labels[signature] = full_labels

    return label_mapping, all_labels


def process_file(df: pd.DataFrame, category: str) -> pd.DataFrame:
    df[['Title', 'Content', 'Description']] = df[['Title', 'Content', 'Description']].fillna("")
    df['text_signature'] = df.apply(get_text_signature, axis=1)
    df['merged_text'] = df.apply(merge_text, axis=1)

    dedup_df = df.drop_duplicates(subset=['text_signature'])
    label_mapping, all_labels = parallel_labeling(dedup_df, category)

    df['Labels_Mapping'] = df['text_signature'].map(label_mapping).apply(ensure_list_or_none)
    df['Labels'] = df['text_signature'].map(all_labels).apply(lambda x: ", ".join(x) if isinstance(x, list) else "")

    return df.drop(columns=["merged_text", "text_signature"])


# ========================== Streamlit UI ==========================

st.set_page_config(page_title="Social Listening Auto Labeling", layout="wide")
st.title("🎯 Social Listening Auto Labeling (Agentic AI + LangChain)")

st.markdown("""
Easily upload a dataset and auto-label social posts using an AI-powered workflow.
""")

# Layout: 2 columns
col1, col2 = st.columns([1, 2])

with col1:
    st.header("🔧 Input Settings")

    category = st.selectbox("📌 Chọn ngành (Category):", options=CATEGORIES)

    uploaded_file = st.file_uploader("📁 Upload Excel (.xlsx)", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        required_cols = {"Title", "Content", "Description"}

        if not required_cols.issubset(df.columns):
            st.error(f"❌ Missing required columns: {required_cols}")
        else:
            max_rows = len(df)
            num_rows = st.slider("🔢 Number of rows to process", min_value=1, max_value=max_rows, value=min(100, max_rows))

            if category and st.button("🚀 Start Labeling"):
                df_subset = df.head(num_rows)

                with st.spinner("⚙️ Processing... please wait."):
                    processed_df = process_file(df_subset, category)

                st.session_state["processed_df"] = processed_df
                st.success("✅ Labeling complete!")

with col2:
    st.header("📊 Results & Preview")

    if "processed_df" in st.session_state:
        processed_df = st.session_state["processed_df"]

        with st.expander("🔍 Preview Labeled Data", expanded=True):
            st.dataframe(processed_df[['Title', 'Content', 'Description', 'Type', 'SiteName', 'Topic', 'Labels', 'Labels_Mapping']], 
                         use_container_width=True, height=400)

        # Optional: Display label statistics
        with st.expander("📈 Label Distribution"):
            label_counts = processed_df["Labels"].value_counts().reset_index()
            label_counts.columns = ["Label", "Count"]
            st.dataframe(label_counts, use_container_width=True)

        # Export button
        output = BytesIO()
        processed_df.to_excel(output, index=False)
        output.seek(0)

        st.download_button("📥 Download Labeled Excel", output, file_name="labeled_result.xlsx", use_container_width=True)
    else:
        st.info("📝 No data processed yet. Please upload a file and start labeling.")
