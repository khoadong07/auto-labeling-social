import os
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from summa.summarizer import summarize
from dotenv import load_dotenv

load_dotenv()

# === Langfuse tracking ===
langfuse_handler = CallbackHandler()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)


# === Hàm tóm tắt nội dung nếu dài hơn 100 từ (không dùng LLM) ===
def summarize_text_locally(text: str, word_limit: int = 50) -> str:
    summary = summarize(text, words=word_limit, language='english')
    if not summary:
        summary = '. '.join(text.split('. ')[:2])
    return summary

# === Chuẩn hóa nội dung đầu vào: cắt 100 từ hoặc tóm tắt nếu dài ===
def prepare_text(text: str) -> str:
    words = re.findall(r'\w+|\S', text)
    if len(words) > 100:
        print("📌 Nội dung dài > 100 từ, đang tóm tắt bằng TextRank...")
        return summarize_text_locally(text)
    return ' '.join(words[:100])

# === LLM từ DeepInfra ===
llm = ChatOpenAI(
    model="accounts/fireworks/models/llama4-scout-instruct-basic",
    temperature=0.4,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
)

# === Parser JSON chuẩn ===
parser = JsonOutputParser()

# === Prompt tổng quát theo ngành (KHÔNG chứa nguồn) ===
prompt = ChatPromptTemplate.from_template(
    """
Bạn là một chuyên gia phân tích dữ liệu mạng xã hội trong ngành "{domain}".

Nhiệm vụ của bạn:
1. Phân tích nội dung dưới đây và suy luận tối đa 3 nhãn thể hiện chủ đề chính (labels) bằng tiếng Việt.
2. Đánh giá độ tin cậy từ 0 đến 1 (confidence).

Trả về đúng định dạng JSON sau:
{{
  "labels": ["...", "..."],
  "confidence": ...
}}

Nội dung: "{text}"
"""
)

# === Gộp thành một Agentic Chain chuẩn hóa ===
label_chain = (
    {
        "text": lambda x: prepare_text(x["text"]),
        "domain": lambda x: x["domain"],
    }
    | prompt
    | llm
    | parser
)

# === Hàm gán nhãn chính ===
def label_social_post(text: str, domain: str) -> dict:
    """
    Gán nhãn cho bài viết social listening theo ngành.

    Args:
        text (str): Nội dung bài viết.
        domain (str): Ngành (banking, healthcare, ecommerce,...)

    Returns:
        dict: labels, confidence
    """
    return label_chain.invoke({
        "text": text,
        "domain": domain,
    }, config={"callbacks": [langfuse_handler]})