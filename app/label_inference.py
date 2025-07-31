import os
import re

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from summa.summarizer import summarize

from ads_predict import predict_ads

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
prompt = ChatPromptTemplate.from_template("""
Bạn là chuyên gia phân tích dữ liệu mạng xã hội.

Yêu cầu:
1. Dịch "{topic_name}" sang tiếng Việt.
2. Phân tích nội dung bên dưới và trích tối đa 3 nhãn (labels) bằng tiếng Việt, phản ánh đúng chủ đề đã dịch.
3. Chỉ chọn nhãn thực sự liên quan đến nội dung.
4. Loại bỏ nhãn nếu:
   - Trùng hoặc gần nghĩa với chủ đề đã dịch;
   - Chứa từ khóa liên quan ngành "{domain}" (bằng tiếng Việt, tiếng Anh, viết tắt hoặc viết hoa/thường);
   - Là tên riêng (công ty, cá nhân, tổ chức, địa danh).
5. Gán độ tin cậy (confidence) từ 0 đến 1.

Chỉ trả về đúng định dạng JSON:
{{
  "labels": ["...", "..."],
  "confidence": ...
}}

Nội dung: "{text}"
""")

# === Gộp thành một Agentic Chain chuẩn hóa ===
label_chain = (
        {
            "text": lambda x: prepare_text(x["text"]),
            "domain": lambda x: x["domain"],
            "topic_name": lambda x: x["topic_name"],
        }
        | prompt
        | llm
        | parser
)


def label_social_post(text: str, category: str, type: str, site_name: str, topic_name: str) -> dict:
    text_lower = text.lower()
    # check ads service
    ads_predict = predict_ads(text)
    if ads_predict and type not in ('newsTopic', 'fbPageTopic'):
        return {
            "labels": ["Rao vặt"],
            "confidence": 1.0
        }
    if type != 'newsTopic':
        if any(keyword in text_lower for keyword in ["minigame", "mini game", "mini-game"]):
            return {
                "labels": ["Minigame"],
                "confidence": 1.0
            }

        if any(keyword in text_lower for keyword in ["tuyển dụng", "tuyển nhân sự", "tuyển ctv"]):
            return {
                "labels": ["Tuyển dụng"],
                "confidence": 1.0
            }

        if any(keyword in text_lower for keyword in ["livestream", "live stream"]):
            return {
                "labels": ["Livestream"],
                "confidence": 1.0
            }

    if category in ['FMCG', 'Retail', 'Banking', 'Digital Payments', 'Insurance',
                    'Investment Services', 'Real Estate Development',
                    'Energy & Utilities', 'Software & IT Services',
                    'Telecommunications & Internet', 'Electronic Products',
                    'Food & Beverage', 'Home & Living', 'Hospitality & Leisure',
                    'Conglomerates', 'Automotive']:

        if site_name == 'fireant.vn' or any(
                keyword in text_lower for keyword in ["chứng khoán", "index", "in-dex", "vn30", "vnindex"]):
            return {
                "labels": ["Chứng khoán"],
                "confidence": 1.0
            }
    try:
        label_inf = label_chain.invoke(
            {
                "text": text,
                "domain": category,
                "topic_name": topic_name
            },
            config={"callbacks": [langfuse_handler]},
        )
        if label_inf is not None:
            label = label_inf.get("labels")
            if len(label) > 0:
                return label_inf
            else:
                return {"labels": ["Đề cập chung"], "confidence": 1.0}
    except OutputParserException as e:
        print("⚠️ LLM trả về sai định dạng JSON:", e)
        return {"labels": ["Đề cập chung"], "confidence": 1.0}
    except Exception as e:
        print("❌ Lỗi không xác định:", e)

    return {
        "labels": [],
        "confidence": 0.0
    }
