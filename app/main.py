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
    site_id: str
    site_name: str
    description: str
    title: str
    content: str

class LabelRequest(BaseModel):
    category: str
    data: List[InputItem]

class LabelResult(BaseModel):
    id: str
    topic_id: str
    site_id: str
    type: str
    label: str
    label_id: str = None
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

# ====================== Label Map ======================


def map_label_to_id(label_name):
    label_mapping = {"Ra mắt sản phẩm mới": "68898a3c16a3634d8333820d","Thiết kế bao bì": "68898a3c16a3634d8333820e","Công nghệ cải tiến": "68898a3c16a3634d8333820f","Chất lượng sản phẩm": "68898a3c16a3634d83338210","Hương vị": "68898a3c16a3634d83338211","Nguồn gốc – Xuất xứ": "68898a3c16a3634d83338212","An toàn vệ sinh": "68898a3c16a3634d83338213","Công dụng": "68898a3c16a3634d83338214","Dị vật": "68898a3c16a3634d83338215","Trải nghiệm sử dụng": "68898a3c16a3634d83338216","Thành phần": "68898a3c16a3634d83338217","Độ đa dạng menu": "68898a3c16a3634d83338218","App/Website": "68898a3c16a3634d83338219","Đồng kiểm": "68898a3c16a3634d8333821a","Dịch vụ smart banking": "68898a3c16a3634d8333821b","Dịch vụ chuyển tiền": "68898a3c16a3634d8333821c","Tài khoản cá nhân": "68898a3c16a3634d8333821d","Tài khoản doanh nghiệp": "68898a3c16a3634d8333821e","Credit cards cá nhân": "68898a3c16a3634d8333821f","Vay tiêu dùng": "68898a3c16a3634d83338220","Tín dụng doanh nghiệp": "68898a3c16a3634d83338221","Tiền gửi cá nhân": "68898a3c16a3634d83338222","Tiền gửi doanh nghiệp": "68898a3c16a3634d83338223","Thẻ ghi nợ": "68898a3c16a3634d83338224","Thẻ tín dụng": "68898a3c16a3634d83338225","Dịch vụ bảo hiểm": "68898a3c16a3634d83338226","Thông tin sản phẩm": "68898a3c16a3634d83338227","Hiệu suất ứng dụng": "68898a3c16a3634d83338228","Thanh toán hóa đơn": "68898a3c16a3634d83338229","Nạp tiền": "68898a3c16a3634d8333822a","Rút tiền": "68898a3c16a3634d8333822b","Thanh toán QR": "68898a3c16a3634d8333822c","Tiết kiệm trực tuyến": "68898a3c16a3634d8333822d","Vay trực tuyến": "68898a3c16a3634d8333822e","Chuyển tiền IBFT": "68898a3c16a3634d8333822f","Bảo mật": "68898a3c16a3634d83338230","Công cụ giao dịch": "68898a3c16a3634d83338231","Nền tảng giao dịch": "68898a3c16a3634d83338232","Chứng chỉ quỹ": "68898a3c16a3634d83338233","Cổ phiếu": "68898a3c16a3634d83338234","Danh mục đầu tư": "68898a3c16a3634d83338235","Khả năng sinh lời": "68898a3c16a3634d83338236","Nguồn cung đơn hàng": "68898a3c16a3634d83338237","Hệ thống dẫn đường": "68898a3c16a3634d83338238","Cơ sở vật chất": "68898a3c16a3634d83338239","Quy trình đổi trả": "68898a3c16a3634d8333823a","Đổi trả sản phẩm": "68898a3c16a3634d8333823b","Số lượng đơn hàng": "68898a3c16a3634d8333823c","Thời gian giao hàng": "68898a3c16a3634d8333823d","Không/giao trễ": "68898a3c16a3634d8333823e","Thiết kế": "68898a3c16a3634d8333823f","Tính năng": "68898a3c16a3634d83338240","Chi phí vận chuyển": "68898a3c16a3634d83338241","Chuyên khoa y tế": "68898a3c16a3634d83338242","Dịch vụ tư vấn": "68898a3c16a3634d83338243","Thủ tục hành chính": "68898a3c16a3634d83338244","Nghiên cứu & phát triển": "68898a3c16a3634d83338245","Thanh toán thẻ": "68898a3c16a3634d83338246","Dịch vụ khách hàng": "68898a3c16a3634d83338247","Bảo hành sản phẩm": "68898a3c16a3634d83338248","Nâng cấp sản phẩm": "68898a3c16a3634d83338249","Sửa chữa sản phẩm": "68898a3c16a3634d8333824a","Phụ kiện": "68898a3c16a3634d8333824b","Tùy chỉnh sản phẩm": "68898a3c16a3634d8333824c","Dịch vụ thuê bao": "68898a3c16a3634d8333824d","Hoán đổi sản phẩm": "68898a3c16a3634d8333824e","Hoàn tiền": "68898a3c16a3634d8333824f","Lắp đặt sản phẩm": "68898a3c16a3634d83338250","Bảo trì": "68898a3c16a3634d83338251","Thân thiện môi trường": "68898a3c16a3634d83338252","Chiến dịch": "68898a3c16a3634d83338253","Chương trình khuyến mãi": "68898a3c16a3634d83338254","KM Eshop/Ecommerce": "68898a3c16a3634d83338255","Sự kiện": "68898a3c16a3634d83338256","Hoạt động trên Fanpage": "68898a3c16a3634d83338257","Voucher": "68898a3c16a3634d83338258","Minigame": "68898a3c16a3634d83338259","Livestream": "68898a3c16a3634d8333825a","Bài đăng tương tác": "68898a3c16a3634d8333825b","Thông cáo báo chí": "68898a3c16a3634d8333825c","Hoạt động cộng đồng": "68898a3c16a3634d8333825d","Hoạt động truyền thông": "68898a3c16a3634d8333825e","Chương trình ưu đãi": "68898a3c16a3634d8333825f","Hợp tác quảng bá": "68898a3c16a3634d83338260","Nhận diện thương hiệu": "68898a3c16a3634d83338261","Chương trình khách hàng trung thành": "68898a3c16a3634d83338262","Tiếp thị liên kết": "68898a3c16a3634d83338263","Quảng cáo": "68898a3c16a3634d83338264","Hội thảo trực tuyến": "68898a3c16a3634d83338265","Thảo luận giá cả": "68898a3c16a3634d83338266","So sánh giá": "68898a3c16a3634d83338267","Chính sách giảm giá": "68898a3c16a3634d83338268","Rao vặt": "68898a3c16a3634d83338269","Tỉ giá tiền tệ": "68898a3c16a3634d8333826a","Lãi suất cho vay": "68898a3c16a3634d8333826b","Lãi suất tiền gửi": "68898a3c16a3634d8333826c","Lãi suất/hồ sơ": "68898a3c16a3634d8333826d","Phí/thu phí": "68898a3c16a3634d8333826e","Lãi suất": "68898a3c16a3634d8333826f","Quy trình/thủ tục": "68898a3c16a3634d83338270","Phí giao dịch": "68898a3c16a3634d83338271","Bảng giá/điện": "68898a3c16a3634d83338272","Thuế": "68898a3c16a3634d83338273","Tình trạng hàng hóa": "68898a3c16a3634d83338274","Thanh toán trả góp": "68898a3c16a3634d83338275","Trải nghiệm khách hàng": "68898a3c16a3634d83338276","Chăm sóc khách hàng": "68898a3c16a3634d83338277","Đăng ký mẫu thử": "68898a3c16a3634d83338278","Tư vấn trực tuyến": "68898a3c16a3634d83338279","Phục vụ khách hàng": "68898a3c16a3634d8333827a","Dịch vụ call center": "68898a3c16a3634d8333827b","Quấy rối khách hàng": "68898a3c16a3634d8333827c","Phản hồi/đánh giá": "68898a3c16a3634d8333827d","Độ hài lòng khách hàng": "68898a3c16a3634d8333827e","Khiếu nại khách hàng": "68898a3c16a3634d8333827f","Đánh giá sản phẩm": "68898a3c16a3634d83338280","Trung thành khách hàng": "68898a3c16a3634d83338281","Giới thiệu khách hàng": "68898a3c16a3634d83338282","Hỗ trợ qua chat": "68898a3c16a3634d83338283","Hỗ trợ qua email": "68898a3c16a3634d83338284","Khảo sát ý kiến": "68898a3c16a3634d83338285","Hiệu suất tài chính": "68898a3c16a3634d83338286","Đầu tư tài chính": "68898a3c16a3634d83338287","Lợi nhuận doanh nghiệp": "68898a3c16a3634d83338288","Rủi ro tài chính": "68898a3c16a3634d83338289","Chứng khoán": "68898a3c16a3634d8333828a","Hình ảnh thương hiệu": "68898a3c16a3634d8333828b","Ban lãnh đạo": "68898a3c16a3634d8333828c","Chi nhánh/liên doanh": "68898a3c16a3634d8333828d","Đại hội cổ đông": "68898a3c16a3634d8333828e","Giải thưởng công ty": "68898a3c16a3634d8333828f","Hoạt động kinh doanh": "68898a3c16a3634d83338290","Quan hệ nhà đầu tư": "68898a3c16a3634d83338291","Định giá/đầu tư": "68898a3c16a3634d83338292","M&A/tái cấu trúc": "68898a3c16a3634d83338293","Hoạt động hợp tác": "68898a3c16a3634d83338294","Mở rộng kinh doanh": "68898a3c16a3634d83338295","Cổ tức": "68898a3c16a3634d83338296","Công ty con": "68898a3c16a3634d83338297","Chương trình CSR": "68898a3c16a3634d83338298","Tiết kiệm năng lượng": "68898a3c16a3634d83338299","Bảo vệ môi trường": "68898a3c16a3634d8333829a","Hỗ trợ cộng đồng": "68898a3c16a3634d8333829b","ESG bền vững": "68898a3c16a3634d8333829c","Quản lý chất thải": "68898a3c16a3634d8333829d","Tiết kiệm nước": "68898a3c16a3634d8333829e","Năng lượng tái tạo": "68898a3c16a3634d8333829f","Hoạt động từ thiện": "68898a3c16a3634d833382a0","Vấn đề an toàn": "68898a3c16a3634d833382a1","Tai tiếng công ty": "68898a3c16a3634d833382a2","Thu hồi sản phẩm": "68898a3c16a3634d833382a3","Khiếu nại lớn": "68898a3c16a3634d833382a4","Phản hồi khủng hoảng": "68898a3c16a3634d833382a5","Tẩy chay thương hiệu": "68898a3c16a3634d833382a6","Rủi ro/gian lận": "68898a3c16a3634d833382a7","Tranh tụng pháp lý": "68898a3c16a3634d833382a8","Văn hóa công ty": "68898a3c16a3634d833382a9","Tuyển dụng": "68898a3c16a3634d833382aa","Phúc lợi nhân viên": "68898a3c16a3634d833382ab","Hiệu suất tài xế": "68898a3c16a3634d833382ac","Hoạt động nội bộ": "68898a3c16a3634d833382ad","Đối tác tài xế": "68898a3c16a3634d833382ae","Đào tạo nhân viên": "68898a3c16a3634d833382af","Lương nhân viên": "68898a3c16a3634d833382b0","Chế độ phúc lợi": "68898a3c16a3634d833382b1","Giữ chân nhân viên": "68898a3c16a3634d833382b2","Đánh giá hiệu suất": "68898a3c16a3634d833382b3","Chính sách pháp lý": "68898a3c16a3634d833382b4","Chính sách thuế": "68898a3c16a3634d833382b5","Cạnh tranh ngành": "68898a3c16a3634d833382b6","Hợp tác/đối tác": "68898a3c16a3634d833382b7","Hoạt động kinh doanh": "68898a3c16a3634d833382b8","So sánh thương hiệu": "68898a3c16a3634d833382b9","Chương trình học bổng": "68898a3c16a3634d833382ba","Điều khoản chính sách": "68898a3c16a3634d833382bb","Phân tích thị trường": "68898a3c16a3634d833382bc","Thay đổi quy định": "68898a3c16a3634d833382bd","Chính sách thương mại": "68898a3c16a3634d833382be","Chính sách môi trường": "68898a3c16a3634d833382bf"
    }
    return label_mapping.get(label_name, "Label không tồn tại")


# ====================== API Endpoint ======================

@app.post("/api/label-inference", response_model=LabelResponse)
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
        site_name = row["site_name"]
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
            site_id=row["site_id"],
            type=row["type"],
            ref_label_map=best_label,
            label=best_label[0] if best_label else None,
            label_id=map_label_to_id(best_label[0]) if best_label else None,
            ref_llm_label=full_labels,
            process_time=duration
        ))

    return LabelResponse(results=results)
