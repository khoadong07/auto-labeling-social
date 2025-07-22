from label_inference import label_social_post

if __name__ == "__main__":
    example = {
        "text": """
        Ngân hàng Techcombank vừa công bố ra mắt dịch vụ tiết kiệm online với lãi suất ưu đãi hơn so với gửi tiết kiệm trực tiếp tại quầy. 
        Khách hàng có thể mở sổ tiết kiệm qua ứng dụng mobile banking chỉ với vài thao tác đơn giản. 
        Ngoài ra, chương trình còn áp dụng thêm ưu đãi hoàn tiền và miễn phí chuyển khoản cho khách hàng đăng ký trong tháng 7.
        """,
        "domain": "banking"
    }

    result = label_social_post(**example)
    print("\n✅ Kết quả phân tích và gán nhãn:")
    print(result)