# 🎯 Company Similarity and Recommendation  
### 📚 Đồ án tốt nghiệp môn *Data Science and Machine Learning*

---

## 📝 Giới thiệu

Đây là một ứng dụng Web được xây dựng bằng **Streamlit**, hỗ trợ **ứng viên tìm kiếm công ty phù hợp** dựa trên:

- Sự **tương đồng giữa các công ty** (Company Similarity)
- **Đề xuất công ty** phù hợp theo tiêu chí cá nhân (Recommendation)

Ứng dụng sử dụng các mô hình học máy và các thư viện xử lý dữ liệu hiện đại để phân tích thông tin công ty và nhu cầu người dùng.

---

## 🎯 Mục tiêu

- Giúp ứng viên tìm được các công ty tương tự với một công ty đã biết hoặc theo từ khóa.
- Gợi ý các công ty phù hợp dựa trên nguyện vọng: loại công ty, ngành nghề, quy mô, quốc gia, ngày làm việc, chính sách OT,...

---

## 🚀 Công nghệ sử dụng

### 🔎 Company Similarity
- **Mô hình:** Gensim (Word2Vec) và Cosine Similarity  
- **Thư viện:** `gensim`, `sklearn`, `pandas`

### 💡 Recommendation
- **Mô hình phân loại:** scikit-learn (Logistic Regression, RandomForest,...) và **PySpark MLlib**  
- **Xử lý dữ liệu:** `pyspark`, `pandas`

---

## 📁 Cấu trúc dữ liệu và thư mục

- `data/Overview_Companies.xlsx` – Dữ liệu tổng quan về công ty  
- `data/translated_data.csv` – Dữ liệu công ty đã được dịch  
- `data/sentiment_by_company.csv` – Dữ liệu phân tích cảm xúc  
- `data/top2_clusters_per_company.csv` – Dữ liệu phân cụm  
- `files/` – Chứa các file `.txt` phục vụ xử lý/nghiên cứu  
- `models/` – Các mô hình đã huấn luyện  
- `images/` – Hình ảnh minh họa giao diện

---

## 🛠 Cài đặt và chạy ứng dụng

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng
```bash
streamlit run class.py
```


---

## 👨‍💻 Tác giả

- **Võ Minh Trí** – *Phụ trách phần Recommendation*  
  📧 trivm203@gmail.com  

- **Phạm Thị Thu Thảo** – *Phụ trách phần Company Similarity*  
  📧 thaofpham@gmail.com

---

## 📄 Giấy phép

Project phục vụ mục đích học tập – không sử dụng cho mục đích thương mại.
🔥