import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import string
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from gensim import corpora, models, similarities

# Cấu hình trang
st.set_page_config(
    page_title="AI Company Recommendation System AT IT_Viec",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .company-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .cluster-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
    }
    .sentiment-positive {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .sentiment-neutral {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .prediction-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 1️⃣ Khởi tạo session_state cho các biến đơn giản
default_state = {
    "data_loaded": False,
    "model_trained": False,
    "selectbox_company": "-- Chọn công ty --",
    "query_text": "",
    "prev_selectbox_company": "-- Chọn công ty --",
    "prev_query_text": ""
}
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Class xử lý văn bản tích hợp từ code của bạn
class TextProcessor:
    def __init__(self):
        self.teen_dict = {}
        self.stopwords_lst = []
        self.wrong_lst = []
        self.english_dict = {}
        self.load_dictionaries()
    
    def load_dictionaries(self):
        """Load các từ điển xử lý văn bản từ files hoặc fallback"""
        try:
            # Load teencode từ file hoặc fallback
            teencode_paths = ['files/teencode.txt', 'teencode.txt']
            for path in teencode_paths:
                if os.path.exists(path):
                    with open(path, 'r', encoding="utf8") as file:
                        teen_lst = file.read().split('\n')
                        for line in teen_lst:
                            if '\t' in line:
                                key, value = line.split('\t', 1)
                                self.teen_dict[key] = str(value)
                    break
            else:
                # Fallback teencode dictionary
                self.teen_dict = {
                    'ko': 'không', 'k': 'không', 'dc': 'được', 'vs': 'với',
                    'tks': 'thanks', 'ty': 'thank you', 'ok': 'okay', 'oke': 'okay',
                    'bt': 'bình thường', 'nc': 'nói chuyện', 'kb': 'kết bạn'
                }
            
            # Load stopwords từ file hoặc fallback
            stopwords_paths = ['files/vietnamese-stopwords_rev.txt', 'vietnamese-stopwords_rev.txt']
            for path in stopwords_paths:
                if os.path.exists(path):
                    with open(path, 'r', encoding="utf8") as file:
                        self.stopwords_lst = file.read().split('\n')
                    break
            else:
                # Fallback stopwords
                self.stopwords_lst = [
                    'và', 'của', 'có', 'là', 'được', 'trong', 'với', 'để', 'cho', 'từ',
                    'một', 'các', 'này', 'đó', 'những', 'nhiều', 'rất', 'cũng', 'sẽ',
                    'đã', 'đang', 'về', 'theo', 'như', 'khi', 'nếu', 'vì', 'do', 'bởi'
                ]
            
            # Load wrong words từ file hoặc fallback
            wrong_words_paths = ['files/wrong-word_rev.txt', 'wrong-word_rev.txt']
            for path in wrong_words_paths:
                if os.path.exists(path):
                    with open(path, 'r', encoding="utf8") as file:
                        self.wrong_lst = file.read().split('\n')
                    break
            else:
                self.wrong_lst = ['zzz', 'xxx', 'aaa', 'bbb', 'ccc']

            # Load english-vnmese dictionary từ file hoặc fallback nếu không tồn tại
            english_dict_paths = ['files/english-vnmese_rev.txt', 'english-vnmese_rev.txt']
            for path in english_dict_paths:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf8') as file:
                        for line in file:
                            if '\t' in line:
                                key, value = line.strip().split('\t', 1)
                                self.english_dict[key] = value
                    break
            else:
                # Fallback từ điển đơn giản nếu không có file
                self.english_dict = {
                    "hello": "xin chào",
                    "world": "thế giới",
                    "example": "ví dụ",
                    "test": "kiểm tra"}
                    
        except Exception as e:
            st.warning(f"⚠️ Không thể load một số file từ điển: {e}")
    
    def clean_text(self, text):
        """Làm sạch văn bản cơ bản - tích hợp từ code của bạn"""
        if not text or pd.isna(text):
            return ""
        text = str(text).lower()  # Hoa -> thường
        text = re.sub(rf"[{string.punctuation}]", "", text)  # Bỏ dấu câu
        text = re.sub(r"\b(g|ml)\b", "", text)  # Bỏ từ 'g' hoặc 'ml' khi nó là cả từ
        text = re.sub(r"\s+", " ", text).strip()  # Xóa khoảng trắng thừa
        text = re.sub(r"^[\-\+\*\•\●\·\~\–\—\>]+", "", text)  # Bỏ dấu đầu câu
        return text
    
    def translate_english(self,text):
        """Dịch từ tiếng Anh sang tiếng Việt nếu phát hiện văn bản là tiếng Anh"""
        def is_english(text):
            if not text:
                return False
            text = text.lower()
            eng_chars = re.findall(r"[a-z]", text)
            non_eng_chars = re.findall(r"[à-ỹ]", text.lower())
            return len(eng_chars) > len(non_eng_chars)

        if not is_english(text):
            return text  # Nếu không phải tiếng Anh thì giữ nguyên
        words = text.split()
        translated = [self.english_dict.get(word, word) for word in words]
        return " ".join(translated)
    
    def fix_teencode(self, text):
        """Sửa teencode - tích hợp từ code của bạn"""
        words = text.split()
        corrected = [self.teen_dict.get(word, word) for word in words]
        return " ".join(corrected)
    
    def remove_wrongword(self, text):
        """Loại bỏ từ sai"""
        words = text.split()
        trueword = [word for word in words if word not in self.wrong_lst]
        return " ".join(trueword)
    
    def remove_stopword(self, text):
        """Loại bỏ stopwords - tích hợp từ code của bạn"""
        words = text.split()
        stopword = [word for word in words if word not in self.stopwords_lst]
        return " ".join(stopword)
    
    def clean_pipeline(self, text):
        """Pipeline xử lý văn bản hoàn chỉnh - tích hợp từ code của bạn"""
        if not text or pd.isna(text):
            return ""
        text = self.clean_text(text)
        text = self.translate_english(text)
        text = self.fix_teencode(text)
        text = self.remove_wrongword(text)
        text = self.remove_stopword(text)
        return text

# Class hệ thống ML với dữ liệu thực
class CompanyRecommendationSystem:
    def __init__(self):
        self.clustercty = None
        self.tfidf = TfidfVectorizer(max_features=500)
        self.pca = PCA(n_components=50, random_state=42)
        self.best_model = None
        self.df_structured_encoded = None
        self.X_full = None
        self.text_processor = TextProcessor()
        self.structured_cols = ['Company Type', 'Company industry', 'Company size', 'Country', 'Working days', 'Overtime Policy']
        self.text_cols = ['Company overview_new', "Why you'll love working here_new", 'Our key skills_new', 'keyword']

    def load_gensim_models(self):
        """Load pre-trained Gensim models"""
        try:
            # Gensim Problem 1
            if os.path.exists("models/gensim_dictionary.pkl"):
                with open("models/gensim_dictionary.pkl", "rb") as f:
                    self.gensim_dictionary = pickle.load(f)
                with open("models/gensim_corpus.pkl", "rb") as f:
                    self.gensim_corpus = pickle.load(f)
                self.gensim_tfidf = models.TfidfModel.load("models/gensim_tfidf_model.tfidf")
                self.gensim_index = similarities.SparseMatrixSimilarity.load("models/gensim_similarity.index")
                
                # Gensim Problem 2
                with open("models/gensim_dictionary_2.pkl", "rb") as f:
                    self.gensim_dictionary_2 = pickle.load(f)
                self.gensim_tfidf_2 = models.TfidfModel.load("models/gensim_tfidf_model_2.tfidf")
                self.gensim_index_2 = similarities.SparseMatrixSimilarity.load("models/gensim_similarity_2.index")
                
                return True
            else:
                st.warning("⚠️ Không tìm thấy Gensim models. Vui lòng train models trước.")
                return False
        except Exception as e:
            st.error(f"❌ Lỗi load Gensim models: {e}")
            return False

    def load_cosine_models(self):
        """Load pre-trained Cosine Similarity models"""
        try:
            if os.path.exists("models/cosine_index.pkl"):
                with open("models/cosine_index.pkl", "rb") as f:
                    self.cosine_index = pickle.load(f)
                with open("models/cosine_tfidf_2.pkl", "rb") as f:
                    self.cosine_tfidf_2 = pickle.load(f)
                with open("models/cosine_tfidf_matrix_2.pkl", "rb") as f:
                    self.cosine_tfidf_matrix_2 = pickle.load(f)
                return True
            else:
                st.warning("⚠️ Không tìm thấy Cosine models. Vui lòng train models trước.")
                return False
        except Exception as e:
            st.error(f"❌ Lỗi load Cosine models: {e}")
            return False

    def find_similar_companies_gem(self, company_id, corpus, tfidf, index, top_n):
        # 1. Lấy vector TF-IDF của công ty được chọn
        tfidf_vec = tfidf[corpus[company_id]]
        # 2. Tính cosine similarity giữa công ty này và tất cả các công ty khác
        sims = index[tfidf_vec]  # (Là tính cosin giữa vector hỏi và matran vector đang có, cosin gần 1 thì càng // hay trùng nhau: [a, b] x [b, 1] = [a, 1])
        # 3. Sắp xếp theo độ tương tự giảm dần, loại chính nó ra, lấy top 5
        top_similar_gem_find = sorted([(i, sim) for i, sim in enumerate(sims) if i != company_id],key=lambda x: x[1],reverse=True)[:top_n]
        # 4. Lấy ID và similarity
        company_ids = [i[0] for i in top_similar_gem_find]
        similarities = [round(i[1], 4) for i in top_similar_gem_find]
        # 5. Lấy dữ liệu từ gốc
        df_gem_find = df.iloc[company_ids].copy()
        df_gem_find['similarity'] = similarities

        return df_gem_find, top_similar_gem_find, company_id

    def search_similar_companies_gem(self, query_text, clean_pipeline, dictionary, tfidf_model, index_2, data, top_n):
        # 1. Làm sạch và tách từ
        clean_text = self.text_processor.clean_pipeline(query_text)
        tokens = clean_text.split()  # hoặc dùng tokenizer riêng nếu bạn có
        # 2. Chuyển sang dạng vector BoW
        bow_vector = dictionary.doc2bow(tokens)
        # 3. Chuyển sang vector TF-IDF
        tfidf_vector = tfidf_model[bow_vector]
        # 4. Tính độ tương tự với toàn bộ công ty
        sims = index_2[tfidf_vector]
        # 5. Sắp xếp theo độ tương tự giảm dần
        top_similar_gem_search = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_n]
        # 6. Lấy ID và similarity
        company_ids = [i[0] for i in top_similar_gem_search]
        similarities = [round(i[1], 4) for i in top_similar_gem_search]
        # 7. Lấy dữ liệu từ gốc
        df_gem_search = data.iloc[company_ids].copy()
        df_gem_search['similarity'] = similarities

        return df_gem_search, top_similar_gem_search, query_text

    def find_similar_companies_cos(self, cosine_similarities, data, company_id, top_n):
        # Bỏ chính nó ra bằng cách gán -1
        sim_scores = cosine_similarities[company_id].copy()
        sim_scores[company_id] = -1
        # Lấy top_n chỉ số công ty tương tự nhất
        similar_indices = sim_scores.argsort()[-top_n:][::-1]
        # Tạo danh sách (score, index)
        top_similar_cos_find = [(i, sim_scores[i]) for i in similar_indices]
        # Lấy dòng dữ liệu công ty từ DataFrame
        df_cos_find = data.iloc[similar_indices].copy()
        df_cos_find["similarity"] = [sim_scores[i] for i in similar_indices]

        return top_similar_cos_find, df_cos_find, company_id

    def search_similar_companies_cos(self,query_text_2, vectorizer, tfidf_matrix, data, top_n=5):
        # 1. Làm sạch từ khóa truy vấn
        cleaned_query = ml_system.text_processor.clean_pipeline(query_text_2)
        # 2. Chuyển thành vector TF-IDF (dạng 1×n)
        query_vector = vectorizer.transform([cleaned_query])  # giữ nguyên từ điển cũ
        # 3. Tính độ tương đồng cosine với toàn bộ công ty
        sims = cosine_similarity(query_vector, tfidf_matrix)[0]  # kết quả là vector 1D
        # 4. Lấy top N công ty có điểm similarity cao nhất
        similar_indices = sims.argsort()[-top_n:][::-1]  # sắp xếp giảm dần
        # 5. Tạo kết quả danh sách điểm và chỉ số/
        top_similarity_cos_search = [(sims[i], i) for i in similar_indices]
        # 6. Tạo DataFrame các công ty tương tự
        df_cos_search = data.iloc[similar_indices].copy()
        df_cos_search["similarity"] = [sims[i] for i in similar_indices]

        return top_similarity_cos_search , df_cos_search, query_text_2

   # Hàm lấy index từ danh sách công ty chọn
    def suggest_company_name(self, df, key = None, on_change=None):
        # Tạo mapping: tên công ty → id (nếu tên trùng nhau sẽ lấy ID đầu tiên)
        company_mapping = (df.set_index("Company Name")["id"].to_dict())
        # Danh sách tên công ty, thêm lựa chọn đầu tiên là "Tất cả" hoặc "-- Chọn công ty --"
        company_list = ["-- Chọn công ty --"] + sorted(company_mapping.keys())

        # Tạo selectbox
        selected_name = st.selectbox(
            "Choose company:",
            options=company_list,
            key=key,
            on_change=on_change)
        
        # Nếu người dùng chưa chọn công ty cụ thể → trả None
        if selected_name == "-- Chọn công ty --":
            selected_id = None
        else:
            selected_id = company_mapping.get(selected_name, None)
        return selected_name, selected_id

    def show_company_detail(self, data, title=None, expanded=False):
        """Hiển thị chi tiết công ty"""
        with st.expander(title or f"{data['Company Name']}", expanded=expanded):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**🏭 Ngành:** {data.get('Company industry', 'N/A')}")
                st.markdown(f"**👥 Quy mô:** {data.get('Company size', 'N/A')}")
                st.markdown(f"**🌍 Quốc gia:** {data.get('Country', 'N/A')}")
                st.markdown(f"**🏢 Loại:** {data.get('Company Type', 'N/A')}")
            
            with col2:
                st.markdown(f"**📅 Làm việc:** {data.get('Working days', 'N/A')}")
                st.markdown(f"**⏰ Chính sách OT:** {data.get('Overtime Policy', 'N/A')}")
                st.markdown(f"**😊 Địa chỉ:** {data.get('Location', 'N/A')}")
                st.markdown(f"**🔑 Website:** {data.get('Href', 'N/A')}")
                    
            # Company details
            if 'Company overview' in data:
                st.markdown("**📝 Mô tả công ty:**")
                st.write(data.get('Company overview', 'Không có thông tin'))
            
            if "Why you'll love working here" in data:
                st.markdown("**💝 Tại sao bạn sẽ yêu thích:**")
                st.write(data.get("Why you'll love working here", 'Không có thông tin'))
            
            if 'Our key skills' in data:
                st.markdown("**🔧 Kỹ năng cần thiết:**")
                st.write(data.get('Our key skills', 'Không có thông tin'))

    def draw_similarity_bar_chart(self,data):
        data["similarity"] = data["similarity"].clip(0, 1)
        # Sắp xếp dữ liệu tăng dần theo similarity
        df_sorted = data.sort_values(by="similarity", ascending=True)

        # Tạo biểu đồ
        fig = px.bar(
            df_sorted,
            x="similarity",
            y="Company Name",
            orientation='h',
            color="similarity",
            # color_continuous_scale="Blues",
            color_continuous_scale=[
            [0.0, '#BBDEFB'],  # đậm
            [0.5, '#42A5F5'],
            [1.0, '#1565C0']   # nhạt
        ],
            text="similarity",
            hover_data=["Company Type", "Company industry"],  # 👉 Tooltip mở rộng
            title="So sánh độ tương đồng"
        )

        fig.update_layout(
            xaxis=dict(range=[0, 1]),
            yaxis_title="Tên công ty",
            xaxis_title="Độ tương đồng",
            height=450,
            plot_bgcolor="white"
        )
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside', textfont=dict(color='#0D47A1'))

        st.plotly_chart(fig, use_container_width=True)
    
    def show_similarity_results(self, data, cols_show=None):
        """Hiển thị biểu đồ, bảng và chi tiết công ty tương đồng"""
        
        st.subheader("🏢 Công ty tương đồng:")
        subtab1, subtab2 = st.tabs(["📊 Biểu đồ", "📋 Dữ liệu"])

        with subtab1:
            self.draw_similarity_bar_chart(data)

        with subtab2:
            if cols_show is None:
                cols_show = ["Company Name", "similarity"]
            st.dataframe(data[cols_show].style.format({"similarity": "{:.4f}"}))

        st.subheader("🏢 Thông tin công ty tương đồng:")
        for idx, row in data.iterrows():
            self.show_company_detail(
                row,
                title=f"{row['Company Name']} (Similarity: {row['similarity']:.4f})"
            )

    def handle_input_conflict(self):
            # Nếu user chọn công ty → reset ô nhập từ khóa
            if st.session_state.selectbox_company != st.session_state.prev_selectbox_company:
                st.session_state.query_text = ""
                st.session_state.prev_selectbox_company = st.session_state.selectbox_company
                st.session_state.prev_query_text = ""

            # Nếu user gõ từ khóa → reset chọn công ty
            elif st.session_state.query_text != st.session_state.prev_query_text:
                st.session_state.selectbox_company = "-- Chọn công ty --"
                st.session_state.prev_query_text = st.session_state.query_text
                st.session_state.prev_selectbox_company = "-- Chọn công ty --"

# Đọc dữ liệu
df = pd.read_excel('data/Overview_Companies.xlsx')
cols_show = ["Company Name", "Company Type", "Company industry", "similarity"]

# Load dữ liệu nếu chưa load
# 2️⃣ Khởi tạo hệ thống nếu chưa có
if not st.session_state.data_loaded:
    with st.spinner("🔄 Đang load dữ liệu..."):
        ml_system = CompanyRecommendationSystem()
        ml_system.load_gensim_models()
        ml_system.load_cosine_models()
        st.session_state.ml_system = ml_system
        st.session_state.data_loaded = True
else:
    ml_system = st.session_state.ml_system

#------- I.Giao diện Streamlit -----
# 1.1. Hình ảnh đầu tiên
st.image('images/channels4_banner.jpg', use_container_width=True)

# 1.2.Slidebar
st.sidebar.markdown("## 🤖 AI Company Recommendation System")
st.sidebar.markdown("---")

#Menu categories
page = st.sidebar.radio("📋 Select Category:",["Company Similarity", "Recommendation"])

# Thông tin người tạo
st.sidebar.markdown("---")
st.sidebar.markdown("### 👥 Creators Information")
st.sidebar.markdown("Võ Minh Trí")
st.sidebar.markdown("Email: trivm203@gmail.com")
st.sidebar.markdown("Phạm Thị Thu Thảo")
st.sidebar.markdown("Email: thaofpham@gmail.com")

# 1.3.Các tab nằm ngang
tab1, tab2, tab3 = st.tabs(["Business Objective", "Build Project", "New Prediction"])

#------- II.Nội dung chính từng tab -----
#2.1. Tóm tắt dự án (Business Objective)
with tab1:
    #2.1.1.  Company Similarity
    if page == "Company Similarity":
        st.markdown('<h1 class="main-header">🎯 Business Objective</h1>', unsafe_allow_html=True)
        st.markdown("""
        ### 🔍 Đề xuất công ty tương tự và phù hợp

        Dự án **Company Similarity** ứng dụng các kỹ thuật **Xử lý ngôn ngữ tự nhiên (NLP)** và **học máy không giám sát** nhằm xây dựng hệ thống đề xuất công ty dựa trên nội dung mô tả.

        Sử dụng kết hợp **Gensim (TF-IDF Vectorizer)** và **Cosine Similarity**, hệ thống giải quyết hai bài toán chính:

        ---

        #### 📌 Bài toán 1: Đề xuất các công ty tương tự
        Người dùng chọn một công ty bất kỳ từ danh sách. Hệ thống sẽ phân tích nội dung mô tả của công ty đó và đề xuất **5 công ty có nội dung tương đồng nhất**.

        #### 📌 Bài toán 2: Tìm công ty phù hợp với một mô tả cụ thể
        Người dùng nhập vào một đoạn mô tả mong muốn (ví dụ: môi trường làm việc, công nghệ sử dụng, phong cách quản lý...). Hệ thống sẽ tính toán độ tương đồng giữa mô tả này với các công ty trong cơ sở dữ liệu và **đề xuất công ty phù hợp nhất**.

        ---

        #### ⚙️ Công nghệ sử dụng:
        - **Gensim**: Vector hóa văn bản bằng TF-IDF  
        - **Cosine Similarity**: Đo độ tương đồng giữa các vector mô tả
        """)

    #2.1.2.  Recommendation
    elif page == "Recommendation":
        st.markdown('<h1 class="main-header">🎯 Business Objective</h1>', unsafe_allow_html=True)
    
        st.markdown("""
        <div class="section-header">Mục tiêu của dự án</div>
    
        Hệ thống đề xuất công ty này được phát triển nhằm:
        
        ### 🎯 Mục tiêu chính:
        - **Hỗ trợ ứng viên**: Giúp ứng viên tìm kiếm công ty phù hợp với mong muốn và kỹ năng của họ
        - **Tối ưu hóa tuyển dụng**: Cải thiện quá trình matching giữa ứng viên và nhà tuyển dụng
        - **Phân tích thị trường**: Cung cấp insights về xu hướng tuyển dụng và yêu cầu công việc
        
        ### 🔍 Tính năng chính:
        1. **Phân tích công ty**: Cung cấp thông tin chi tiết về các công ty hàng đầu
        2. **Đề xuất thông minh**: Sử dụng machine learning để đề xuất công ty phù hợp
        3. **So sánh công ty**: Giúp ứng viên so sánh các lựa chọn khác nhau
        4. **Dự đoán xu hướng**: Phân tích và dự đoán xu hướng tuyển dụng
        
        ### 📊 Lợi ích:
        - Tiết kiệm thời gian tìm kiếm việc làm
        - Tăng tỷ lệ match thành công
        - Cung cấp thông tin minh bạch về thị trường lao động
        - Hỗ trợ quyết định nghề nghiệp
        
        ### 🤖 AI Features:
        - **Vietnamese Text Processing**: Xử lý văn bản tiếng Việt với teencode, stopwords
        - **TF-IDF Vectorization**: Chuyển đổi văn bản thành vector số
        - **K-means Clustering**: Phân nhóm công ty theo đặc điểm
        - **Multiple ML Models**: Random Forest, SVM, Logistic Regression, etc.
        - **Cosine Similarity**: Tính toán độ tương đồng thông minh
        """, unsafe_allow_html=True)

#2.2. Mô tả thuật toán (Build Project)       
with tab2:
    #2.2.1. Company Similarity
    if page == "Company Similarity":
        st.markdown('<h1 class="main-header">🔨 Build Project</h1>', unsafe_allow_html=True)
        #a) Mô tả thuật toán
        st.markdown("#### ⚙️ MODEL DESCRIPTION")
        #Gensim
        with st.expander("🧠 GENSIM"):
            st.markdown("""
            #### **📝 Bài toán 1: Dựa trên dữ liệu phân loại**
    - Sử dụng: `Company Type`, `Company industry`, `Company size`, `Country`, `Working days`, `Overtime Policy`
    - Tạo **dictionary**, **corpus**.
    - Vector hóa bằng **TF-IDF**.
    - Tính độ tương tự và chọn **Top 5 công ty giống nhất**.
    """)
            st.markdown("""
            #### **📝 Bài toán 2: Dựa trên mô tả tự do**
    - Dùng các trường: `Company Overview`, `Key Skills`, `Why you'll love working here`.
    - Tạo TF-IDF và vector hóa từ **truy vấn người dùng**.
    - So sánh và chọn **công ty phù hợp nhất**.
    """)
        #Cosine-similarity
        with st.expander("📊 COSINE-SIMILARITY" ):
            st.markdown("""
            #### **📝 Bài toán 1: Dựa trên dữ liệu phân loại**
    - Sử dụng: `Company Type`, `Company industry`, `Company size`, `Country`, `Working days`, `Overtime Policy`
    - Vector hóa các trường phân loại bằng **TF-IDF**.
    - Tính toán **cosine similarity** giữa các vector công ty.
    - Lọc các cặp công ty có độ tương tự **lớn hơn 0.5** để trực quan hóa.
    - Khi người dùng chọn 1 công ty:
        + Lấy **hàng tương ứng trong ma trận độ tương tự**.
        + **Sắp xếp** theo độ tương đồng giảm dần.
        + Trả về **Top 5 công ty tương đồng nhất**.
    """)
            st.markdown("""
            #### **📝 Bài toán 2: Dựa trên mô tả tự do**
    - Dùng các trường: `Company Overview`, `Key Skills`, `Why you'll love working here`.
    - Tạo **TF-IDF vector** từ các mô tả tự do của từng công ty.
    - Biến đổi **truy vấn hoặc từ khóa người dùng nhập** thành vector TF-IDF.
    - Tính toán **cosine similarity** giữa truy vấn và tất cả công ty.
    - Trả về **công ty có độ tương đồng cao nhất** với truy vấn.
    """) 
        #b) EDA dữ liệu    
        st.markdown("#### 🧭 Input Data EDA")
        
        tab4, tab5 = st.tabs(["📊 Categories Analysis", "📝 Free Text Analysis"])

        with tab4:
            st.image("images/eda_input_text_cot.png", caption="📊 Tổng quan 6 trường phân loại\n(Company Type, Industry, Size, Country, ...)")
            st.markdown("")
            st.image("images/eda_company_type_piechart.png", caption="🥧 Phân phối các công ty dựa trên loại hình hoạt động")

        with tab5:
            st.image("images/eda_noidungcty_word_cloud.png", caption="☁️ WordCloud: Tổng hợp mô tả công ty\n(Overview, Skills, Why you'll love)")
            st.markdown("")
            st.image("images/eda_kill_word_cloud.png", caption="🎯 Kỹ năng yêu cầu tại các công ty")
                                
    #2.2.2. Recommendation
    elif page == "Recommendation":
        st.markdown('<h1 class="main-header">🔨 Build Project</h1>', unsafe_allow_html=True)

        try:
            with open("data/training_meta.json", "r", encoding="utf-8") as f:
                training_meta = json.load(f)
            sklearn_df = pd.read_csv("data/sklearn_results.csv")
            with open("data/pyspark_results.json", "r", encoding="utf-8") as f:
                pyspark_results = json.load(f)
            with open("models/cluster_data.pkl", "rb") as f:
                clustercty = pickle.load(f)

            st.success("✅ Đã load kết quả từ mô hình đã train!")

            # Hiển thị metric tổng quan
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Tổng công ty", len(clustercty))
            with col2:
                st.metric("🎯 Số cluster", clustercty['cluster'].nunique())
            with col3:
                st.metric("🏆 Best model", training_meta["best_model"])
            with col4:
                st.metric("📈 Accuracy", f"{training_meta['best_accuracy'] * 100:.2f}%")

            # Hiển thị bảng và biểu đồ
            st.markdown("### 📊 So sánh các mô hình")
            st.image("images/comparison_all_models.png", caption="So sánh mô hình Sklearn & PySpark", use_container_width=True)

            st.markdown("### 🍰 Phân bố công ty theo Cluster")
            st.image("images/cluster_distribution_pie.png", caption="Tỷ lệ các cụm công ty", use_container_width=True)

            st.markdown("### 🔍 Kết quả chi tiết:")
            st.subheader("🔬 Sklearn Models")
            sklearn_df = pd.read_csv("data/sklearn_results.csv", index_col=0)
            sklearn_df.reset_index(inplace=True)
            sklearn_df.rename(columns={"index": "Model"}, inplace=True)
            sklearn_df["Accuracy (%)"] = sklearn_df["Sklearn Accuracy"] * 100
            st.dataframe(sklearn_df[["Model", "Sklearn Accuracy", "Accuracy (%)"]], use_container_width=True)



            st.subheader("⚡ PySpark Models")
            pyspark_df = pd.DataFrame.from_dict(pyspark_results, orient="index", columns=["Accuracy"])
            pyspark_df["Accuracy (%)"] = pyspark_df["Accuracy"] * 100
            st.dataframe(pyspark_df[["Accuracy (%)"]], use_container_width=True)

        except Exception as e:
            st.error(f"❌ Không thể load kết quả: {e}")
            st.info("📌 Vui lòng chạy file `train_and_save_model.py` trước.")

#2.3. Dự đoán kết quả (New Prediction)
with tab3:
    #2.3.1.Company Similarity 
    if page == "Company Similarity":

        st.markdown('<h1 class="main-header">🔮 COMPANY SIMILARITY</h1>', unsafe_allow_html=True)
        #input
        ml_system.handle_input_conflict()  

        col1, col2, col3 = st.columns([4, 1.5, 1.5])
        with col1:               
            # Đặt mặc định nếu chưa có key trong session_state
            selected_name, selected_id = ml_system.suggest_company_name(df, key="selectbox_company")

            query_text = st.text_area("Or type a keyword: ", 
                                    placeholder="VD: CNTT chuyên về mảng Blockchain, AI. Công ty làm việc với khách hàng Nhật Bản...",
                                    height=70,
                                    key="query_text")
      
        with col2:
            selected_model = st.selectbox("📋 Model:", ["Gensim", "Cosine-similarity"], key="selectbox_algo")
        with col3:
            top_n = st.slider("🔢 Number of results:", 1, 10, 5)

        # ✅ Thêm nút tìm kiếm
        tinh_button = st.button("🔍 Search:", use_container_width=True)   

        # ✅ Chỉ chạy nếu nhấn nút
        if tinh_button:
            if st.session_state.selectbox_algo=="Gensim":
                if st.session_state.query_text.strip():
                    # PROCESS
                    df_gem_search, top_similar_gem_search, query_text = ml_system.search_similar_companies_gem(
                        query_text=query_text,
                        clean_pipeline=ml_system.text_processor.clean_pipeline,
                        dictionary=ml_system.gensim_dictionary_2,
                        tfidf_model=ml_system.gensim_tfidf_2,
                        index_2=ml_system.gensim_index_2,
                        data=df,
                        top_n=top_n
                    )

                    # OUTPUT
                    if df_gem_search is not None and not df_gem_search.empty:
                        search_id, search_name = top_similar_gem_search[0]

                        ml_system.show_similarity_results(df_gem_search, cols_show=cols_show)

                elif st.session_state.selectbox_company != "-- Chọn công ty --":
                    # PROCESS
                    df_gem_find, top_similar_gem_find, selected_id = ml_system.find_similar_companies_gem(
                        company_id=selected_id,
                        corpus=ml_system.gensim_corpus,
                        tfidf=ml_system.gensim_tfidf,
                        index=ml_system.gensim_index,
                        top_n=top_n
                    )

                    # OUTPUT
                    st.subheader("🏢 Thông tin công ty đang tìm kiếm")
                    ml_system.show_company_detail(df[df['id'] == selected_id].iloc[0])

                    ml_system.show_similarity_results(df_gem_find, cols_show=cols_show)
                else:
                    # Cảnh báo người dùng chưa nhập gì cả
                    st.warning("⚠️ Vui lòng chọn công ty hoặc nhập từ khóa.")
                
            elif st.session_state.selectbox_algo=="Cosine-similarity": 
                if st.session_state.query_text.strip():
                    #process
                    top_similarity_cos_search , df_cos_search, query_text_2 = ml_system.search_similar_companies_cos(
                        query_text_2=query_text, 
                        vectorizer=ml_system.cosine_tfidf_2, 
                        tfidf_matrix=ml_system.cosine_tfidf_matrix_2, 
                        data=df, 
                        top_n=top_n)
                    #output
                    if df_cos_search is not None and not df_cos_search.empty:
                        search_id, search_name = top_similarity_cos_search[0]

                        ml_system.show_similarity_results(df_cos_search, cols_show=cols_show)
                
                elif st.session_state.selectbox_company != "-- Chọn công ty --":
                    #process
                    top_similar_cos_find, df_cos_find, selected_id = ml_system.find_similar_companies_cos(
                        ml_system.cosine_index, 
                        df, company_id=selected_id, 
                        top_n=top_n)
                    #output
                    st.subheader("🏢 Thông tin công ty đang tìm kiếm")
                    ml_system.show_company_detail(df[df['id'] == selected_id].iloc[0])

                    ml_system.show_similarity_results(df_cos_find, cols_show=cols_show)
                
                else:
                    # Cảnh báo người dùng chưa nhập gì cả
                    st.warning("⚠️ Vui lòng chọn công ty hoặc nhập từ khóa.")     

    #2.3.2. Recommendation
    elif page == "Recommendation":
        st.markdown('<h1 class="main-header">🔮 RECOMMENDATION</h1>', unsafe_allow_html=True)
        try:
            # Load các thành phần đã huấn luyện
            with open("models/best_model.pkl", "rb") as f:
                best_model = pickle.load(f)
            with open("models/tfidf.pkl", "rb") as f:
                tfidf = pickle.load(f)
            with open("models/pca.pkl", "rb") as f:
                pca = pickle.load(f)
            with open("models/encoded_columns.pkl", "rb") as f:
                encoded_columns = pickle.load(f)
            with open("models/cluster_data.pkl", "rb") as f:
                clustercty = pickle.load(f)

            processor = TextProcessor()
            st.success("✅ Mô hình và dữ liệu đã được tải thành công!")

            # Nhập input từ người dùng
            col1, col2 = st.columns(2)
            with col1:
                company_type = st.selectbox("🏢 Loại công ty", clustercty['Company Type'].unique())
                company_industry = st.selectbox("🏭 Ngành nghề", clustercty['Company industry'].unique())
                company_size = st.selectbox("👥 Quy mô", clustercty['Company size'].unique())
            with col2:
                country = st.selectbox("🌍 Quốc gia", clustercty['Country'].unique())
                working_days = st.selectbox("📅 Ngày làm việc", clustercty['Working days'].unique())
                ot_policy = st.selectbox("⏰ Chính sách OT", clustercty['Overtime Policy'].unique())

            text_input = st.text_area("📝 Mô tả mong muốn của bạn:",
            placeholder="Ví dụ: Tôi muốn làm việc trong môi trường công nghệ, sử dụng AI và machine learning, có cơ hội phát triển...",
            height=100)
            threshold = st.slider("🎯 Ngưỡng độ tương đồng:", 0.0, 1.0, 0.1, 0.05)

            if st.button("🔍 Đề xuất công ty", use_container_width=True):
                if not text_input.strip():
                    st.warning("⚠️ Vui lòng nhập mô tả mong muốn của bạn.")
                    st.stop()

                with st.spinner("🤖 AI đang phân tích..."):
                    cleaned_text = processor.clean_pipeline(text_input)
                    text_vec = tfidf.transform([cleaned_text])

                    input_df = pd.DataFrame([{
                        "Company Type": company_type,
                        "Company industry": company_industry,
                        "Company size": company_size,
                        "Country": country,
                        "Working days": working_days,
                        "Overtime Policy": ot_policy
                    }])
                    input_encoded = pd.get_dummies(input_df)
                    for col in encoded_columns:
                        if col not in input_encoded.columns:
                            input_encoded[col] = 0
                    input_encoded = input_encoded[encoded_columns]

                    X_full = pd.concat([
                        input_encoded.reset_index(drop=True),
                        pd.DataFrame(text_vec.toarray(), columns=tfidf.get_feature_names_out())
                    ], axis=1)
                    X_pca = pca.transform(X_full)
                    pred_cluster = best_model.predict(X_pca)[0]

                    from sklearn.metrics.pairwise import cosine_similarity
                    all_vecs = tfidf.transform(clustercty["combined_text"])
                    sim_scores = cosine_similarity(text_vec, all_vecs).flatten()

                    clustercty["similarity_score"] = sim_scores
                    matched = clustercty[(clustercty["cluster"] == pred_cluster) & (clustercty["similarity_score"] >= threshold)]
                    matched = matched.sort_values(by="similarity_score", ascending=False).head(10)

                    if not matched.empty:
                        st.markdown(f"""
                        <div class="prediction-container">
                            <h3>🎯 Kết quả Dự đoán</h3>
                            <p><strong>Cluster được dự đoán:</strong> {pred_cluster}</p>
                            <p><strong>Số công ty phù hợp:</strong> {len(matched)}</p>
                            <p><strong>Ngưỡng tương đồng:</strong> {threshold}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("### 🏆 Các công ty được đề xuất:")

                        # 💡 Dùng lại show_company_detail()
                        ml_system = CompanyRecommendationSystem()
                        ml_system.clustercty = clustercty  # truyền data vào class để hiển thị nội dung

                        for idx, row in matched.iterrows():
                            ml_system.show_company_detail(
                                row,
                                title=f"#{idx+1} {row['Company Name']} - Similarity: {row['similarity_score']:.3f}",
                                expanded=False
                            )

                        if len(matched) > 1:
                            st.markdown("### 📊 Biểu đồ độ tương đồng:")
                            fig = px.bar(
                                matched,
                                x="Company Name",
                                y="similarity_score",
                                title="🎯 Độ tương đồng các công ty được đề xuất",
                                labels={'x': 'Công ty', 'y': 'Độ tương đồng'}
                            )
                            fig.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("❌ Không tìm thấy công ty phù hợp với tiêu chí đã chọn.")

        except Exception as e:
            st.error(f"❌ Lỗi khi tải mô hình hoặc dữ liệu: {e}")
    
    
# Adding a footer
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
.footer p {
    font-size: 17px;  /* Bạn có thể chỉnh to hơn nếu muốn, ví dụ 28px */
    color: blue;
    margin: 5px 0;  /* Để chữ không dính sát mép footer */
}

</style>
<div class="footer">
<p> Trung tâm Tin Học - Trường Đại Học Khoa Học Tự Nhiên <br> Đồ án tốt nghiệp Data Science and Machine Learning </p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)