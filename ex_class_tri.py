import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import string
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from gensim import corpora, models, similarities

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
    from pyspark.ml.feature import VectorAssembler, StringIndexer
    from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression, DecisionTreeClassifier as SparkDecisionTreeClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml import Pipeline
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False

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
        margin-bottom: 1rem;stream
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

# Khởi tạo session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Class xử lý văn bản tích hợp từ code của bạn
class TextProcessor:
    def __init__(self):
        self.teen_dict = {}
        self.stopwords_lst = []
        self.wrong_lst = []
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
        
    @st.cache_data
    def load_data(_self):
        """Load và xử lý dữ liệu thực - KHÔNG tạo sample data"""
        try:
            # Đường dẫn file dữ liệu thực
            data_paths = {
                'translated_data': 'data/translated_data.csv',
                'top2_clusters': 'data/top2_clusters_per_company.csv', 
                'sentiment_data': 'data/sentiment_by_company.csv'
            }
            
            # Kiểm tra file tồn tại
            missing_files = []
            for name, path in data_paths.items():
                if not os.path.exists(path):
                    missing_files.append(path)
            
            if missing_files:
                st.markdown(f"""
                <div class="error-box">
                    <h4>❌ Không tìm thấy các file dữ liệu cần thiết:</h4>
                    <ul>
                        {''.join([f'<li>{file}</li>' for file in missing_files])}
                    </ul>
                    <p><strong>Hướng dẫn:</strong></p>
                    <ol>
                        <li>Tạo thư mục <code>data/</code> trong project</li>
                        <li>Upload các file CSV vào thư mục <code>data/</code></li>
                        <li>Đảm bảo tên file chính xác như trên</li>
                        <li>Refresh lại trang</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
                return False
            
            # Load dữ liệu thực
            st.info("📂 Đang load dữ liệu thực từ files...")
            
            with st.spinner("Loading translated_data.csv..."):
                clustercty = pd.read_csv(data_paths['translated_data'])
                st.success(f"✅ Loaded {len(clustercty)} companies from translated_data.csv")
            
            with st.spinner("Loading cluster data..."):
                new_data = pd.read_csv(data_paths['top2_clusters'])
                st.success(f"✅ Loaded {len(new_data)} cluster records")
            
            with st.spinner("Loading sentiment data..."):
                sentiment_cln = pd.read_csv(data_paths['sentiment_data'])
                st.success(f"✅ Loaded {len(sentiment_cln)} sentiment records")
            
            # Kiểm tra dữ liệu không trống
            if clustercty.empty or new_data.empty or sentiment_cln.empty:
                st.error("❌ Một hoặc nhiều file CSV trống! Vui lòng kiểm tra dữ liệu.")
                return False
            
            # Merge dữ liệu theo logic của bạn
            st.info("🔗 Đang merge dữ liệu...")
            new_data = pd.merge(new_data, sentiment_cln[['Company Name','sentiment_group']], on='Company Name', how='left')
            clustercty = clustercty.merge(new_data[['Company Name', 'keyword', 'sentiment_group']], on='Company Name', how='left')
            
            # Xử lý cột không cần thiết
            if 'Unnamed: 0' in clustercty.columns:
                clustercty.drop(columns=['Unnamed: 0'], inplace=True)
            
            # Điền giá trị null
            clustercty['keyword'].fillna('không xác định', inplace=True)
            clustercty['sentiment_group'].fillna('neutral', inplace=True)
            
            # Kiểm tra các cột cần thiết
            required_cols = _self.structured_cols + _self.text_cols
            missing_cols = [col for col in required_cols if col not in clustercty.columns]
            
            if missing_cols:
                st.error(f"❌ Thiếu các cột cần thiết: {missing_cols}")
                st.info("📋 Các cột có sẵn: " + ", ".join(clustercty.columns.tolist()))
                return False
            
            # Lưu dữ liệu
            _self.clustercty = clustercty
            st.success(f"🎉 Load dữ liệu thành công: {len(clustercty)} công ty với đầy đủ thông tin!")
            
            # Hiển thị thống kê
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Tổng công ty", len(clustercty))
            with col2:
                st.metric("🏭 Số ngành", clustercty['Company industry'].nunique())
            with col3:
                st.metric("😊 Có sentiment", clustercty['sentiment_group'].notna().sum())
            with col4:
                st.metric("🔑 Có keywords", clustercty['keyword'].notna().sum())
            
            return True
            
        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
                <h4>❌ Lỗi load dữ liệu:</h4>
                <p><strong>Chi tiết lỗi:</strong> {str(e)}</p>
                <p><strong>Hướng dẫn khắc phục:</strong></p>
                <ol>
                    <li>Kiểm tra format file CSV (UTF-8 encoding)</li>
                    <li>Đảm bảo các cột cần thiết có trong file</li>
                    <li>Kiểm tra đường dẫn file</li>
                    <li>Thử load từng file riêng lẻ để debug</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            return False
    
    
    def recommend_companies(self, user_input, text_input, threshold=0.1):
        """Đề xuất công ty"""
        try:
            # Xử lý text input
            cleaned_text = self.text_processor.clean_pipeline(text_input)
            tfidf_vec = self.tfidf.transform([cleaned_text])
            
            # Xử lý structured input
            structured_df = pd.DataFrame([user_input])
            structured_encoded = pd.get_dummies(structured_df)
            
            # Đảm bảo có đủ columns
            missing_cols = set(self.df_structured_encoded.columns) - set(structured_encoded.columns)
            for col in missing_cols:
                structured_encoded[col] = 0
            structured_encoded = structured_encoded[self.df_structured_encoded.columns]
            
            # Gộp features
            user_input_vector = pd.concat([
                structured_encoded.reset_index(drop=True), 
                pd.DataFrame(tfidf_vec.toarray(), columns=self.tfidf.get_feature_names_out())
            ], axis=1)
            
            # PCA transform
            user_input_pca = self.pca.transform(user_input_vector)
            
            # Predict cluster
            predicted_cluster = self.best_model.predict(user_input_pca)[0]
            
            # Tính Cosine Similarity
            company_text_vectors = self.tfidf.transform(self.clustercty['combined_text'])
            similarity_scores = cosine_similarity(tfidf_vec, company_text_vectors).flatten()
            self.clustercty['similarity_score'] = similarity_scores
            
            # Lọc công ty
            matched = self.clustercty[
                (self.clustercty['cluster'] == predicted_cluster) & 
                (self.clustercty['similarity_score'] >= threshold)
            ].copy()
            
            matched = matched.sort_values(by='similarity_score', ascending=False).head(10)
            
            return matched, predicted_cluster
            
        except Exception as e:
            st.error(f"❌ Lỗi đề xuất: {e}")
            return pd.DataFrame(), -1
    
    def get_companies_by_cluster_sentiment(self, cluster_id=None, sentiment=None):
        """Lấy công ty theo cluster và sentiment"""
        if self.clustercty is None:
            return pd.DataFrame()
        
        filtered_data = self.clustercty.copy()
        
        if cluster_id is not None:
            filtered_data = filtered_data[filtered_data['cluster'] == cluster_id]
        
        if sentiment is not None:
            filtered_data = filtered_data[filtered_data['sentiment_group'] == sentiment]
        
        return filtered_data

    def load_gensim_models(self):
        """Load pre-trained Gensim models"""
        try:
            # Gensim Problem 1
            if os.path.exists("gensim_dictionary.pkl"):
                with open("gensim_dictionary.pkl", "rb") as f:
                    self.gensim_dictionary = pickle.load(f)
                with open("gensim_corpus.pkl", "rb") as f:
                    self.gensim_corpus = pickle.load(f)
                self.gensim_tfidf = models.TfidfModel.load("gensim_tfidf_model.tfidf")
                self.gensim_index = similarities.SparseMatrixSimilarity.load("gensim_similarity.index")
                
                # Gensim Problem 2
                with open("gensim_dictionary_2.pkl", "rb") as f:
                    self.gensim_dictionary_2 = pickle.load(f)
                self.gensim_tfidf_2 = models.TfidfModel.load("gensim_tfidf_model_2.tfidf")
                self.gensim_index_2 = similarities.SparseMatrixSimilarity.load("gensim_similarity_2.index")
                
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
            if os.path.exists("cosine_index.pkl"):
                with open("cosine_index.pkl", "rb") as f:
                    self.cosine_index = pickle.load(f)
                with open("cosine_tfidf_2.pkl", "rb") as f:
                    self.cosine_tfidf_2 = pickle.load(f)
                with open("cosine_tfidf_matrix_2.pkl", "rb") as f:
                    self.cosine_tfidf_matrix_2 = pickle.load(f)
                return True
            else:
                st.warning("⚠️ Không tìm thấy Cosine models. Vui lòng train models trước.")
                return False
        except Exception as e:
            st.error(f"❌ Lỗi load Cosine models: {e}")
            return False

    def find_similar_companies_gensim(self, company_id, top_n=5):
        """Tìm công ty tương tự bằng Gensim - Problem 1"""
        try:
            if not hasattr(self, 'gensim_tfidf'):
                st.error("❌ Gensim models chưa được load!")
                return pd.DataFrame(), []
            
            # Lấy vector TF-IDF của công ty được chọn
            tfidf_vec = self.gensim_tfidf[self.gensim_corpus[company_id]]
            
            # Tính cosine similarity
            sims = self.gensim_index[tfidf_vec]
            
            # Sắp xếp và lấy top N
            top_similar = sorted([(i, sim) for i, sim in enumerate(sims) if i != company_id], 
                               key=lambda x: x[1], reverse=True)[:top_n]
            
            # Lấy dữ liệu
            company_ids = [i[0] for i in top_similar]
            similarities = [round(i[1], 4) for i in top_similar]
            
            df_result = self.clustercty.iloc[company_ids].copy()
            df_result['similarity'] = similarities
            
            return df_result, top_similar
            
        except Exception as e:
            st.error(f"❌ Lỗi Gensim similarity: {e}")
            return pd.DataFrame(), []

    def search_companies_gensim(self, query_text, top_n=5):
        """Tìm kiếm công ty bằng Gensim - Problem 2"""
        try:
            if not hasattr(self, 'gensim_tfidf_2'):
                st.error("❌ Gensim search models chưa được load!")
                return pd.DataFrame(), []
            
            # Làm sạch và tách từ
            clean_text = self.text_processor.clean_pipeline(query_text)
            tokens = clean_text.split()
            
            # Chuyển sang vector BoW
            bow_vector = self.gensim_dictionary_2.doc2bow(tokens)
            
            # Chuyển sang TF-IDF
            tfidf_vector = self.gensim_tfidf_2[bow_vector]
            
            # Tính độ tương tự
            sims = self.gensim_index_2[tfidf_vector]
            
            # Sắp xếp và lấy top N
            top_similar = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_n]
            
            # Lấy dữ liệu
            company_ids = [i[0] for i in top_similar]
            similarities = [round(i[1], 4) for i in top_similar]
            
            df_result = self.clustercty.iloc[company_ids].copy()
            df_result['similarity'] = similarities
            
            return df_result, top_similar
            
        except Exception as e:
            st.error(f"❌ Lỗi Gensim search: {e}")
            return pd.DataFrame(), []

    def find_similar_companies_cosine(self, company_id, top_n=5):
        """Tìm công ty tương tự bằng Cosine Similarity - Problem 1"""
        try:
            if not hasattr(self, 'cosine_index'):
                st.error("❌ Cosine models chưa được load!")
                return pd.DataFrame(), []
            
            # Bỏ chính nó ra
            sim_scores = self.cosine_index[company_id].copy()
            sim_scores[company_id] = -1
            
            # Lấy top N
            similar_indices = sim_scores.argsort()[-top_n:][::-1]
            
            # Tạo kết quả
            top_similar = [(i, sim_scores[i]) for i in similar_indices]
            df_result = self.clustercty.iloc[similar_indices].copy()
            df_result["similarity"] = [sim_scores[i] for i in similar_indices]
            
            return df_result, top_similar
            
        except Exception as e:
            st.error(f"❌ Lỗi Cosine similarity: {e}")
            return pd.DataFrame(), []

    def search_companies_cosine(self, query_text, top_n=5):
        """Tìm kiếm công ty bằng Cosine Similarity - Problem 2"""
        try:
            if not hasattr(self, 'cosine_tfidf_2'):
                st.error("❌ Cosine search models chưa được load!")
                return pd.DataFrame(), []
            
            # Làm sạch query
            cleaned_query = self.text_processor.clean_pipeline(query_text)
            
            # Chuyển thành vector TF-IDF
            query_vector = self.cosine_tfidf_2.transform([cleaned_query])
            
            # Tính độ tương đồng cosine
            sims = cosine_similarity(query_vector, self.cosine_tfidf_matrix_2)[0]
            
            # Lấy top N
            similar_indices = sims.argsort()[-top_n:][::-1]
            
            # Tạo kết quả
            top_similar = [(sims[i], i) for i in similar_indices]
            df_result = self.clustercty.iloc[similar_indices].copy()
            df_result["similarity_score"] = [sims[i] for i in similar_indices]
            
            return df_result, top_similar
            
        except Exception as e:
            st.error(f"❌ Lỗi Cosine search: {e}")
            return pd.DataFrame(), []

    def suggest_company_name(self):
        """Tạo selectbox cho việc chọn công ty"""
        if self.clustercty is None or self.clustercty.empty:
            return None, None
        
        # Tạo mapping: tên công ty → index
        company_mapping = {}
        for idx, row in self.clustercty.iterrows():
            company_mapping[row['Company Name']] = idx
        
        # Tạo danh sách tên công ty
        company_list = sorted(company_mapping.keys())
        
        # Tạo selectbox
        selected_name = st.selectbox(
            "🏢 Chọn công ty:",
            options=company_list,
            key="company_selector"
        )
        
        # Lấy index tương ứng
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
                st.markdown(f"**⏰ OT Policy:** {data.get('Overtime Policy', 'N/A')}")
                st.markdown(f"**😊 Sentiment:** {data.get('sentiment_group', 'N/A')}")
                st.markdown(f"**🔑 Keywords:** {data.get('keyword', 'N/A')}")
            
            if 'Company overview_new' in data:
                st.markdown("**📝 Mô tả công ty:**")
                st.write(data.get('Company overview_new', 'Không có thông tin'))
            
            if "Why you'll love working here_new" in data:
                st.markdown("**💝 Tại sao bạn sẽ yêu thích:**")
                st.write(data.get("Why you'll love working here_new", 'Không có thông tin'))
            
            if 'Our key skills_new' in data:
                st.markdown("**🔧 Kỹ năng cần thiết:**")
                st.write(data.get('Our key skills_new', 'Không có thông tin'))

import subprocess
class PySparkMLSystem:
    def __init__(self):
        self.spark = None
        self.spark_df_ml = None
        self.pyspark_results = {}

# Sidebar
st.sidebar.markdown("## 🤖 AI Company Recommendation System")
st.sidebar.markdown("---")

# Menu categories
menu_options = ["Business Objective", "Company Suggest", "Build Project", "New Prediction"]
selected_menu = st.sidebar.selectbox("📋 Select Category:", menu_options)

# Thông tin người tạo
st.sidebar.markdown("---")
st.sidebar.markdown("### 👥 Creators Information")
st.sidebar.markdown("**Creator 1:**")
st.sidebar.markdown("📧 Võ Minh Trí")
st.sidebar.markdown("✉️ trivm203@gmail.com")
st.sidebar.markdown("**Creator 2:**") 
st.sidebar.markdown("📧 Phạm Thị Thu Thảo")
st.sidebar.markdown("✉️ phamthithuthao@email.com")

# Main content
if selected_menu == "Business Objective":
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

elif selected_menu == "Company Suggest":
    st.markdown('<h1 class="main-header">🏢 Company Suggestions</h1>', unsafe_allow_html=True)

    # Load dữ liệu nếu chưa load
    if not st.session_state.data_loaded:
        with st.spinner("🔄 Đang load dữ liệu..."):
            ml_system = CompanyRecommendationSystem()
            if ml_system.load_data():
                ml_system.load_gensim_models()
                ml_system.load_cosine_models()
                st.session_state.ml_system = ml_system
                st.session_state.data_loaded = True
            else:
                st.stop()

    ml_system = st.session_state.ml_system

    tab1, tab2, tab3 = st.tabs(["🧠 Gensim Recommendation", "📊 Cosine Similarity", "🔍 Smart Text Search"])

    # ---------------------- TAB 1: Gensim Recommendation ---------------------- #
    with tab1:
        st.markdown("### 🧠 Đề xuất công ty bằng Gensim")

        selected_company = st.selectbox("🏢 Chọn công ty:", ml_system.clustercty['Company Name'].unique())
        top_n = st.slider("🔢 Số công ty đề xuất:", 1, 10, 5)

        if st.button("🔍 Tìm công ty tương tự (Gensim)", key="gensim_button"):
            st.markdown("---")
            selected_id = ml_system.clustercty[ml_system.clustercty['Company Name'] == selected_company].index[0]

            st.markdown("### 🏢 Công ty được chọn:")
            ml_system.show_company_detail(ml_system.clustercty.loc[selected_id], expanded=True)

            df_result, _ = ml_system.find_similar_companies_gensim(selected_id, top_n)
            if not df_result.empty:
                st.markdown("### 🤝 Các công ty tương tự:")
                for idx, row in df_result.iterrows():
                    ml_system.show_company_detail(row, title=f"#{idx+1} {row['Company Name']} - Similarity: {row['similarity']:.3f}")
            else:
                st.info("❌ Không tìm thấy công ty tương tự.")

    # ---------------------- TAB 2: Cosine Similarity Search ---------------------- #
    with tab2:
        st.markdown("### 📊 Tìm công ty tương tự bằng mô tả (Cosine Similarity)")

        query_text = st.text_area("📝 Nhập mô tả công ty mong muốn:", placeholder="Ví dụ: môi trường năng động, công nghệ mới, AI, machine learning...")
        top_n_cosine = st.slider("🔢 Số công ty tương tự:", 1, 10, 5)

        if st.button("🔍 Tìm theo mô tả (Cosine)", key="cosine_button"):
            if query_text.strip():
                df_result, _ = ml_system.search_companies_cosine(query_text, top_n_cosine)

                if not df_result.empty:
                    st.markdown("### 🔎 Kết quả tương tự:")
                    for idx, row in df_result.iterrows():
                        ml_system.show_company_detail(row, title=f"#{idx+1} {row['Company Name']} - Similarity: {row['similarity_score']:.3f}")
                else:
                    st.warning("❌ Không tìm thấy công ty phù hợp.")

    with tab3:
        st.markdown("### 🔍 Tìm công ty theo từ khóa")

        keyword_query = st.text_input("Tìm theo từ khóa:", placeholder="Ví dụ: fintech, AI, công nghệ, lương cao...")

        if keyword_query.strip():
            cleaned_query = ml_system.text_processor.clean_pipeline(keyword_query)
            query_words = set(cleaned_query.split())

            search_results = ml_system.clustercty.copy()
            search_results['search_score'] = 0

            for idx, row in search_results.iterrows():
                combined_text = ' '.join([
                    row.get('Company overview_new', ''),
                    row.get('Our key skills_new', ''),
                    row.get("Why you'll love working here_new", ''),
                    row.get('keyword', '')
                ])
                cleaned = ml_system.text_processor.clean_pipeline(combined_text)
                words = set(cleaned.split())
                search_results.at[idx, 'search_score'] = len(words.intersection(query_words)) / len(query_words)

            top_matches = search_results[search_results['search_score'] > 0]
            top_matches = top_matches.sort_values(by='search_score', ascending=False).head(10)

            if not top_matches.empty:
                st.markdown("### ✅ Kết quả tìm thấy:")
                for idx, row in top_matches.iterrows():
                    ml_system.show_company_detail(row, title=f"#{idx+1} {row['Company Name']} - Score: {row['search_score']:.3f}")
            else:
                st.warning("Không tìm thấy công ty nào chứa từ khóa phù hợp.")

elif selected_menu == "Build Project":
    st.markdown('<h1 class="main-header">🔨 Build Project</h1>', unsafe_allow_html=True)

    try:
        with open("saved_model/training_meta.json", "r", encoding="utf-8") as f:
            training_meta = json.load(f)
        sklearn_df = pd.read_csv("saved_model/sklearn_results.csv")
        with open("saved_model/pyspark_results.json", "r", encoding="utf-8") as f:
            pyspark_results = json.load(f)
        with open("saved_model/cluster_data.pkl", "rb") as f:
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
        st.image("saved_model/comparison_all_models.png", caption="So sánh mô hình Sklearn & PySpark", use_column_width=True)

        st.markdown("### 🍰 Phân bố công ty theo Cluster")
        st.image("saved_model/cluster_distribution_pie.png", caption="Tỷ lệ các cụm công ty", use_column_width=True)

        st.markdown("### 🔍 Kết quả chi tiết:")
        st.subheader("🔬 Sklearn Models")
        sklearn_df = pd.read_csv("saved_model/sklearn_results.csv", index_col=0)
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


elif selected_menu == "New Prediction":
    st.markdown('<h1 class="main-header">🔮 New Prediction</h1>', unsafe_allow_html=True)

    try:
        # Load các thành phần đã huấn luyện
        with open("saved_model/best_model.pkl", "rb") as f:
            best_model = pickle.load(f)
        with open("saved_model/tfidf.pkl", "rb") as f:
            tfidf = pickle.load(f)
        with open("saved_model/pca.pkl", "rb") as f:
            pca = pickle.load(f)
        with open("saved_model/encoded_columns.pkl", "rb") as f:
            encoded_columns = pickle.load(f)
        with open("saved_model/cluster_data.pkl", "rb") as f:
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



