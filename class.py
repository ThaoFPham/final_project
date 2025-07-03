import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import string
import os
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
    
    def prepare_features(self):
        """Chuẩn bị features với error handling tốt hơn"""
        try:
            if self.clustercty is None or self.clustercty.empty:
                st.error("❌ Dữ liệu chưa được load hoặc trống!")
                return False
            
            st.info("🔧 Đang chuẩn bị features...")
            
            # Kiểm tra các cột text
            text_cols_available = [col for col in self.text_cols if col in self.clustercty.columns]
            if not text_cols_available:
                st.error(f"❌ Không tìm thấy cột text nào trong: {self.text_cols}")
                return False
            
            st.info(f"📝 Sử dụng các cột text: {text_cols_available}")
            
            # Xử lý văn bản với text processor
            with st.spinner("Đang xử lý văn bản..."):
                self.clustercty['combined_text'] = self.clustercty[text_cols_available].fillna('').agg(' '.join, axis=1)
                self.clustercty['combined_text'] = self.clustercty['combined_text'].apply(self.text_processor.clean_pipeline)
            
            # Kiểm tra kết quả xử lý text
            if self.clustercty['combined_text'].isna().all():
                st.error("❌ Tất cả combined_text đều null sau khi xử lý!")
                return False
            
            # TF-IDF
            with st.spinner("Đang thực hiện TF-IDF vectorization..."):
                tfidf_matrix = self.tfidf.fit_transform(self.clustercty['combined_text'])
                df_tfidf = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=self.tfidf.get_feature_names_out()
                )
            
            st.success(f"✅ TF-IDF: {df_tfidf.shape[1]} features")
            
            # Kiểm tra các cột structured
            structured_cols_available = [col for col in self.structured_cols if col in self.clustercty.columns]
            if not structured_cols_available:
                st.error(f"❌ Không tìm thấy cột structured nào trong: {self.structured_cols}")
                return False
            
            st.info(f"🏗️ Sử dụng các cột structured: {structured_cols_available}")
            
            # One-hot encode
            with st.spinner("Đang thực hiện One-hot encoding..."):
                self.df_structured_encoded = pd.get_dummies(self.clustercty[structured_cols_available], drop_first=True)
            
            st.success(f"✅ One-hot encoding: {self.df_structured_encoded.shape[1]} features")
            
            # Gộp dữ liệu
            X_concat = pd.concat([
                self.df_structured_encoded.reset_index(drop=True), 
                df_tfidf.reset_index(drop=True)
            ], axis=1)
            
            st.info(f"🔗 Combined features: {X_concat.shape[1]} total features")
            
            # PCA
            with st.spinner("Đang thực hiện PCA dimensionality reduction..."):
                n_components = min(50, X_concat.shape[1] - 1, X_concat.shape[0] - 1)
                self.pca = PCA(n_components=n_components, random_state=42)
                self.X_full = self.pca.fit_transform(X_concat)
            
            st.success(f"✅ PCA completed: {self.X_full.shape[1]} components")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Lỗi chuẩn bị features: {str(e)}")
            return False
    
    def find_optimal_clusters(self):
        """Tìm số cluster tối ưu"""
        K = range(2, min(11, len(self.clustercty)))
        silhouette_scores = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, k in enumerate(K):
            status_text.text(f'Đang test k={k}...')
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.X_full)
            score = silhouette_score(self.X_full, labels)
            silhouette_scores.append(score)
            progress_bar.progress((i + 1) / len(K))
        
        status_text.empty()
        progress_bar.empty()
        
        best_k = K[silhouette_scores.index(max(silhouette_scores))]
        
        # Visualize silhouette scores
        fig = px.line(x=list(K), y=silhouette_scores, 
                     title="Silhouette Score vs Number of Clusters",
                     labels={'x': 'Number of Clusters (k)', 'y': 'Silhouette Score'})
        fig.add_vline(x=best_k, line_dash="dash", line_color="red", 
                     annotation_text=f"Best k={best_k}")
        st.plotly_chart(fig, use_container_width=True)
        
        return best_k, silhouette_scores
    
    def train_models(self):
        """Training các mô hình"""
        try:
            # Tìm số cluster tối ưu
            best_k, silhouette_scores = self.find_optimal_clusters()
        
            # Clustering với k tối ưu
            final_kmeans = KMeans(n_clusters=best_k, random_state=42)
            cluster_labels = final_kmeans.fit_predict(self.X_full)
            self.clustercty['cluster'] = cluster_labels
        
            # Chia train/test
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_full, cluster_labels, test_size=0.2, random_state=42
            )
        
            # Các mô hình
            models = {
                "Random Forest": RandomForestClassifier(n_estimators=50),
                "Logistic Regression": LogisticRegression(max_iter=500),
                "Naive Bayes": GaussianNB(),
                "Decision Tree": DecisionTreeClassifier(max_depth=10),
                "KNN": KNeighborsClassifier(n_neighbors=3)
            }
        
            results = []
            trained_models = {}
        
            progress_bar = st.progress(0)
            status_text = st.empty()
        
            for i, (name, model) in enumerate(models.items()):
                status_text.text(f'Training {name}...')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                results.append((name, acc))
                trained_models[name] = model
                progress_bar.progress((i + 1) / len(models))
        
            status_text.empty()
            progress_bar.empty()
            
            # Chọn mô hình tốt nhất
            best_model_name, best_acc = max(results, key=lambda x: x[1])
            self.best_model = trained_models[best_model_name]
        
            st.success(f"🏆 Mô hình tốt nhất: {best_model_name} với accuracy: {best_acc:.3f}")
        
            return results, best_model_name, best_acc, best_k
        
        except Exception as e:
            st.error(f"❌ Lỗi training models: {e}")
            return [], "", 0, 0
    
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
    def suggest_company_name(self, df, key = None):
        # Tạo mapping: tên công ty → id (nếu tên trùng nhau sẽ lấy ID đầu tiên)
        company_mapping = (df.set_index("Company Name")["id"].to_dict())
        # Danh sách tên công ty, thêm lựa chọn đầu tiên là "Tất cả" hoặc "-- Chọn công ty --"
        company_list = ["-- Chọn công ty --"] + sorted(company_mapping.keys())

        # Tạo selectbox
        selected_name = st.selectbox(
            "Nhập tên công ty:",
            options=company_list,
            key=key)
        
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

    def draw_similarity_bar_chart(self,df):
        df["similarity"] = df["similarity"].clip(0, 1)
        # Sắp xếp dữ liệu tăng dần theo similarity
        df_sorted = df.sort_values(by="similarity", ascending=True)

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

import subprocess
class PySparkMLSystem:
    def __init__(self):
        self.spark = None
        self.spark_df_ml = None
        self.pyspark_results = {}

    def initialize_spark(self):
        """Khởi tạo Spark Session với kiểm tra Java"""
        try:
            # Kiểm tra java -version
            result = subprocess.run(['java', '-version'], capture_output=True, text=True)
            if result.returncode != 0:
                st.error("❌ Java không được cài đặt hoặc không trong PATH")
                return False
        except FileNotFoundError:
            st.error("❌ Java không được tìm thấy. Vui lòng cài đặt Java 8 hoặc 11.")
            return False

        try:
            # Tạo SparkSession an toàn
            self.spark = SparkSession.builder \
                .appName("CompanyRecommendation") \
                .config("spark.driver.memory", "2g") \
                .config("spark.executor.memory", "2g") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .getOrCreate()

            # Test Spark bằng cách tạo DataFrame
            test_df = self.spark.createDataFrame([(1, "test")], ["id", "value"])
            test_df.count()  # ép thực thi

            self.spark.sparkContext.setLogLevel("ERROR")
            st.success("✅ Spark đã được khởi tạo thành công!")
            return True

        except Exception as e:
            st.error(f"❌ Lỗi khởi tạo Spark: {e}")
            return False

    
    def prepare_spark_data(self, X_concat, cluster_labels):
        """Chuẩn bị dữ liệu cho PySpark"""
        try:
            # Thêm cluster labels vào X_concat
            X_concat_with_labels = X_concat.copy()
            X_concat_with_labels['cluster'] = cluster_labels
            
            # Convert to Spark DataFrame
            self.spark_df_ml = self.spark.createDataFrame(X_concat_with_labels)
            
            # Assemble features
            feature_columns = X_concat.columns.tolist()
            assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
            self.spark_df_ml = assembler.transform(self.spark_df_ml)
            
            # Select features and labels
            self.spark_df_ml = self.spark_df_ml.select("features", "cluster")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Lỗi chuẩn bị dữ liệu Spark: {e}")
            return False
    
    def train_pyspark_models(self):
        """Training PySpark models"""
        try:
            # Split data
            train_data, test_data = self.spark_df_ml.randomSplit([0.8, 0.2], seed=42)
            
            # Evaluator
            evaluator = MulticlassClassificationEvaluator(
                labelCol="cluster", 
                predictionCol="prediction", 
                metricName="accuracy"
            )
            
            # Logistic Regression
            lr = SparkLogisticRegression(featuresCol="features", labelCol="cluster")
            lr_model = lr.fit(train_data)
            lr_predictions = lr_model.transform(test_data)
            lr_accuracy = evaluator.evaluate(lr_predictions)
            
            # Decision Tree
            dt = SparkDecisionTreeClassifier(featuresCol="features", labelCol="cluster")
            dt_model = dt.fit(train_data)
            dt_predictions = dt_model.transform(test_data)
            dt_accuracy = evaluator.evaluate(dt_predictions)
            
            self.pyspark_results = {
                "PySpark Logistic Regression": lr_accuracy,
                "PySpark Decision Tree": dt_accuracy
            }
            
            return self.pyspark_results
            
        except Exception as e:
            st.error(f"❌ Lỗi training PySpark models: {e}")
            return {}
    
    def stop_spark(self):
        """Dừng Spark Session"""
        if self.spark:
            self.spark.stop()

# Đọc dữ liệu
df = pd.read_excel('data/Overview_Companies.xlsx')
cols_show = ["Company Name", "Company Type", "Company industry", "similarity"]

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
        st.header("COMPANY SIMILARITY")
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
    #2.2.1. Company Similarity
    if page == "Company Similarity":
        st.header("COMPANY SIMILARITY")
        #a) Gensim
        with st.expander("🧠 Thuật toán GENSIM"):
            st.markdown('#### Bài toán 1:')
            st.write('Dùng các nội dung phân loại (Company Type, Company industry, Company size,...) làm dữ liệu đầu vào')
            st.image('images/gen1_dau vao.png')
            st.write('Dùng gensim tạo từ điển dictionary và từ điển tần số từ corpus')
            st.image('images/gen1_dictionary.png')
            st.image('images/gen1_copus.png')
            st.write('Vector hóa bằng tf-idf để tạo ma trận thưa thớt')
            st.write('Lấy vector tf-idf của 1 công ty được chọn rồi tính tỉ số tương tự so với ma trận thưa')
            st.write('Sắp xếp và lấy top5')
            st.image('images/gen1_top5.png')
            st.image('images/gen1_top5_df.png')
            st.markdown('#### Bài toán 2:')
            st.write("Dùng các nội dung mô tả tự do (Company overview, Our key skills, Why you'll love working here) làm dữ liệu đầu vào")
            st.image('images/gen2_input.png')
            st.write('Các bước tạo từ điển và tf-idf tương tự')
            st.write('Từ khóa tìm kiếm sẽ được biến đổi thành vector và so sánh chỉ số tương tự')
            st.write('sắp xếp và lấy công ty tương đồng nhất')
            st.image('images/gen2_top1.png')
        #b) Cosine-similarity
        with st.expander("📊 Thuật toán COSINE-SIMILARITY" ):
            st.markdown('#### Bài toán 1:')
            st.write('Dùng các nội dung phân loại (Company Type, Company industry, Company size,...) làm dữ liệu đầu vào')
            st.image('images/gen1_dau vao.png')
            st.write('Vector hóa trực tiếp bằng tf-idf để tạo ma trận thưa thớt')
            st.write('Tính tỉ số tương tự toàn bộ ma trận thưa')
            st.write('Trực quan hóa các công ty có chỉ số tương tự >0.5')
            st.image('images/cos1_matran.png')
            st.write('Chọn 1 công ty, thuật toán sẽ lấy hàng ngang, sắp xếp và lấy top5')
            st.image('images/cos1_top5.png')
            st.image('images/cos1_top5_df.png')
            st.markdown('#### Bài toán 2:')
            st.write("Dùng các nội dung mô tả tự do (Company overview, Our key skills, Why you'll love working here) làm dữ liệu đầu vào")
            st.image('images/gen2_input.png')
            st.write('Các bước tạo tf-idf tương tự')
            st.write('Từ khóa tìm kiếm sẽ được biến đổi thành vector và so sánh chỉ số tương tự')
            st.write('sắp xếp và lấy công ty tương đồng nhất')
            st.image('images/cos2_top1.png')
    
    #2.2.2. Recommendation
    elif page == "Recommendation":
        st.header("COMPAYNY SIMILARITY")
        st.markdown('<h1 class="main-header">🔨 Build Project</h1>', unsafe_allow_html=True)
    
        # Khởi tạo và load dữ liệu
        if not st.session_state.data_loaded:
            with st.spinner("🔄 Đang load dữ liệu thực..."):
                ml_system = CompanyRecommendationSystem()
                if ml_system.load_data():
                    st.session_state.ml_system = ml_system
                    st.session_state.data_loaded = True
                else:
                    st.stop()
        
        # Training options
        st.markdown("### 🚀 Training Machine Learning Models:")
        
        sklearn_training = st.button("🔬 Train Sklearn Models", use_container_width=True)
        
        if PYSPARK_AVAILABLE:
            pyspark_training = st.button("⚡ Train PySpark Models", use_container_width=True)
        
        else:
            st.info("💡 **PySpark không khả dụng** - Chỉ sử dụng Sklearn Models")
        
        #a) Sklearn Training
        if sklearn_training and st.session_state.data_loaded:
            if not st.session_state.model_trained:
                with st.spinner("🤖 Đang training sklearn models..."):
                    ml_system = st.session_state.ml_system
                    
                    if ml_system.prepare_features():
                        results, best_model_name, best_acc, best_k = ml_system.train_models()
                        
                        if results:
                            st.session_state.model_trained = True
                            st.session_state.training_results = results
                            st.session_state.best_model_name = best_model_name
                            st.session_state.best_acc = best_acc
                            st.session_state.best_k = best_k
                            st.success("✅ Sklearn Training hoàn tất!")
                            st.rerun()
        
        #b) PySpark Training
        if PYSPARK_AVAILABLE and 'pyspark_training' in locals() and pyspark_training:
            if not st.session_state.get('pyspark_trained', False):
                if not st.session_state.model_trained:
                    st.warning("⚠️ Vui lòng train sklearn models trước!")
                else:
                    with st.spinner("⚡ Đang training PySpark models..."):
                        try:
                            ml_system = st.session_state.ml_system
                            pyspark_system = PySparkMLSystem()
                            
                            if pyspark_system.initialize_spark():
                                # Chuẩn bị dữ liệu
                                X_concat = pd.concat([
                                    ml_system.df_structured_encoded.reset_index(drop=True),
                                    pd.DataFrame(
                                        ml_system.tfidf.transform(ml_system.clustercty['combined_text']).toarray(),
                                        columns=ml_system.tfidf.get_feature_names_out()
                                    )
                                ], axis=1)
                                
                                cluster_labels = ml_system.clustercty['cluster'].values
                                
                                if pyspark_system.prepare_spark_data(X_concat, cluster_labels):
                                    pyspark_results = pyspark_system.train_pyspark_models()
                                    
                                    if pyspark_results:
                                        st.session_state.pyspark_trained = True
                                        st.session_state.pyspark_results = pyspark_results
                                        st.success("✅ PySpark Training hoàn tất!")
                                        st.rerun()
                                
                                pyspark_system.stop_spark()
                            
                        except Exception as e:
                            st.error(f"❌ Lỗi PySpark training: {e}")
        
        # Hiển thị kết quả
        if st.session_state.get('model_trained', False):
            st.markdown("### 📊 Kết quả Training")
            
            ml_system = st.session_state.ml_system
            
            # Metrics overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Số công ty", len(ml_system.clustercty))
            with col2:
                st.metric("🎯 Số clusters", st.session_state.best_k)
            with col3:
                st.metric("🏆 Best Model", st.session_state.best_model_name)
            with col4:
                st.metric("📈 Best Accuracy", f"{st.session_state.best_acc:.3f}")
            
            # Sklearn Results
            st.markdown("#### 🔬 Sklearn Models Results:")
            sklearn_results_df = pd.DataFrame(st.session_state.training_results, columns=['Mô hình', 'Accuracy'])
            sklearn_results_df['Accuracy (%)'] = (sklearn_results_df['Accuracy'] * 100).round(2)
            sklearn_results_df['Framework'] = 'Sklearn'
            
            st.dataframe(sklearn_results_df[['Mô hình', 'Accuracy (%)', 'Framework']], use_container_width=True)
            
            # PySpark Results (if available)
            if st.session_state.get('pyspark_trained', False):
                st.markdown("#### ⚡ PySpark Models Results:")
                pyspark_results = st.session_state.pyspark_results
                pyspark_results_df = pd.DataFrame(list(pyspark_results.items()), columns=['Mô hình', 'Accuracy'])
                pyspark_results_df['Accuracy (%)'] = (pyspark_results_df['Accuracy'] * 100).round(2)
                pyspark_results_df['Framework'] = 'PySpark'
                
                st.dataframe(pyspark_results_df[['Mô hình', 'Accuracy (%)', 'Framework']], use_container_width=True)
                
                # Combined comparison
                st.markdown("#### 🆚 Sklearn vs PySpark Comparison:")
                combined_results = pd.concat([
                    sklearn_results_df[['Mô hình', 'Accuracy (%)', 'Framework']],
                    pyspark_results_df[['Mô hình', 'Accuracy (%)', 'Framework']]
                ], ignore_index=True)
                
                fig_comparison = px.bar(
                    combined_results, 
                    x='Mô hình', 
                    y='Accuracy (%)',
                    color='Framework',
                    title="Sklearn vs PySpark Models Comparison",
                    barmode='group'
                )
                fig_comparison.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Sklearn visualization
            fig_sklearn = px.bar(
                sklearn_results_df, 
                x='Mô hình', 
                y='Accuracy (%)',
                title="Sklearn Models Performance",
                color='Accuracy (%)',
                color_continuous_scale='viridis'
            )
            fig_sklearn.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_sklearn, use_container_width=True)
            
            # Cluster analysis
            st.markdown("#### 🎯 Cluster Analysis:")
            if 'cluster' in ml_system.clustercty.columns:
                cluster_stats = ml_system.clustercty.groupby('cluster').agg({
                    'Company Name': 'count',
                    'Company industry': lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A',
                    'sentiment_group': lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A'
                })
                cluster_stats.columns = ['Số công ty', 'Ngành chủ đạo', 'Sentiment chủ đạo']
                st.dataframe(cluster_stats, use_container_width=True)
                
                # Cluster distribution
                fig_cluster = px.pie(
                    values=cluster_stats['Số công ty'],
                    names=cluster_stats.index,
                    title="Phân bố công ty theo Cluster"
                )
                st.plotly_chart(fig_cluster, use_container_width=True)

#2.3. Dự đoán kết quả (New Prediction)
with tab3:
    #2.3.1.Company Similarity 
    if page == "Company Similarity":
        st.header('COMPANY SIMILARITY')
        #input      
        col1, col2, col3 = st.columns([4, 1.5, 1.5])
        with col1:
            selected_name, selected_id = ml_system.suggest_company_name(df, key="selectbox_company_1")
            query_text = st.text_input("Hoặc nhập từ khóa:")
        with col2:
            selected_model = st.selectbox("📋 Thuật toán:", ["Gensim", "Cosine-similarity"], key="selectbox_algo")
        with col3:
            top_n = st.slider("🔢 Số công ty tương tự:", 1, 10, 5)

        # ✅ Thêm nút tìm kiếm
        tinh_button = st.button("🔍 Tìm kiếm:", use_container_width=True)   

        # ✅ Chỉ chạy nếu nhấn nút
        if tinh_button:
            if selected_model=="Gensim":
                if query_text.strip():
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

                        st.subheader("🏢 Công ty tương đồng với từ khóa tìm kiếm")
                        subtab1, subtab2 = st.tabs(["📊 Biểu đồ", "📋 Dữ liệu"])
                        with subtab1:
                            ml_system.draw_similarity_bar_chart(df_gem_search)
                        with subtab2:
                            st.dataframe(df_gem_search[cols_show].style.format({"similarity": "{:.4f}"}))
                                            
                        st.subheader("🏢 Thông tin công ty tương tự")

                        for idx, row in df_gem_search.iterrows():
                            ml_system.show_company_detail(
                                row,
                                title=f"{row['Company Name']} (Similarity: {row['similarity']:.4f})"
                            )

                elif selected_id:
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

                    st.subheader("🏙️ Các công ty tương tự")
                    subtab1, subtab2 = st.tabs(["📊 Biểu đồ", "📋 Dữ liệu"])
                    with subtab1:
                        ml_system.draw_similarity_bar_chart(df_gem_find)
                    with subtab2:
                        st.dataframe(df_gem_find[cols_show].style.format({"similarity": "{:.4f}"}))
                                        
                    st.subheader("🏢 Thông tin công ty tương tự")

                    for idx, row in df_gem_find.iterrows():
                        ml_system.show_company_detail(
                            row,
                            title=f"{row['Company Name']} (Similarity: {row['similarity']:.4f})"
                        )
                else:
                    # Cảnh báo người dùng chưa nhập gì cả
                    st.warning("⚠️ Vui lòng chọn công ty hoặc nhập từ khóa.")
                
            elif selected_model=="Cosine-similarity": 
                if query_text.strip():
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

                        st.subheader("🏢 Công ty tương đồng với từ khóa tìm kiếm")
                        subtab1, subtab2 = st.tabs(["📊 Biểu đồ", "📋 Dữ liệu"])
                        with subtab1:
                            ml_system.draw_similarity_bar_chart(df_cos_search)
                        with subtab2:
                            st.dataframe(df_cos_search[cols_show].style.format({"similarity": "{:.4f}"}))
                                            
                        st.subheader("🏢 Thông tin công ty tương tự")
                        for idx, row in df_cos_search.iterrows():
                            ml_system.show_company_detail(
                                row,
                                title=f"{row['Company Name']} (Similarity: {row['similarity']:.4f})"
                            )
                elif selected_id:
                    #process
                    top_similar_cos_find, df_cos_find, selected_id = ml_system.find_similar_companies_cos(
                        ml_system.cosine_index, 
                        df, company_id=selected_id, 
                        top_n=top_n)
                    #output
                    st.subheader("🏢 Thông tin công ty đang tìm kiếm")
                    ml_system.show_company_detail(df[df['id'] == selected_id].iloc[0])

                    st.subheader("🏙️ Các công ty tương tự")
        
                    subtab1, subtab2 = st.tabs(["📊 Biểu đồ", "📋 Dữ liệu"])
                    with subtab1:
                        ml_system.draw_similarity_bar_chart(df_cos_find)
                    with subtab2:
                        st.dataframe(df_cos_find[cols_show].style.format({"similarity": "{:.4f}"}))
                                            
                    st.subheader("🏢 Thông tin công ty tương tự")
                    for idx, row in df_cos_find.iterrows():
                        ml_system.show_company_detail(
                            row,
                            title=f"{row['Company Name']} (Similarity: {row['similarity']:.4f})"
                        ) 
                else:
                    # Cảnh báo người dùng chưa nhập gì cả
                    st.warning("⚠️ Vui lòng chọn công ty hoặc nhập từ khóa.")     

    #2.3.2. Recommendation
    elif page == "Recommendation":
        st.markdown('<h1 class="main-header">🔮 RECOMMENDATION</h1>', unsafe_allow_html=True)
        
        if not st.session_state.get('model_trained', False):
            st.warning("⚠️ Vui lòng train model ở mục 'Build Project' trước khi sử dụng tính năng này!")
            st.stop()
        
        ml_system = st.session_state.ml_system
        
        st.markdown("### 🎯 Nhập thông tin để nhận đề xuất công ty phù hợp")
        
        # User input form
        col1, col2 = st.columns(2)
        
        with col1:
            company_type = st.selectbox("🏢 Loại công ty:", ml_system.clustercty['Company Type'].unique())
            company_industry = st.selectbox("🏭 Ngành nghề:", ml_system.clustercty['Company industry'].unique())
            company_size = st.selectbox("👥 Quy mô công ty:", ml_system.clustercty['Company size'].unique())
        
        with col2:
            country = st.selectbox("🌍 Quốc gia:", ml_system.clustercty['Country'].unique())
            working_days = st.selectbox("📅 Ngày làm việc:", ml_system.clustercty['Working days'].unique())
            overtime_policy = st.selectbox("⏰ Chính sách OT:", ml_system.clustercty['Overtime Policy'].unique())
        
        # Text input
        text_input = st.text_area(
            "📝 Mô tả mong muốn của bạn:",
            placeholder="Ví dụ: Tôi muốn làm việc trong môi trường công nghệ, sử dụng AI và machine learning, có cơ hội phát triển...",
            height=100
        )
        
        # Similarity threshold
        threshold = st.slider("🎯 Ngưỡng độ tương đồng:", 0.0, 1.0, 0.1, 0.05)
        
        if st.button("🔮 Dự đoán và Đề xuất", use_container_width=True):
            if text_input.strip():
                with st.spinner("🤖 AI đang phân tích và đề xuất..."):
                    # Chuẩn bị input
                    user_input = {
                        'Company Type': company_type,
                        'Company industry': company_industry,
                        'Company size': company_size,
                        'Country': country,
                        'Working days': working_days,
                        'Overtime Policy': overtime_policy
                    }
                    
                    # Lấy đề xuất
                    matched_companies, predicted_cluster = ml_system.recommend_companies(
                        user_input, text_input, threshold
                    )
                    
                    if not matched_companies.empty:
                        # Hiển thị kết quả prediction
                        st.markdown(f"""
                        <div class="prediction-container">
                            <h3>🎯 Kết quả Dự đoán</h3>
                            <p><strong>Cluster được dự đoán:</strong> {predicted_cluster}</p>
                            <p><strong>Số công ty phù hợp:</strong> {len(matched_companies)}</p>
                            <p><strong>Ngưỡng tương đồng:</strong> {threshold}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("### 🏆 Top công ty được đề xuất:")
                        
                        # Hiển thị từng công ty
                        for idx, (_, company) in enumerate(matched_companies.iterrows(), 1):
                            similarity = company['similarity_score']
                            
                            with st.expander(f"#{idx} 🏢 {company['Company Name']} - Similarity: {similarity:.3f}"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown(f"**🏭 Ngành:** {company.get('Company industry', 'N/A')}")
                                    st.markdown(f"**👥 Quy mô:** {company.get('Company size', 'N/A')}")
                                    st.markdown(f"**🌍 Quốc gia:** {company.get('Country', 'N/A')}")
                                    st.markdown(f"**🏢 Loại:** {company.get('Company Type', 'N/A')}")
                                
                                with col2:
                                    st.markdown(f"**📅 Làm việc:** {company.get('Working days', 'N/A')}")
                                    st.markdown(f"**⏰ OT Policy:** {company.get('Overtime Policy', 'N/A')}")
                                    st.markdown(f"**😊 Sentiment:** {company.get('sentiment_group', 'N/A')}")
                                    st.markdown(f"**🔑 Keywords:** {company.get('keyword', 'N/A')}")
                                
                                # Similarity score highlight
                                st.markdown(f"""
                                <div class="similarity-score">
                                    🎯 Độ tương đồng: {similarity:.3f} ({similarity*100:.1f}%)
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Company details
                                if 'Company overview_new' in company:
                                    st.markdown("**📝 Mô tả công ty:**")
                                    st.write(company.get('Company overview_new', 'Không có thông tin'))
                                
                                if "Why you'll love working here_new" in company:
                                    st.markdown("**💝 Tại sao bạn sẽ yêu thích:**")
                                    st.write(company.get("Why you'll love working here_new", 'Không có thông tin'))
                                
                                if 'Our key skills_new' in company:
                                    st.markdown("**🔧 Kỹ năng cần thiết:**")
                                    st.write(company.get('Our key skills_new', 'Không có thông tin'))
                        
                        # Visualization
                        if len(matched_companies) > 1:
                            st.markdown("### 📊 Biểu đồ độ tương đồng:")
                            fig_similarity = px.bar(
                                x=matched_companies['Company Name'],
                                y=matched_companies['similarity_score'],
                                title="Độ tương đồng của các công ty được đề xuất",
                                labels={'x': 'Công ty', 'y': 'Độ tương đồng'}
                            )
                            fig_similarity.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig_similarity, use_container_width=True)
                    
                    else:
                        st.warning(f"❌ Không tìm thấy công ty phù hợp với ngưỡng tương đồng {threshold}")
                        st.info("💡 **Gợi ý:** Thử giảm ngưỡng tương đồng hoặc thay đổi mô tả mong muốn")
            else:
                st.warning("⚠️ Vui lòng nhập mô tả mong muốn của bạn!")
    
    
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






