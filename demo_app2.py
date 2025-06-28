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
    st.warning("⚠️ PySpark không được cài đặt. Chỉ sử dụng sklearn models.")

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
    .creator-info {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 2rem;
    }
    .prediction-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    .similarity-score {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
        color: #1976d2;
    }
    .ai-mode {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        text-align: center;
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
</style>
""", unsafe_allow_html=True)

# Khởi tạo session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Class xử lý văn bản theo code của bạn
class TextProcessor:
    def __init__(self):
        self.teen_dict = {}
        self.stopwords_lst = []
        self.wrong_lst = []
        self.load_dictionaries()
    
    def load_dictionaries(self):
        """Load các từ điển xử lý văn bản"""
        try:
            # Load teencode
            if os.path.exists('files/teencode.txt'):
                with open('files/teencode.txt', 'r', encoding="utf8") as file:
                    teen_lst = file.read().split('\n')
                    for line in teen_lst:
                        if '\t' in line:
                            key, value = line.split('\t', 1)
                            self.teen_dict[key] = str(value)
            
            # Load stopwords
            if os.path.exists('files/vietnamese-stopwords_rev.txt'):
                with open('files/vietnamese-stopwords_rev.txt', 'r', encoding="utf8") as file:
                    self.stopwords_lst = file.read().split('\n')
            
            # Load wrong words
            if os.path.exists('files/wrong-word_rev.txt'):
                with open('files/wrong-word_rev.txt', 'r', encoding="utf8") as file:
                    self.wrong_lst = file.read().split('\n')
                    
        except Exception as e:
            st.warning(f"Không thể load một số file từ điển: {e}")
    
    def clean_text(self, text):
        """Làm sạch văn bản cơ bản"""
        if not text or pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(rf"[{string.punctuation}]", "", text)
        return text
    
    def fix_teencode(self, text):
        """Sửa teencode"""
        words = text.split()
        corrected = [self.teen_dict.get(word, word) for word in words]
        return " ".join(corrected)
    
    def remove_wrongword(self, text):
        """Loại bỏ từ sai"""
        words = text.split()
        trueword = [word for word in words if word not in self.wrong_lst]
        return " ".join(trueword)
    
    def remove_stopword(self, text):
        """Loại bỏ stopwords"""
        words = text.split()
        stopword = [word for word in words if word not in self.stopwords_lst]
        return " ".join(stopword)
    
    def clean_pipeline(self, text):
        """Pipeline xử lý văn bản hoàn chỉnh theo code của bạn"""
        if not text or pd.isna(text):
            return ""
        text = self.clean_text(text)
        text = self.fix_teencode(text)
        text = self.remove_wrongword(text)
        text = self.remove_stopword(text)
        return text

# Class hệ thống ML theo code của bạn
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
        """Load và xử lý dữ liệu theo code của bạn"""
        try:
            # Đường dẫn file - có thể điều chỉnh theo cấu trúc thư mục của bạn
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
                st.error(f"Không tìm thấy các file: {missing_files}")
                return False
            
            # Load dữ liệu theo code của bạn
            clustercty = pd.read_csv(data_paths['translated_data'])
            new_data = pd.read_csv(data_paths['top2_clusters'])
            sentiment_cln = pd.read_csv(data_paths['sentiment_data'])
            
            # Merge dữ liệu theo code của bạn
            new_data = pd.merge(new_data, sentiment_cln[['Company Name','sentiment_group']], on='Company Name', how='left')
            clustercty = clustercty.merge(new_data[['Company Name', 'keyword', 'sentiment_group']], on='Company Name', how='left')
            
            # Xử lý cột không cần thiết
            if 'Unnamed: 0' in clustercty.columns:
                clustercty.drop(columns=['Unnamed: 0'], inplace=True)
            
            # Điền giá trị null
            clustercty['keyword'].fillna('không xác định', inplace=True)
            clustercty['sentiment_group'].fillna('neutral', inplace=True)
            
            _self.clustercty = clustercty
            return True
            
        except Exception as e:
            st.error(f"Lỗi load dữ liệu: {e}")
            return False
    
    def prepare_features(self):
        """Chuẩn bị features theo code của bạn"""
        try:
            # Xử lý văn bản theo code của bạn
            self.clustercty['combined_text'] = self.clustercty[self.text_cols].fillna('').agg(' '.join, axis=1)
            self.clustercty['combined_text'] = self.clustercty['combined_text'].apply(self.text_processor.clean_pipeline)
            
            # TF-IDF theo code của bạn
            df_tfidf = pd.DataFrame(
                self.tfidf.fit_transform(self.clustercty['combined_text']).toarray(),
                columns=self.tfidf.get_feature_names_out()
            )
            
            # One-hot encode theo code của bạn
            self.df_structured_encoded = pd.get_dummies(self.clustercty[self.structured_cols], drop_first=True)
            
            # Gộp dữ liệu
            X_concat = pd.concat([
                self.df_structured_encoded.reset_index(drop=True), 
                df_tfidf.reset_index(drop=True)
            ], axis=1)
            
            # PCA theo code của bạn
            self.X_full = self.pca.fit_transform(X_concat)
            
            return True
            
        except Exception as e:
            st.error(f"Lỗi chuẩn bị features: {e}")
            return False
    
    def find_optimal_clusters(self):
        """Tìm số cluster tối ưu theo code của bạn"""
        K = range(2, 11)
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
        """Training các mô hình theo code của bạn"""
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
        
            # Các mô hình theo code của bạn
            models = {
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Naive Bayes": GaussianNB(),
                "SVM": SVC(),
                "Decision Tree": DecisionTreeClassifier(),
                "KNN": KNeighborsClassifier(),
                "Gradient Boosting": GradientBoostingClassifier()
            }
        
            results = []
            trained_models = {}  #  Lưu các mô hình đã train
        
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
            
             #  Chọn và SỬ DỤNG mô hình tốt nhất
            best_model_name, best_acc = max(results, key=lambda x: x[1])
            self.best_model = trained_models[best_model_name]  #  Sử dụng mô hình tốt nhất thực sự
        
            # Lưu thông tin mô hình tốt nhất
            st.success(f"🏆 Mô hình tốt nhất: {best_model_name} với accuracy: {best_acc:.3f}")
        
            return results, best_model_name, best_acc, best_k
        
        except Exception as e:
            st.error(f"Lỗi training models: {e}")
            return [], "", 0, 0
    
    def recommend_companies(self, user_input, text_input, threshold=0.1):
        """Đề xuất công ty theo code của bạn"""
        try:
            # Xử lý text input theo code của bạn
            cleaned_text = self.text_processor.clean_pipeline(text_input)
            tfidf_vec = self.tfidf.transform([cleaned_text])
            
            # Xử lý structured input theo code của bạn
            structured_df = pd.DataFrame([user_input])
            structured_encoded = pd.get_dummies(structured_df)
            
            # Đảm bảo có đủ columns như trong code của bạn
            missing_cols = set(self.df_structured_encoded.columns) - set(structured_encoded.columns)
            for col in missing_cols:
                structured_encoded[col] = 0
            structured_encoded = structured_encoded[self.df_structured_encoded.columns]
            
            # Gộp features theo code của bạn
            user_input_vector = pd.concat([
                structured_encoded.reset_index(drop=True), 
                pd.DataFrame(tfidf_vec.toarray(), columns=self.tfidf.get_feature_names_out())
            ], axis=1)
            
            # PCA transform theo code của bạn
            user_input_pca = self.pca.transform(user_input_vector)
            
            # Predict cluster theo code của bạn
            predicted_cluster = self.best_model.predict(user_input_pca)[0]
            
            # Tính Cosine Similarity theo code của bạn
            company_text_vectors = self.tfidf.transform(self.clustercty['combined_text'])
            similarity_scores = cosine_similarity(tfidf_vec, company_text_vectors).flatten()
            self.clustercty['similarity_score'] = similarity_scores
            
            # Lọc công ty theo code của bạn
            matched = self.clustercty[
                (self.clustercty['cluster'] == predicted_cluster) & 
                (self.clustercty['similarity_score'] >= threshold)
            ].copy()
            
            matched = matched.sort_values(by='similarity_score', ascending=False).head(10)
            
            return matched, predicted_cluster
            
        except Exception as e:
            st.error(f"Lỗi đề xuất: {e}")
            return pd.DataFrame(), -1

class PySparkMLSystem:
    def __init__(self):
        self.spark = None
        self.spark_df_ml = None
        self.pyspark_results = {}
        
    def initialize_spark(self):
        """Khởi tạo Spark Session với error handling tốt hơn"""
        try:
            # Kiểm tra Java environment
            import os
            java_home = os.environ.get('JAVA_HOME')
            if not java_home:
                st.warning("⚠️ JAVA_HOME không được set. Đang thử khởi tạo Spark...")
        
            # Cấu hình Spark với các settings an toàn hơn
            from pyspark.sql import SparkSession
        
            self.spark = SparkSession.builder \
                .appName("CompanyRecommendation") \
                .config("spark.driver.memory", "1g") \
                .config("spark.executor.memory", "1g") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.driver.host", "localhost") \
                .getOrCreate()
        
            # Test Spark hoạt động
            test_df = self.spark.createDataFrame([(1, "test")], ["id", "value"])
            test_df.count()  # Test operation
        
            # Giảm log level
            self.spark.sparkContext.setLogLevel("ERROR")
        
            st.success(" Spark đã được khởi tạo thành công!")
            return True
        
        except Exception as e:
            error_msg = str(e)
            if "JavaPackage" in error_msg:
                st.error("""
             **Lỗi Java/Spark Environment**
            
            **Nguyên nhân có thể:**
            - Java không được cài đặt hoặc JAVA_HOME không đúng
            - PySpark không tương thích với Java version
            - Spark không được cấu hình đúng
            
            **Giải pháp:**
            1. Cài đặt Java 8 hoặc 11: `sudo apt install openjdk-11-jdk`
            2. Set JAVA_HOME: `export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64`
            3. Restart terminal và thử lại
            
            **Hoặc chỉ sử dụng Sklearn models để tiếp tục.**
            """)
            else:
                st.error(f" Lỗi khởi tạo Spark: {error_msg}")
        
            return False
    
    def prepare_spark_data(self, clustercty, X_concat, cluster_labels):
        """Chuẩn bị dữ liệu cho PySpark"""
        try:
            # Schema cho Spark DataFrame
            schema = StructType([
                StructField("id", IntegerType(), True),
                StructField("Company Name", StringType(), True),
                StructField("Company Type", StringType(), True),
                StructField("Company industry", StringType(), True),
                StructField("Company size", StringType(), True),
                StructField("Country", StringType(), True),
                StructField("Working days", StringType(), True),
                StructField("Overtime Policy", StringType(), True),
                StructField("Company overview_new", StringType(), True),
                StructField("Why you'll love working here_new", StringType(), True),
                StructField("Our key skills_new", StringType(), True),
                StructField("keyword", StringType(), True),
                StructField("sentiment_group", StringType(), True),
                StructField("combined_text", StringType(), True),
                StructField("cluster", IntegerType(), True)
            ])
            
            # Thêm cluster labels vào X_concat
            X_concat_with_labels = X_concat.copy()
            X_concat_with_labels['cluster'] = cluster_labels
            
            # Convert to Spark DataFrame
            self.spark_df_ml = self.spark.createDataFrame(X_concat_with_labels)
            
            # Assemble features
            feature_columns = X_concat.columns.tolist()
            assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
            self.spark_df_ml = assembler.transform(self.spark_df_ml)
            
            # Index labels
            indexer = StringIndexer(inputCol="cluster", outputCol="label")
            self.spark_df_ml = indexer.fit(self.spark_df_ml).transform(self.spark_df_ml)
            
            # Select features and labels
            self.spark_df_ml = self.spark_df_ml.select("features", "label")
            
            return True
            
        except Exception as e:
            st.error(f"Lỗi chuẩn bị dữ liệu Spark: {e}")
            return False
    
    def train_pyspark_models(self):
        """Training PySpark models"""
        try:
            # Split data
            train_data, test_data = self.spark_df_ml.randomSplit([0.8, 0.2], seed=42)
            
            # Evaluator
            evaluator = MulticlassClassificationEvaluator(
                labelCol="label", 
                predictionCol="prediction", 
                metricName="accuracy"
            )
            
            # Logistic Regression
            lr = SparkLogisticRegression(featuresCol="features", labelCol="label")
            lr_model = lr.fit(train_data)
            lr_predictions = lr_model.transform(test_data)
            lr_accuracy = evaluator.evaluate(lr_predictions)
            
            # Decision Tree
            dt = SparkDecisionTreeClassifier(featuresCol="features", labelCol="label")
            dt_model = dt.fit(train_data)
            dt_predictions = dt_model.transform(test_data)
            dt_accuracy = evaluator.evaluate(dt_predictions)
            
            self.pyspark_results = {
                "PySpark Logistic Regression": lr_accuracy,
                "PySpark Decision Tree": dt_accuracy
            }
            
            return self.pyspark_results
            
        except Exception as e:
            st.error(f"Lỗi training PySpark models: {e}")
            return {}
    
    def stop_spark(self):
        """Dừng Spark Session"""
        if self.spark:
            self.spark.stop()

# Dữ liệu mẫu cho Company Suggest (giữ nguyên)
companies_data = {
    "Google": {
        "overview": "Google là một trong những công ty công nghệ hàng đầu thế giới, chuyên về tìm kiếm trực tuyến, điện toán đám mây và công nghệ quảng cáo.",
        "industry": "Technology",
        "size": "Large (100,000+ employees)",
        "country": "USA",
        "similar": ["Microsoft", "Apple", "Amazon"]
    },
    "Microsoft": {
        "overview": "Microsoft là công ty phần mềm đa quốc gia, nổi tiếng với hệ điều hành Windows và bộ ứng dụng Office.",
        "industry": "Technology", 
        "size": "Large (100,000+ employees)",
        "country": "USA",
        "similar": ["Google", "Apple", "IBM"]
    },
    "Apple": {
        "overview": "Apple là công ty công nghệ thiết kế và sản xuất các sản phẩm điện tử tiêu dùng, phần mềm máy tính và dịch vụ trực tuyến.",
        "industry": "Technology",
        "size": "Large (100,000+ employees)", 
        "country": "USA",
        "similar": ["Google", "Microsoft", "Samsung"]
    },
    "Amazon": {
        "overview": "Amazon là công ty thương mại điện tử và điện toán đám mây đa quốc gia có trụ sở tại Seattle, Washington.",
        "industry": "E-commerce/Cloud",
        "size": "Large (1,000,000+ employees)",
        "country": "USA", 
        "similar": ["Google", "Microsoft", "Alibaba"]
    }
}

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
    
    # Select box chọn công ty
    selected_company = st.selectbox("🔍 Chọn công ty để xem thông tin:", list(companies_data.keys()))
    
    if selected_company:
        company_info = companies_data[selected_company]
        
        # Hiển thị thông tin công ty được chọn
        st.markdown(f'<div class="section-header">📋 Thông tin về {selected_company}</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="company-card">
            <h4>🏢 {selected_company}</h4>
            <p><strong>Tổng quan:</strong> {company_info['overview']}</p>
            <p><strong>Ngành:</strong> {company_info['industry']}</p>
            <p><strong>Quy mô:</strong> {company_info['size']}</p>
            <p><strong>Quốc gia:</strong> {company_info['country']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Hiển thị các công ty tương tự
        st.markdown('<div class="section-header">🔗 Các công ty tương tự</div>', unsafe_allow_html=True)
        
        cols = st.columns(len(company_info['similar']))
        
        for idx, similar_company in enumerate(company_info['similar']):
            with cols[idx]:
                st.markdown(f"**{similar_company}**")
                
                # Expander cho thông tin chi tiết
                with st.expander(f"Xem thông tin {similar_company}"):
                    if similar_company in companies_data:
                        similar_info = companies_data[similar_company]
                        st.write(f"**Tổng quan:** {similar_info['overview']}")
                        st.write(f"**Ngành:** {similar_info['industry']}")
                        st.write(f"**Quy mô:** {similar_info['size']}")
                        st.write(f"**Quốc gia:** {similar_info['country']}")

elif selected_menu == "Build Project":
    st.markdown('<h1 class="main-header">🔨 Build Project</h1>', unsafe_allow_html=True)
    
    # Khởi tạo và load dữ liệu
    if not st.session_state.data_loaded:
        with st.spinner("🔄 Đang load dữ liệu..."):
            ml_system = CompanyRecommendationSystem()
            if ml_system.load_data():
                st.session_state.ml_system = ml_system
                st.session_state.data_loaded = True
                st.success("✅ Dữ liệu đã được load thành công!")
            else:
                st.error("❌ Không thể load dữ liệu. Vui lòng kiểm tra file dữ liệu.")
                st.stop()
    
    # Tùy chọn training
    st.markdown("### 🚀 Chọn phương thức Training:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sklearn_training = st.button("🔬 Train Sklearn Models", use_container_width=True)
    
    with col2:
        if PYSPARK_AVAILABLE:
            pyspark_training = st.button("⚡ Train PySpark Models", use_container_width=True)
        else:
            st.button("⚡ PySpark Not Available", disabled=True, use_container_width=True)
    
    # Sklearn Training
    if sklearn_training and st.session_state.data_loaded:
        if not st.session_state.model_trained:
            with st.spinner("🤖 Đang chuẩn bị features và training sklearn models..."):
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
    
   # PySpark Training
    if PYSPARK_AVAILABLE and 'pyspark_training' in locals() and pyspark_training:
        if not st.session_state.get('pyspark_trained', False):
            with st.spinner("⚡ Đang khởi tạo Spark và training PySpark models..."):
                # Cần sklearn models trước
                if not st.session_state.model_trained:
                    st.warning("⚠️ Vui lòng train sklearn models trước!")
                else:
                    try:
                        ml_system = st.session_state.ml_system
                        pyspark_system = PySparkMLSystem()
                    
                        if pyspark_system.initialize_spark():
                            # Chuẩn bị dữ liệu từ sklearn pipeline
                            X_concat = pd.concat([
                                ml_system.df_structured_encoded.reset_index(drop=True),
                                pd.DataFrame(
                                    ml_system.tfidf.transform(ml_system.clustercty['combined_text']).toarray(),
                                    columns=ml_system.tfidf.get_feature_names_out()
                                )
                            ], axis=1)
                        
                            cluster_labels = ml_system.clustercty['cluster'].values
                        
                            if pyspark_system.prepare_spark_data(ml_system.clustercty, X_concat, cluster_labels):
                                pyspark_results = pyspark_system.train_pyspark_models()
                            
                                if pyspark_results:
                                    st.session_state.pyspark_trained = True
                                    st.session_state.pyspark_results = pyspark_results
                                    st.session_state.pyspark_system = pyspark_system
                                    st.success("✅ PySpark Training hoàn tất!")
                                    st.rerun()
                        
                            pyspark_system.stop_spark()
                        else:
                            st.info("💡 **Không thể khởi tạo PySpark.** Bạn vẫn có thể sử dụng Sklearn models để tiếp tục dự án!")
                            
                    except Exception as e:
                        st.error(f"❌ Lỗi trong quá trình PySpark training: {e}")
                        st.info("💡 Hãy sử dụng Sklearn models để tiếp tục!")
    
    # Hiển thị kết quả so sánh
    if st.session_state.get('model_trained', False):
        st.markdown("""
        <div class="section-header">📊 Kết quả nghiên cứu và so sánh mô hình</div>
        
        ### 🧪 Phương pháp nghiên cứu:
        
        #### 1. Thu thập dữ liệu
        - **Nguồn dữ liệu**: IT_Viec
        
        #### 2. Tiền xử lý dữ liệu
        - Làm sạch dữ liệu thiếu và bất thường
        - Xử lý văn bản tiếng Việt (teencode, stopwords, wrong words)
        - TF-IDF Vectorization với 500 features
        - One-hot encoding cho dữ liệu có cấu trúc
        - PCA giảm chiều xuống 50 components
        
        #### 3. So sánh Sklearn vs PySpark
        - **Sklearn**: Phù hợp cho datasets nhỏ-trung bình, dễ sử dụng
        - **PySpark**: Scalable cho big data, distributed computing
        - **Performance**: So sánh accuracy và thời gian training
        """, unsafe_allow_html=True)
        
        # Metrics overview
        ml_system = st.session_state.ml_system
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Số công ty", len(ml_system.clustercty))
        with col2:
            st.metric("🎯 Số clusters", st.session_state.best_k)
        with col3:
            st.metric("🏆 Best Sklearn", st.session_state.best_model_name)
        with col4:
            st.metric("📈 Best Accuracy", f"{st.session_state.best_acc:.3f}")
        
        # Sklearn Results
        st.markdown("### 🔬 Sklearn Models Results:")
        sklearn_results_df = pd.DataFrame(st.session_state.training_results, columns=['Mô hình', 'Accuracy'])
        sklearn_results_df['Accuracy (%)'] = (sklearn_results_df['Accuracy'] * 100).round(2)
        sklearn_results_df['Framework'] = 'Sklearn'
        
        st.dataframe(sklearn_results_df[['Mô hình', 'Accuracy (%)', 'Framework']], use_container_width=True)
        
        # PySpark Results (if available)
        if st.session_state.get('pyspark_trained', False):
            st.markdown("### ⚡ PySpark Models Results:")
            pyspark_results = st.session_state.pyspark_results
            pyspark_results_df = pd.DataFrame(list(pyspark_results.items()), columns=['Mô hình', 'Accuracy'])
            pyspark_results_df['Accuracy (%)'] = (pyspark_results_df['Accuracy'] * 100).round(2)
            pyspark_results_df['Framework'] = 'PySpark'
            
            st.dataframe(pyspark_results_df[['Mô hình', 'Accuracy (%)', 'Framework']], use_container_width=True)
            
            # Combined comparison
            st.markdown("### 🆚 Sklearn vs PySpark Comparison:")
            
            # Combine results for comparison
            combined_results = pd.concat([
                sklearn_results_df[['Mô hình', 'Accuracy (%)', 'Framework']],
                pyspark_results_df[['Mô hình', 'Accuracy (%)', 'Framework']]
            ], ignore_index=True)
            
            # Comparison chart
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
            
            # Performance analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔬 Sklearn Advantages:")
                st.markdown("""
                - ✅ Dễ sử dụng và debug
                - ✅ Rich ecosystem và documentation
                - ✅ Phù hợp cho prototyping
                - ✅ Nhiều algorithms có sẵn
                - ✅ Tích hợp tốt với pandas/numpy
                """)
            
            with col2:
                st.markdown("#### ⚡ PySpark Advantages:")
                st.markdown("""
                - ✅ Scalable cho big data
                - ✅ Distributed computing
                - ✅ Memory optimization
                - ✅ Fault tolerance
                - ✅ Integration với Hadoop ecosystem
                """)
            
            # Best model comparison
            best_sklearn = sklearn_results_df.loc[sklearn_results_df['Accuracy (%)'].idxmax()]
            best_pyspark = pyspark_results_df.loc[pyspark_results_df['Accuracy (%)'].idxmax()]
            
            st.markdown("### 🏆 Best Models Comparison:")
            
            comparison_metrics = pd.DataFrame({
                'Metric': ['Best Model', 'Accuracy (%)', 'Framework'],
                'Sklearn': [best_sklearn['Mô hình'], best_sklearn['Accuracy (%)'], 'Sklearn'],
                'PySpark': [best_pyspark['Mô hình'], best_pyspark['Accuracy (%)'], 'PySpark']
            })
            
            st.dataframe(comparison_metrics, use_container_width=True)
            
        else:
            st.info("💡 Train PySpark models để xem so sánh chi tiết!")
        
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
        st.markdown("### 🎯 Phân tích Clusters:")
        cluster_analysis = ml_system.clustercty.groupby('cluster').agg({
            'Company Name': 'count',
            'Company industry': lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A',
            'Company size': lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A',
            'sentiment_group': lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A'
        }).round(2)
        
        cluster_analysis.columns = ['Số công ty', 'Ngành chủ đạo', 'Quy mô chủ đạo', 'Sentiment chủ đạo']
        st.dataframe(cluster_analysis, use_container_width=True)
        
        # Final conclusions
        conclusion_text = f"""
        ### 🏆 Kết luận:
        - **Sklearn Best**: {st.session_state.best_model_name} với accuracy {st.session_state.best_acc:.3f}
        - **Số clusters tối ưu**: {st.session_state.best_k} clusters
        - **Tổng số công ty**: {len(ml_system.clustercty)} công ty
        - **Features**: {ml_system.X_full.shape[1]} features sau PCA
        """
        
        if st.session_state.get('pyspark_trained', False):
            pyspark_best = max(st.session_state.pyspark_results.items(), key=lambda x: x[1])
            conclusion_text += f"""
        - **PySpark Best**: {pyspark_best[0]} với accuracy {pyspark_best[1]:.3f}
        - **Framework Winner**: {'Sklearn' if st.session_state.best_acc > pyspark_best[1] else 'PySpark'}
            """
        
        conclusion_text += """
        
        ### 🔍 Insights quan trọng:
        1. **Text processing** đóng vai trò quan trọng trong việc phân loại công ty
        2. **Clustering** giúp nhóm các công ty có đặc điểm tương đồng
        3. **Sentiment analysis** cung cấp thông tin về văn hóa công ty
        4. **Framework choice** phụ thuộc vào kích thước dữ liệu và yêu cầu scalability
        5. **Sklearn** phù hợp cho datasets nhỏ-trung bình với độ chính xác cao
        6. **PySpark** cần thiết khi scale lên big data và distributed computing
        """
        
        st.markdown(conclusion_text, unsafe_allow_html=True)
    
    else:
        st.info("👆 Nhấn nút 'Train Sklearn Models' để bắt đầu phân tích!")

elif selected_menu == "New Prediction":
    st.markdown('<h1 class="main-header">🤖 AI-Powered Company Recommendation</h1>', unsafe_allow_html=True)
    
    # Kiểm tra dữ liệu và model đã sẵn sàng
    if not st.session_state.data_loaded or not st.session_state.model_trained:
        st.warning("⚠️ Vui lòng vào mục 'Build Project' để load dữ liệu và train model trước!")
        st.stop()
    
    ml_system = st.session_state.ml_system
    
    st.success("✅ Hệ thống AI đã sẵn sàng với dữ liệu thực!")
    
    # Form nhập liệu
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
    st.markdown("### 🤖 Nhập thông tin để nhận đề xuất từ AI")
    
    with st.form("ai_prediction_form"):
        col1, col2 = st.columns(2)
        
        # Lấy options từ dữ liệu thực
        with col1:
            company_type_options = sorted(ml_system.clustercty['Company Type'].dropna().unique().tolist())
            company_type = st.selectbox("🏢 Company Type:", company_type_options)
            
            industry_options = sorted(ml_system.clustercty['Company industry'].dropna().unique().tolist())
            company_industry = st.selectbox("🏭 Company Industry:", industry_options)
            
            size_options = sorted(ml_system.clustercty['Company size'].dropna().unique().tolist())
            company_size = st.selectbox("👥 Company Size:", size_options)
        
        with col2:
            country_options = sorted(ml_system.clustercty['Country'].dropna().unique().tolist())
            country = st.selectbox("🌍 Country:", country_options)
            
            working_days_options = sorted(ml_system.clustercty['Working days'].dropna().unique().tolist())
            working_days = st.selectbox("📅 Working Days:", working_days_options)
            
            overtime_options = sorted(ml_system.clustercty['Overtime Policy'].dropna().unique().tolist())
            overtime_policy = st.selectbox("⏰ Overtime Policy:", overtime_options)
        
        # Text input cho mong muốn
        st.markdown("### 💭 Mô tả mong muốn của bạn:")
        user_expectations = st.text_area(
            "Hãy chia sẻ chi tiết về công ty và công việc lý tưởng:",
            placeholder="Ví dụ: Tôi muốn làm việc trong môi trường công nghệ năng động, có cơ hội học AI/ML, lương cao, work-life balance tốt, đội ngũ trẻ trung sáng tạo...",
            height=120
        )
        
        # Threshold slider
        threshold = st.slider(
            "🎯 Độ tương đồng tối thiểu:",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Điều chỉnh để lọc các công ty phù hợp hơn"
        )
        
        submitted = st.form_submit_button("🚀 Tìm kiếm với AI", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Xử lý khi form được submit
    if submitted:
        if user_expectations.strip():
            with st.spinner("🤖 AI đang phân tích và tìm kiếm..."):
                user_input = {
                    'Company Type': company_type,
                    'Company industry': company_industry,
                    'Company size': company_size,
                    'Country': country,
                    'Working days': working_days,
                    'Overtime Policy': overtime_policy
                }
                
                recommendations, predicted_cluster = ml_system.recommend_companies(
                    user_input, user_expectations, threshold
                )
            
            if not recommendations.empty:
                st.markdown("## 🎯 Kết quả từ AI")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 AI Cluster", f"Cluster {predicted_cluster}")
                with col2:
                    st.metric("📊 Công ty phù hợp", len(recommendations))
                with col3:
                    avg_similarity = recommendations['similarity_score'].mean()
                    st.metric("📈 Độ tương đồng TB", f"{avg_similarity:.3f}")
                with col4:
                    max_similarity = recommendations['similarity_score'].max()
                    st.metric("🏆 Điểm cao nhất", f"{max_similarity:.3f}")
                
                # Visualization
                if len(recommendations) > 1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_hist = px.histogram(
                            recommendations, 
                            x='similarity_score',
                            title="📊 Phân bố điểm tương đồng AI",
                            labels={'similarity_score': 'Điểm tương đồng', 'count': 'Số lượng'},
                            color_discrete_sequence=['#1f77b4']
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col2:
                        fig_scatter = px.scatter(
                            recommendations.head(10), 
                            x=range(len(recommendations.head(10))),
                            y='similarity_score',
                            hover_data=['Company Name'],
                            title="📈 Điểm tương đồng theo thứ hạng",
                            labels={'x': 'Thứ hạng', 'similarity_score': 'Điểm tương đồng'}
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Detailed recommendations
                st.markdown("### 🏆 Top công ty được AI đề xuất:")
                
                for idx, (_, company) in enumerate(recommendations.head(8).iterrows()):
                    with st.expander(f"🏢 {company['Company Name']} - AI Score: {company['similarity_score']:.3f}"):
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
                        
                        # Company description từ dữ liệu thực
                        if 'Company overview_new' in company:
                            st.markdown("**📝 Mô tả công ty:**")
                            st.write(company.get('Company overview_new', 'Không có thông tin'))
                        
                        if "Why you'll love working here_new" in company:
                            st.markdown("**💝 Tại sao bạn sẽ yêu thích làm việc ở đây:**")
                            st.write(company.get("Why you'll love working here_new", 'Không có thông tin'))
                        
                        # AI Score visualization
                        score = company['similarity_score']
                        if score > 0.3:
                            color, level = "green", "Rất phù hợp"
                        elif score > 0.15:
                            color, level = "orange", "Phù hợp"
                        else:
                            color, level = "blue", "Có thể phù hợp"
                            
                        st.markdown(f"""
                        <div style="background: linear-gradient(90deg, {color}20, {color}40); 
                                    padding: 0.8rem; border-radius: 8px; border-left: 4px solid {color};">
                            🤖 AI Confidence: {score:.3f} ({score*100:.1f}%) - <strong>{level}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Action buttons
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button(f"📧 Liên hệ", key=f"ai_contact_{idx}"):
                                st.success("✅ Đã lưu thông tin liên hệ!")
                        with col2:
                            if st.button(f"💾 Lưu", key=f"ai_save_{idx}"):
                                st.success("✅ Đã thêm vào danh sách yêu thích!")
                        with col3:
                            if st.button(f"📊 Phân tích", key=f"ai_analyze_{idx}"):
                                st.info("🔍 Tính năng phân tích chi tiết đang phát triển!")
                
                # Summary table
                st.markdown("### 📊 Bảng tổng hợp AI:")
                display_columns = ['Company Name', 'Company industry', 'Company size', 'Country', 'similarity_score', 'sentiment_group']
                available_columns = [col for col in display_columns if col in recommendations.columns]
                
                display_df = recommendations[available_columns].copy()
                display_df['similarity_score'] = display_df['similarity_score'].round(3)
                display_df = display_df.rename(columns={
                    'Company Name': 'Tên công ty',
                    'Company industry': 'Ngành',
                    'Company size': 'Quy mô',
                    'Country': 'Quốc gia',
                    'similarity_score': 'Điểm AI',
                    'sentiment_group': 'Sentiment'
                })
                
                st.dataframe(
                    display_df.style.background_gradient(subset=['Điểm AI']),
                    use_container_width=True
                )
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    csv = display_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 Tải xuống CSV",
                        data=csv,
                        file_name=f"ai_recommendations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                with col2:
                    json_data = display_df.to_json(orient='records', force_ascii=False)
                    st.download_button(
                        label="📥 Tải xuống JSON",
                        data=json_data,
                        file_name=f"ai_recommendations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
            else:
                st.markdown("""
                <div class="warning-box">
                    ⚠️ <strong>AI không tìm thấy công ty phù hợp!</strong><br>
                    Hãy thử:
                    <ul>
                        <li>Giảm độ tương đồng tối thiểu</li>
                        <li>Mở rộng hoặc thay đổi mô tả mong muốn</li>
                        <li>Thử các tiêu chí lựa chọn khác</li>
                        <li>Sử dụng từ khóa đơn giản hơn</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("❌ Vui lòng nhập mô tả mong muốn để AI có thể phân tích!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Trung tâm Tin Học - Trường Đại Học Khoa Học Tự Nhiên | Tri Vo and Thao Pham <br>"
    "</div>", 
    unsafe_allow_html=True
)
