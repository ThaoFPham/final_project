import os
import re
import json
import string
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier as SparkDecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Set Java Environment
os.environ["JAVA_HOME"] = r"C:/Program Files/Java/jdk-11"
os.environ["PYSPARK_PYTHON"] = os.path.abspath(os.path.join(os.getcwd(), ".venv/Scripts/python.exe"))
os.environ["PATH"] += os.pathsep + os.path.join(os.environ["JAVA_HOME"], "bin")

SAVE_DIR = Path("saved_model")
SAVE_DIR.mkdir(exist_ok=True)

# ---------------------- Text Processor -------------------------
class TextProcessor:
    def __init__(self):
        self.teen_dict = {}
        self.stopwords_lst = []
        self.wrong_lst = []
        self.load_dictionaries()

    def load_dictionaries(self):
        with open("files/teencode.txt", "r", encoding="utf8") as f:
            for line in f:
                if '\t' in line:
                    key, value = line.strip().split('\t', 1)
                    self.teen_dict[key] = value
        with open("files/vietnamese-stopwords_rev.txt", "r", encoding="utf8") as f:
            self.stopwords_lst = [line.strip() for line in f if line.strip()]
        with open("files/wrong-word_rev.txt", "r", encoding="utf8") as f:
            self.wrong_lst = [line.strip() for line in f if line.strip()]

    def clean_pipeline(self, text):
        if not text or pd.isna(text): return ""
        text = text.lower()
        text = re.sub(rf"[{string.punctuation}]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = " ".join([self.teen_dict.get(w, w) for w in text.split()])
        text = " ".join([w for w in text.split() if w not in self.stopwords_lst])
        text = " ".join([w for w in text.split() if w not in self.wrong_lst])
        return text

# ---------------------- Save Helper -------------------------
def save_pickle(obj, name):
    with open(SAVE_DIR / name, "wb") as f:
        pickle.dump(obj, f)

# ---------------------- Main Pipeline -------------------------
def main():
    print("📦 Loading data...")
    processor = TextProcessor()
    df = pd.read_csv("data/translated_data.csv")
    clusters = pd.read_csv("data/top2_clusters_per_company.csv")
    sentiment = pd.read_csv("data/sentiment_by_company.csv")

    clusters = pd.merge(clusters, sentiment[["Company Name", "sentiment_group"]], on="Company Name", how="left")
    df = pd.merge(df, clusters[["Company Name", "keyword", "sentiment_group"]], on="Company Name", how="left")
    df["keyword"].fillna("khong xac dinh", inplace=True)
    df["sentiment_group"].fillna("neutral", inplace=True)

    text_cols = ["Company overview_new", "Why you'll love working here_new", "Our key skills_new", "keyword"]
    structured_cols = ["Company Type", "Company industry", "Company size", "Country", "Working days", "Overtime Policy"]

    print("🧹 Cleaning text...")
    df["combined_text"] = df[text_cols].fillna("").agg(" ".join, axis=1).apply(processor.clean_pipeline)

    print("🧠 Vectorizing features...")
    tfidf = TfidfVectorizer(max_features=500)
    X_text = tfidf.fit_transform(df["combined_text"])
    df_tfidf = pd.DataFrame(X_text.toarray(), columns=tfidf.get_feature_names_out())

    df_structured = pd.get_dummies(df[structured_cols], drop_first=True)
    X_concat = pd.concat([df_structured.reset_index(drop=True), df_tfidf.reset_index(drop=True)], axis=1)

    pca = PCA(n_components=min(50, X_concat.shape[1], X_concat.shape[0] - 1), random_state=42)
    X_pca = pca.fit_transform(X_concat)

    print("📊 Clustering...")
    kmeans = KMeans(n_clusters=5, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_pca)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, df["cluster"], test_size=0.2, random_state=42)

    print("🤖 Training Sklearn models...")
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(n_neighbors=3)
    }

    results = {}
    best_model = None
    best_name = ""
    best_acc = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = acc
        print(f"✅ {name}: {acc:.3f}")
        if acc > best_acc:
            best_model, best_name, best_acc = model, name, acc

    print("⚡ Training PySpark models...")
    spark = SparkSession.builder.appName("CompanyRecommendation").getOrCreate()

    # Gắn nhãn vào dataframe
    X_concat["label"] = df["cluster"].astype("int")
    spark_df = spark.createDataFrame(X_concat)

    assembler = VectorAssembler(
        inputCols=[col for col in X_concat.columns if col != "label"],
        outputCol="features"
    )
    spark_df = assembler.transform(spark_df).select("features", "label")

    train_data, test_data = spark_df.randomSplit([0.8, 0.2], seed=42)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

    lr_model = SparkLogisticRegression(featuresCol="features", labelCol="label").fit(train_data)
    lr_acc = evaluator.evaluate(lr_model.transform(test_data))
    print(f"🔥 PySpark Logistic Regression: {lr_acc:.3f}")

    dt_model = SparkDecisionTreeClassifier(featuresCol="features", labelCol="label").fit(train_data)
    dt_acc = evaluator.evaluate(dt_model.transform(test_data))
    print(f"🌳 PySpark Decision Tree: {dt_acc:.3f}")

    # ✅ Save everything
    print("💾 Saving results...")
    save_pickle(best_model, "best_model.pkl")
    save_pickle(tfidf, "tfidf.pkl")
    save_pickle(pca, "pca.pkl")
    save_pickle(df_structured.columns.tolist(), "encoded_columns.pkl")
    save_pickle(df, "cluster_data.pkl")

    pd.DataFrame.from_dict(results, orient="index", columns=["Sklearn Accuracy"]).to_csv(SAVE_DIR / "sklearn_results.csv")

    pyspark_results = {
        "PySpark Logistic Regression": lr_acc,
        "PySpark Decision Tree": dt_acc
    }
    with open(SAVE_DIR / "pyspark_results.json", "w", encoding="utf8") as f:
        json.dump(pyspark_results, f, indent=4)

    with open(SAVE_DIR / "training_meta.json", "w", encoding="utf8") as f:
        json.dump({
            "best_model": best_name,
            "best_accuracy": round(best_acc, 4),
            "sklearn_models": list(results.keys()),
            "pyspark_models": list(pyspark_results.keys())
        }, f, indent=4)

    #  Tổng hợp kết quả Sklearn + PySpark
    all_model_names = list(results.keys()) + list(pyspark_results.keys())
    all_model_scores = [acc * 100 for acc in results.values()] + [acc * 100 for acc in pyspark_results.values()]
    model_sources = ["Sklearn"] * len(results) + ["PySpark"] * len(pyspark_results)

    #  Biểu đồ cột so sánh mô hình
    plt.figure(figsize=(10, 6))
    colors = ["cornflowerblue" if src == "Sklearn" else "orange" for src in model_sources]
    plt.bar(all_model_names, all_model_scores, color=colors)
    plt.title("🎯 So sánh Accuracy giữa các mô hình Sklearn & PySpark")
    plt.xlabel("Mô hình")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 110)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "comparison_all_models.png")
    plt.close()

    #  Biểu đồ tròn phân cụm KMeans
    cluster_counts = df["cluster"].value_counts().sort_index()
    labels = [f"Cluster {i}" for i in cluster_counts.index]
    sizes = cluster_counts.values

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=plt.cm.Set3.colors)
    plt.title(" Phân bố số lượng công ty theo cụm (KMeans)")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "cluster_distribution_pie.png")
    plt.close()
    spark.stop()
    print("✅ Done! All models and results saved in /saved_model")

if __name__ == "__main__":
    main()
