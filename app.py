from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load TF-IDF vectorizer từ notebook
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Dữ liệu điều luật (ví dụ)
ARTICLES = {
    132: {
        "title": "Compensation for Damage",
        "content": "Individuals who cause damage must compensate according to civil law."
    },
    8: {
        "title": "Civil Rights Establishment",
        "content": "Civil rights arise from legal acts, lawful conduct, or court judgments."
    },
    584: {
        "title": "Damage Liability",
        "content": "Any person causing damage must compensate unless otherwise provided by law."
    }
}

# Vector hóa điều luật
article_texts = [v["content"] for v in ARTICLES.values()]
article_vectors = vectorizer.transform(article_texts)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data["query"]

    # Vector hóa tình huống pháp lý
    q_vec = vectorizer.transform([query])

    # So khớp cosine similarity
    sims = cosine_similarity(q_vec, article_vectors)[0]
    top_idx = sims.argsort()[-3:][::-1]

    keys = list(ARTICLES.keys())
    results = []

    for i in top_idx:
        aid = keys[i]
        results.append(f"Article {aid}. {ARTICLES[aid]['title']}")

    return jsonify({
        "analysis": "Các điều luật có thể áp dụng cho tình huống này:",
        "references": results
    })

if __name__ == "__main__":
    app.run()
