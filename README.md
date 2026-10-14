# Sentix AI: Offline Sentiment Analysis System 🧠

An enterprise-grade, 100% offline web application that automatically classifies user text into Positive, Negative, or Neutral emotional states. Built to guarantee data sovereignty, this system utilizes a lightweight Scikit-Learn machine learning pipeline embedded directly within a Python Flask backend.

## 🚀 Key Features

* **Zero Cloud Dependency:** Runs entirely on local hardware with sub-second inference speeds, eliminating external API costs and network latency.
* **Strict Categorization Heuristic:** Implements a custom mathematical safeguard that actively overrides predictions to 'Neutral' if the maximum confidence score falls below 40%, preventing AI hallucinations on sarcastic or ambiguous text.
* **Memory Guardrails:** Enforces a strict 500-character input boundary via HTML5 and server-side routing to prevent RAM saturation during high-dimensional TF-IDF vectorization.
* **Client-Side Optimization:** Features a dynamic DOM-based history dashboard with real-time search and 5-item pagination, completely eliminating server-side database query latency.
* **Persistent User State:** Seamlessly saves UI preferences (like Global Dark Mode) to browser `localStorage` and logs analytical history to a local SQLite database.

## 🛠️ Technology Stack

* **Backend:** Python 3, Flask WSGI
* **Machine Learning:** Scikit-Learn (TF-IDF Feature Extraction, Multiclass Logistic Regression)
* **Database:** SQLite3, SQLAlchemy ORM
* **Frontend:** HTML5, Tailwind CSS, Vanilla JavaScript (ES6+)

## ⚙️ Installation and Setup

Since this system is designed for isolated, offline execution, setting it up locally is straightforward.
