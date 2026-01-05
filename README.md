## 🛡️ Hybrid AI Fraud Guardian: ML-Guardian + RAG-Reasoning
## 🚀 Project Overview
Traditional fraud detection systems often struggle with the "Grey Zone"—ambiguous transactions where the model probability falls between 0.3 and 0.7. This project implements a Hybrid AI Architecture that combines the statistical speed of XGBoost with the semantic reasoning of a Retrieval-Augmented Generation (RAG) Agent.

By offloading uncertain cases to a specialized reasoning layer, this system achieved a Precision lift from 71% to 85%.

🏗️ Technical Architecture
The pipeline consists of three distinct layers of defense:

1. The Guardian Layer (XGBoost)
Role: High-speed statistical filtering.

Implementation: An XGBoost classifier trained on 280k+ transactions.

Optimization: Utilized scale_pos_weight to handle extreme class imbalance (0.17% fraud rate) and RobustScaler to manage high-variance transaction amounts.

2. The Memory Layer (FAISS RAG)
Role: Contextual evidence retrieval.

Implementation: A balanced vector library of historical fraud and legitimate cases stored in a Facebook AI Similarity Search (FAISS) index.


Mechanism: Performs L2 Euclidean distance search on the 28 PCA-transformed features to find the 3 "nearest neighbors" from past data.

3. The Reasoning Agent
Role: Explainable AI (XAI) and final verdict.

🧠 Key Engineering Learnings
Grey Zone Management: Establishing a dynamic threshold (0.3–0.7) prevents over-reliance on a single model and reduces false positives.

Vector Similarity in Finance: Euclidean distance in a PCA-reduced space is a highly effective way to identify recurring "Modus Operandi" in fraud rings.

Trust through Transparency: Providing a "Reasoning" narrative satisfies modern FinTech regulatory requirements (GDPR/EU AI Act) for automated decision-making.

📂 Dataset
The project utilizes the Kaggle Credit Card Fraud Detection dataset, containing transactions made by European cardholders in September 2013.

Logic: Interprets retrieved case patterns and semantic signals (e.g., "Micro-Refund" testing behavior) to generate a human-readable justification for every BLOCK or APPROVE decision.
