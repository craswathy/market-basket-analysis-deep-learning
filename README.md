# 🧠 Association Rule Mining and Neural Networks for Smart Retail Recommendations

> **MSc Statistics Research Project — Hybrid recommendation system combining ECLAT, FP-Growth association rule mining with CNN, Bi-LSTM, and CNN-BiLSTM deep learning models on 541,909 retail transactions**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/craswathy/market-basket-analysis-deep-learning/blob/main/Association_Rule_Mining_Algorithms_Eclat%2C_FP_Growth.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/craswathy/market-basket-analysis-deep-learning/blob/main/Deep_Learning_models_CNN%2C_Bi_LSTM%2CCNN_BiLSTM.ipynb)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

---

## 📌 Project Overview

This project builds an intelligent product recommendation system for the retail industry by combining two approaches:

1. **Association Rule Mining** — discovering which products are frequently bought together using ECLAT and FP-Growth algorithms
2. **Deep Learning** — predicting the next product a customer is likely to purchase using CNN, Bi-LSTM, and a CNN-BiLSTM hybrid model

The project was submitted as an MSc Statistics research project at Christ College (Autonomous), Irinjalakuda, University of Calicut (2026).

---

## 🎯 Research Objectives

- Compare ECLAT and FP-Growth algorithms for association rule generation
- Build and evaluate CNN, Bi-LSTM, and CNN-BiLSTM deep learning models for next-product prediction
- Identify product bundling opportunities to improve cross-selling
- Generate personalised recommendations to improve customer retention

---

## 📊 Dataset

| Metric | Value |
|--------|-------|
| **Dataset** | UCI Online Retail Dataset |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+retail) |
| **Original Records** | 541,909 transactions |
| **Time Period** | December 2010 – December 2011 |
| **Country** | United Kingdom (UK-based online gift shop) |
| **Customers** | 4,372 unique customers |
| **Products** | 3,956 unique products |
| **Countries** | 38 countries |

---

## 🔧 Data Preprocessing

Raw data was cleaned systematically before analysis:

| Step | Records Removed | Reason |
|------|----------------|--------|
| Remove duplicates | 5,268 | Exact duplicate rows |
| Remove cancelled transactions | 2,438 | InvoiceNo starting with 'C' |
| Remove returns | 8,923 | Negative quantity values |
| Remove invalid prices | 1,247 | Zero or negative UnitPrice |
| **Records after cleaning** | **524,878** | Ready for analysis |

### Product Name Standardisation
Product names were standardised using a custom `clean_product_name()` function:
- Converted to uppercase
- Removed special characters
- Applied abbreviation replacements (e.g. WHITE → WHT, HANGING → HNG)
- Truncated names longer than 50 characters

### Transaction Basket Creation
- Products appearing in fewer than 30 transactions were filtered out
- **2,583 unique products** retained after filtering
- **18,216 transaction baskets** created for association rule mining
- Single-item baskets removed

---

## 🛒 Part 1 — Association Rule Mining

### Notebook
📓 `Association_Rule_Mining_Algorithms_Eclat,_FP_Growth.ipynb`

### ECLAT Algorithm (Custom Implementation)

ECLAT (Equivalence Class Clustering and bottom-up Lattice Traversal) uses a **vertical data format** — each item stores the set of transaction IDs (tid-list) in which it appears. Frequent itemsets are found by intersecting these tid-lists recursively.

**Parameters used:**
```python
eclat = ECLAT(
    min_support=0.01,      # 1% minimum support (182 transactions)
    min_confidence=0.30,   # 30% minimum confidence
    min_lift=1.0,          # Lift > 1 = positive association
    max_itemset_size=3     # Up to 3-item frequent sets
)
```

**ECLAT Results:**

| Metric | Value |
|--------|-------|
| Total transactions processed | 18,216 |
| Frequent 1-itemsets | 889 |
| Frequent 2-itemsets | 1,156 |
| Frequent 3-itemsets | 4,214 |
| **Total frequent itemsets** | **6,259** |
| Association rules generated | 27,441 |
| Average confidence | 96.99% |
| Average lift | 17.69 |

### FP-Growth Algorithm

FP-Growth uses an **FP-Tree structure** to compress the transaction database, avoiding repeated database scans.

**Parameters used:**
```python
# Same parameter settings as ECLAT for fair comparison
min_support=0.01
min_confidence=0.30
min_lift=1.0
```

**FP-Growth Results:**

| Metric | Value |
|--------|-------|
| Total transactions processed | 18,216 |
| Total frequent itemsets | 2,473 |
| Association rules generated | 2,777 |
| Average confidence | 53.31% |
| Average lift | 14.82 |

### Algorithm Comparison

| Metric | ECLAT | FP-Growth |
|--------|-------|-----------|
| Frequent itemsets found | **6,259** | 2,473 |
| Association rules | **27,441** | 2,777 |
| Average confidence | **96.99%** | 53.31% |
| Average lift | **17.69** | 14.82 |
| Data structure | Vertical (tid-lists) | Horizontal (FP-Tree) |
| Data scan | Single scan | Two scans |
| Best for | Dense data | Sparse data |
| Memory efficiency | Lower | Higher |

**Key insight:** ECLAT discovered 2.5x more frequent itemsets and 10x more association rules than FP-Growth, with significantly higher average confidence. FP-Growth is more memory-efficient but produces fewer, more conservative rules.

### Top Association Rules from ECLAT

| Antecedent (IF) | Consequent (THEN) | Support | Confidence | Lift |
|-----------------|-------------------|---------|------------|------|
| POPPYS PLAYHOUSE LIVINGROOM | POPPYS PLAYHOUSE BATHROOM + KITCHEN | 1.80% | 144.49% | 80.25 |
| HERB MARKER THYME | HERB MARKER ROSEMARY | 1.32% | 101.69% | 77.19 |
| HERB MARKER BASIL + THYME | HERB MARKER ROSEMARY | 1.32% | 100.00% | 77.19 |

---

## 🧠 Part 2 — Deep Learning Models

### Notebook
📓 `Deep_Learning_models_CNN,_Bi_LSTM,CNN_BiLSTM.ipynb`

### Data Preparation for Deep Learning

- Customers with fewer than 10 purchase sessions were filtered out
- Products converted to integer IDs using `product_to_id` mapping
- Sequential purchase histories created per customer (sorted by InvoiceDate)
- **Sequence length:** 10 products (input) → predict 11th product (target)
- **Train/Test split:** 80% / 20% (random_state=42)

### Model 1 — CNN (Convolutional Neural Network)

```python
cnn_model = Sequential([
    Embedding(vocab_size, 64, input_length=seq_len),
    Conv1D(64, 3, activation='relu', padding='same'),
    MaxPooling1D(2),
    Conv1D(128, 3, activation='relu', padding='same'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
```

**Purpose:** Captures local patterns in purchase sequences using convolutional filters — identifies which product combinations appear close together in a customer's history.

### Model 2 — Bi-LSTM (Bidirectional Long Short-Term Memory)

```python
bilstm_model = Sequential([
    Embedding(vocab_size, 64, input_length=seq_len, mask_zero=True),
    Bidirectional(LSTM(64, return_sequences=False, dropout=0.2)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
```

**Purpose:** Captures long-range sequential dependencies in both forward and backward directions — understands the full context of a customer's purchase history.

### Model 3 — CNN-BiLSTM Hybrid (Best Model)

```python
inputs = Input(shape=(seq_len,))
embedding = Embedding(vocab_size, 64, mask_zero=True)(inputs)

# CNN Branch — captures local patterns
cnn_branch = Conv1D(64, 3, activation='relu', padding='same')(embedding)
cnn_branch = GlobalMaxPooling1D()(cnn_branch)

# BiLSTM Branch — captures sequential context
bilstm_branch = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2))(embedding)

# Combine both branches
combined = Concatenate()([cnn_branch, bilstm_branch])
dense = Dense(64, activation='relu')(combined)
dense = Dropout(0.3)(dense)
outputs = Dense(num_classes, activation='softmax')(dense)

hybrid_model = Model(inputs=inputs, outputs=outputs)
```

**Purpose:** Combines CNN's pattern detection with BiLSTM's sequential memory — gets the best of both architectures simultaneously.

**Optimizer:** Adam (learning_rate=0.001)  
**Loss function:** sparse_categorical_crossentropy  
**Batch size:** 64

### Model Results

| Model | Train Accuracy | Test Accuracy |
|-------|---------------|---------------|
| CNN | 59% | 85% |
| Bi-LSTM | 68% | 90% |
| **CNN-BiLSTM** | **71%** | **99%** |

**CNN-BiLSTM outperformed standalone models by 9–14% on test accuracy.**

### Next-Product Prediction

The CNN-BiLSTM model was used for real-time next-product prediction:

```python
def predict_next_product(model, product_names_sequence,
                          product_to_id, vocab_size, seq_len):
    ids = [product_to_id.get(p, 0) for p in product_names_sequence]
    # pad/trim to seq_len, predict, return product name
    ...
```

**Sample prediction:**
```
Input sequence:
['WHT METAL LANTERN',
 'CREAM CUPID HEARTS COAT HANGER',
 'KNITTED UNION FLAG HOT WATER BOTTLE']

Predicted next product: WHT HNG HEART TLIGHT HLDR
Inference time: 37ms
```

---

## 📈 Key Results and Findings

- **CNN-BiLSTM hybrid achieved 99% test accuracy** — the best performing model
- **Real-time inference in 37ms** — suitable for production deployment
- **ECLAT generated 27,441 rules** with 96.99% average confidence
- **Strong product associations found** in Poppys Playhouse toy sets (lift up to 80.25) and herb garden markers (lift up to 77.19)
- **FP-Growth produced higher-quality focused rules** while ECLAT provided comprehensive pattern coverage

---

## 🛠 Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.x | Core language |
| TensorFlow | 2.10 | Deep learning framework |
| Keras | Built-in | Model building API |
| Pandas | Latest | Data manipulation |
| NumPy | Latest | Numerical computing |
| Scikit-learn | Latest | Train/test split, metrics |
| Matplotlib | Latest | Visualisation |
| Seaborn | Latest | Statistical plots |
| NetworkX | Latest | FP-Tree network visualisation |
| mlxtend | Latest | TransactionEncoder |
| itertools | Built-in | Combination generation (ECLAT) |

---

## 📁 Repository Structure

```
market-basket-analysis-deep-learning/
├── notebooks/
│   ├── Association_Rule_Mining_Algorithms_Eclat,_FP_Growth.ipynb
│   └── Deep_Learning_models_CNN,_Bi_LSTM,CNN_BiLSTM.ipynb
├── report/
│   └── CR_ASWATH-PROJECT.pdf
├── images/
│   ├── eclat_results.png
│   ├── fpgrowth_results.png
│   ├── algorithm_comparison.png
│   ├── model_accuracy_comparison.png
│   └── fptree_visualization.png
└── README.md
```

---

## 🚀 How to Run

### Option 1 — Run in Google Colab (Recommended)

Click the **Open in Colab** badges at the top of this README. No installation needed.

### Option 2 — Run Locally

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/craswathy/market-basket-analysis-deep-learning.git
cd market-basket-analysis-deep-learning
```

**Step 2 — Install dependencies:**
```bash
pip install tensorflow keras pandas numpy scikit-learn matplotlib seaborn networkx mlxtend
```

**Step 3 — Download the dataset:**
- Download the UCI Online Retail dataset from [here](https://archive.ics.uci.edu/ml/datasets/online+retail)
- Save as `online_retail_us.csv` in your working directory

**Step 4 — Run the notebooks in order:**
```
1. Association_Rule_Mining_Algorithms_Eclat,_FP_Growth.ipynb
2. Deep_Learning_models_CNN,_Bi_LSTM,CNN_BiLSTM.ipynb
```

---

## 📄 Project Report

The full MSc project report is available in the `report/` folder:
📄 `CR_ASWATH-PROJECT.pdf`

---

## 👩‍💻 Author

**C R Aswathy** — Register No. CCAYMST005  
MSc Statistics, Christ College (Autonomous), Irinjalakuda  
University of Calicut | 2024–2026

📧 aswathycr44@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/c-r-aswathy-67591235a)  
🐱 [GitHub](https://github.com/craswathy)  
📍 Kerala, India | Open to Data Analyst and Data Scientist roles in India and UAE

---

## 📜 Citation

If you use this work, please cite:

```
C R Aswathy (2026). Association Rule Mining and Neural Networks 
for Smart Retail Recommendations. MSc Statistics Project, 
Christ College (Autonomous), University of Calicut.
```

Dataset citation:
```
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository 
[Online Retail Dataset]. Irvine, CA: University of California, 
School of Information and Computer Science.
```
READMEEOF
echo "README created successfully"
wc -l /mnt/user-data/outputs/README_market_basket.md
