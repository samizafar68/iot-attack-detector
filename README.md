# 🔐 Malicious Network Activity Detection in IoT Environments

## 📌 Overview

This project implements an end-to-end machine learning pipeline to detect malicious network activity in IoT environments using the **CIC-IoMT 2024 WiFi-MQTT** dataset. The pipeline encompasses data preprocessing, exploratory data analysis (EDA), dimensionality reduction, model training, evaluation, and prediction on unseen data, culminating in a submission for a Kaggle competition.

Two models were developed—**Random Forest** and **XGBoost**—with Random Forest outperforming XGBoost in evaluation metrics.

---

## 📂 Dataset

- **Training Data:** `train.csv` (7,160,831 rows, 46 columns)  
- **Test Data:** `test_without_label.csv` (1,614,182 rows, 45 columns)  
- **Source:** CIC-IoMT 2024 WiFi-MQTT dataset  
- **Objective:** Classify network traffic as **benign** or **malicious** (e.g., `Benign`, `TCP_IP-DDoS-UDP2`, `MQTT-DDoS-Connect_Flood`, etc.)

---

## 🔁 Machine Learning Pipeline

### 1. 📊 Data Preprocessing

- Handled missing values using:
  - **Median** for numeric columns  
  - **Mode** for categorical columns  
  > *(Note: No missing values were found in the dataset.)*
- Encoded the target `label` column using `LabelEncoder` (**51 unique classes**).
- Standardized numeric features using `StandardScaler` (Z-score normalization).

---

### 2. 🔍 Exploratory Data Analysis (EDA)

- **Class Distribution:**  
  - Visualized using a histogram.
  - Revealed significant class imbalance (e.g., `TCP_IP-DDoS-UDP2` with 207,295 samples vs. `Recon-Ping_Sweep` with 148 samples).

- **Correlation Matrix:**  
  - Generated heatmap to identify feature correlations.  
  - Example: strong correlation between `Rate` and `Srate`.

- **Summary Statistics:**  
  - Confirmed successful standardization with mean ~0 and std ~1.

---

### 3. ⚙️ Dimensionality Reduction

- Applied **Principal Component Analysis (PCA)** to reduce **45 features to 30** components.
- Cumulative explained variance plot showed that 30 components captured **nearly 100% variance**.
- Significantly reduced computational complexity.

---

### 4. 🧠 Model Training

**Models Used:**

- ✅ Random Forest Classifier (`n_estimators=100`, `max_depth=20`)  
- ✅ XGBoost Classifier (`n_estimators=100`, `max_depth=20`)

- Split data into:
  - Training Set: 5,728,664 samples  
  - Validation Set: 1,432,167 samples  
  - **Stratified sampling** was applied.

- Training time for Random Forest: **~28 minutes** (1700.97 seconds)

---

### 5. 📈 Evaluation

**Metrics Used:**

- **Precision**, **Recall**, **F1-score** (macro average)
- **Confusion Matrix** heatmap for visual clarity

**Observations:**

- Random Forest outperformed XGBoost.
- Performed well for well-represented classes like `Benign`, `MQTT-DDoS-Connect_Flood`.
- Struggled with underrepresented classes like `TCP_IP-DDoS-UDP4`.

---

### 6. 🧪 Prediction

- Generated predictions on test data using both models.
- Created:
  - `submission_file.csv` (Random Forest)
  - `submission_xgb_file.csv` (XGBoost)
- Submission format: `index, label`  
  > Example: `0, TCP_IP-DoS-SYN3`  
- Total rows in each submission file: **1,048,575**

---

## 🧩 Challenges Faced

- **Class Imbalance:**  
  - Tackled with stratified sampling and RF robustness  
  - Still impacted performance on rare classes

- **High Dimensionality:**  
  - Solved using PCA (from 45 to 30 components)

- **TPU Initialization Failure for XGBoost:**  
  - Attempted TPU failed; fell back to CPU/GPU  
  - Reduced XGBoost performance

- **Large Dataset Size:**  
  - Managed using PCA and parallel processing (`n_jobs=-1`)  
  - RF training time: 28 minutes

---

## 🧰 Technologies Used

**Languages:**  
- Python 3.8+

**Libraries:**
- Data Manipulation: `Pandas`, `NumPy`
- Machine Learning: `Scikit-learn`, `XGBoost`, `TensorFlow`
- Visualization: `Matplotlib`, `Seaborn`
- Model Persistence: `Joblib`

**Tools:**
- Jupyter Notebook
- Kaggle Notebook (GPU enabled)

---

## 🛠️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/samizafar68/iot-attack-detector.git
