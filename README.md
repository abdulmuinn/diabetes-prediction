# 🩺 Diabetes Prediction using Neural Network (PyTorch)

## 📌 Project Overview
Project ini merupakan implementasi **Deep Learning sederhana menggunakan PyTorch** untuk memprediksi kemungkinan seseorang menderita **diabetes** berdasarkan data medis. Project ini dibuat sebagai **project pembelajaran sekaligus portofolio**, dengan fokus pada pemahaman workflow Machine Learning end-to-end, bukan sekadar akurasi.

---

## 📊 Dataset
- **Nama Dataset**: Pima Indians Diabetes Dataset  
- **Sumber**: Kaggle  
- **Jumlah Data**: 768 sampel  
- **Jumlah Fitur**: 8 fitur medis (Pregnancies, Glucose, BloodPressure, BMI, Age, dll.)  
- **Target**:
  - `0` → Tidak Diabetes  
  - `1` → Diabetes  

Distribusi kelas:
- Tidak diabetes: 500  
- Diabetes: 268  

Dataset bersifat **tidak seimbang**, sehingga evaluasi model tidak hanya mengandalkan akurasi.

---

## 🧠 Model Architecture
Model yang digunakan adalah **Feedforward Neural Network** dengan arsitektur berikut:

Input Layer (8 fitur)
→ Hidden Layer 1 (16 neuron, ReLU)
→ Hidden Layer 2 (8 neuron, ReLU)
→ Output Layer (1 neuron, Sigmoid)


**Detail model:**
- Activation Function:
  - Hidden Layer: ReLU
  - Output Layer: Sigmoid
- Loss Function: Binary Cross Entropy Loss (BCELoss)
- Optimizer: Adam

---

## ⚙️ Data Preprocessing
Tahapan preprocessing yang dilakukan:
1. Memisahkan fitur dan target
2. Train-test split (80% training, 20% testing)
3. Feature scaling menggunakan `StandardScaler`
4. Konversi data ke Tensor PyTorch

---

## 🚀 Training
- Epoch: 2000
- Learning Rate: 0.001
- Training dilakukan menggunakan full batch
- Loss dimonitor selama proses training

---

## 📈 Evaluation
Evaluasi model dilakukan menggunakan:
- Accuracy
- Confusion Matrix
- Precision, Recall, dan F1-score

### 🔹 Baseline Accuracy
Baseline dihitung dengan menebak kelas mayoritas:
Baseline Accuracy: 0.6429

### 🔹 Model Accuracy
Model Accuracy: 0.7338


Model berhasil **mengungguli baseline**, menandakan bahwa model benar-benar belajar dari data.

---

## 🧮 Confusion Matrix (Threshold = 0.5)
[[74 25]
[16 39]]


Interpretasi:
- True Negative (TN): 74
- False Positive (FP): 25
- False Negative (FN): 16
- True Positive (TP): 39

---

## 🎯 Threshold Tuning
Dalam konteks medis, **recall untuk kelas diabetes lebih diprioritaskan** dibandingkan akurasi. Oleh karena itu, dilakukan eksperimen threshold sebagai berikut:

| Threshold | Accuracy | Recall (Diabetes) | False Negative |
|---------|---------|------------------|---------------|
| 0.3 | 0.64 | 0.56 | 24 |
| **0.4** | **0.65** | **0.56** | **24** |
| 0.5 | 0.66 | 0.55 | 25 |

### ✅ Threshold Final: **0.4**

Threshold 0.4 dipilih karena memberikan trade-off terbaik antara akurasi dan kemampuan mendeteksi pasien diabetes, dengan jumlah false negative yang lebih rendah dibanding threshold default.

---

## 🧾 Kesimpulan
Model neural network berhasil memprediksi diabetes dengan performa yang lebih baik dibandingkan baseline. Melalui threshold tuning, model menunjukkan peningkatan kemampuan dalam mendeteksi pasien diabetes dengan mengorbankan sedikit akurasi. Dalam konteks medis, trade-off ini dapat diterima karena mengutamakan deteksi dini pasien berisiko.

---

## 🔮 Future Improvements
Beberapa pengembangan lanjutan yang dapat dilakukan:
- Penanganan class imbalance (class weighting atau oversampling)
- Hyperparameter tuning lanjutan
- Model explainability (SHAP / feature importance)
- Deployment sederhana (web application)

---

## 🛠️ Tech Stack
- Python  
- PyTorch  
- Pandas  
- NumPy  
- Scikit-learn  
- Google Colab  

---

## 👤 Author
**Nama**: Abdul Muin  
**Project Type**: Learning & Portfolio Project  
**Topic**: Deep Learning – Binary Classification  

---

⭐ Jika project ini bermanfaat, silakan beri **star** pada repository ini.

