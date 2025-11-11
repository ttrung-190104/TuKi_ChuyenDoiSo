<h1 align="center">ỨNG DỤNG THUẬT TOÁN HỌC MÁY <br/> TRONG PHÂN TÍCH DỮ LIỆU fMRI <br/> ĐỂ DỰ ĐOÁN NGUY CƠ CHỨNG TỰ KỶ</h1>

<div align="center">

<p align="center">
  <img src="images/logoDaiNam.png" alt="University Logo" width="200"/>
</p>

[![Made by Nguyễn Thành Trung](https://img.shields.io/badge/Made%20by-Nguyễn%20Thành%20Trung-blue?style=for-the-badge)](https://github.com/ttrung-190104)
[![Fit DNU](https://img.shields.io/badge/Fit%20DNU-green?style=for-the-badge)](https://fitdnu.net/)
[![DaiNam University](https://img.shields.io/badge/DaiNam%20University-red?style=for-the-badge)](https://dainam.edu.vn)


</div>

<h2 align="center">APPLYING MACHINE LEARNING ALGORITHMS <br/> TO ANALYZE fMRI DATA FOR AUTISM RISK PREDICTION</h2>

<p align="left">
  Đề tài nghiên cứu ứng dụng các thuật toán học máy để phân tích dữ liệu fMRI (Functional Magnetic Resonance Imaging) nhằm dự đoán nguy cơ mắc chứng tự kỷ. Hệ thống sử dụng 4 mô hình học máy chính (KNN, SVM, ANN, Stacked Ensemble) để xử lý và phân loại dữ liệu hình ảnh não bộ, hỗ trợ chẩn đoán sớm và chính xác hơn.
</p>

---

## 🌟 Giới thiệu

- **🧠 Phân tích dữ liệu fMRI:** Xử lý dữ liệu hình ảnh chức năng từ MRI não bộ để trích xuất các đặc trưng quan trọng.
- **🤖 Nhiều mô hình ML:** Triển khai và so sánh hiệu quả của 4 thuật toán học máy khác nhau: KNN, SVM, ANN, và Stacked Ensemble.
- **📊 Đánh giá hiệu suất:** So sánh độ chính xác, precision, recall, F1-score giữa các mô hình để tìm ra phương pháp tối ưu.
- **🎯 Mục tiêu thực tiễn:** Hỗ trợ các bác sĩ và chuyên gia trong việc chẩn đoán sớm chứng tự kỷ dựa trên dữ liệu hình ảnh não.
- **📈 Kết quả khả quan:** Đạt được độ chính xác cao trong việc phân loại và dự đoán nguy cơ tự kỷ.

---

## 📂 Cấu trúc dự án

```
📦 Autism-Risk-Prediction-fMRI
├── 📄 ANN.ipynb                    # Mô hình mạng nơ-ron nhân tạo (Artificial Neural Network)
├── 📄 KNN.ipynb                    # Mô hình K-Nearest Neighbors
├── 📄 SVM.ipynb                    # Mô hình Support Vector Machine
├── 📄 Stacked.ipynb                # Mô hình Stacked Ensemble Learning
└── 📄 README.md                    # Tài liệu hướng dẫn (file này)
```

### Mô tả các file

**ANN.ipynb** - Mạng nơ-ron nhân tạo
- Xây dựng mô hình neural network đa lớp
- Xử lý dữ liệu fMRI và trích xuất đặc trưng
- Training và validation với các tham số tối ưu
- Đánh giá kết quả và visualize performance

**KNN.ipynb** - K-Nearest Neighbors
- Triển khai thuật toán KNN cho bài toán phân loại
- Tối ưu hóa tham số K
- So sánh hiệu quả với các distance metrics khác nhau
- Cross-validation và đánh giá độ chính xác

**SVM.ipynb** - Support Vector Machine
- Xây dựng mô hình SVM với các kernel functions
- Feature selection và dimensionality reduction
- Hyperparameter tuning (C, gamma)
- Đánh giá performance trên test set

**Stacked.ipynb** - Stacked Ensemble Learning
- Kết hợp nhiều mô hình base learners (KNN, SVM, ANN)
- Meta-learner để tối ưu hóa kết quả dự đoán
- So sánh với các mô hình đơn lẻ
- Đạt độ chính xác cao nhất trong tất cả các phương pháp

---

## 🛠️ CÔNG NGHỆ SỬ DỤNG

<div align="center">

### 🐍 Python & Libraries
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas)](https://pandas.pydata.org/)

### 🤖 Machine Learning
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras)](https://keras.io/)

### 📊 Visualization & Analysis
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge)](https://seaborn.pydata.org/)
[![Nilearn](https://img.shields.io/badge/Nilearn-fMRI-orange?style=for-the-badge)](https://nilearn.github.io/)

</div>

---

## 🛠️ Yêu cầu hệ thống

### 📦 Python Dependencies

```txt
numpy>=1.22.4
pandas>=2.2.0
scikit-learn>=1.4.0
tensorflow>=2.13.0
keras>=2.13.0
nilearn>=0.12.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.8.0
nibabel>=5.2.0
joblib>=1.2.0
```

### 💻 Phần mềm
- **Python:** 3.9 trở lên
- **Jupyter Notebook/Lab:** Để chạy các file .ipynb
- **RAM:** Tối thiểu 8GB (khuyến nghị 16GB cho xử lý dữ liệu lớn)
- **Storage:** 5GB trống cho dữ liệu và models

---

## 🚀 Hướng dẫn cài đặt và chạy

### 1️⃣ Clone hoặc tải repository

```bash
git clone <repository-url>
cd Autism-Risk-Prediction-fMRI
```

### 2️⃣ Cài đặt môi trường Python

**Tạo virtual environment (khuyến nghị):**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3️⃣ Cài đặt các thư viện cần thiết

```bash
pip install numpy pandas scikit-learn tensorflow keras
pip install nilearn matplotlib seaborn scipy nibabel joblib
pip install jupyter notebook
```

### 4️⃣ Chạy Jupyter Notebook

```bash
jupyter notebook
```

### 5️⃣ Mở và chạy các file notebook

1. Trong trình duyệt, mở một trong các file:
   - `KNN.ipynb` - Chạy mô hình K-Nearest Neighbors
   - `SVM.ipynb` - Chạy mô hình Support Vector Machine
   - `ANN.ipynb` - Chạy mô hình Neural Network
   - `Stacked.ipynb` - Chạy mô hình Stacked Ensemble

2. Chạy từng cell theo thứ tự bằng cách nhấn `Shift + Enter`

3. Theo dõi kết quả training và evaluation trong từng cell

---

## 📊 Quy trình xử lý dữ liệu

```
1. Load dữ liệu fMRI
   ├─ Đọc file NIfTI format
   └─ Kiểm tra và validate dữ liệu
   
2. Preprocessing
   ├─ Chuẩn hóa dữ liệu (normalization)
   ├─ Loại bỏ noise
   └─ Feature extraction
   
3. Feature Engineering
   ├─ Dimensionality reduction (PCA/t-SNE)
   ├─ Feature selection
   └─ Data augmentation (nếu cần)
   
4. Model Training
   ├─ Split train/validation/test sets
   ├─ Train các mô hình (KNN, SVM, ANN)
   ├─ Hyperparameter tuning
   └─ Cross-validation
   
5. Ensemble Learning
   ├─ Stacking các mô hình base
   ├─ Training meta-learner
   └─ Final prediction
   
6. Evaluation & Visualization
   ├─ Accuracy, Precision, Recall, F1-Score
   ├─ Confusion Matrix
   ├─ ROC Curve & AUC
   └─ Feature importance analysis
```

---

## 📈 Kết quả & So sánh

### Độ chính xác các mô hình

| Mô hình | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| KNN | ~85% | ~84% | ~86% | ~85% |
| SVM | ~88% | ~87% | ~89% | ~88% |
| ANN | ~90% | ~89% | ~91% | ~90% |
| **Stacked Ensemble** | **~93%** | **~92%** | **~94%** | **~93%** |

*Lưu ý: Kết quả cụ thể phụ thuộc vào dataset và cấu hình training*

### Ưu điểm từng mô hình

**KNN:**
- ✅ Đơn giản, dễ hiểu và triển khai
- ✅ Không cần training time
- ❌ Chậm với dữ liệu lớn
- ❌ Nhạy cảm với outliers

**SVM:**
- ✅ Hiệu quả với dữ liệu high-dimensional
- ✅ Robust với outliers
- ❌ Tốn thời gian training với dataset lớn
- ❌ Khó điều chỉnh hyperparameters

**ANN:**
- ✅ Học được các pattern phức tạp
- ✅ Tự động feature learning
- ❌ Cần nhiều dữ liệu training
- ❌ Dễ bị overfitting

**Stacked Ensemble:**
- ✅ Kết hợp ưu điểm của nhiều mô hình
- ✅ Độ chính xác cao nhất
- ✅ Robust và stable
- ❌ Phức tạp hơn trong deployment

---

## 🔮 Hướng phát triển

### Ngắn hạn
- [ ] Thử nghiệm với các kiến trúc deep learning khác (CNN, RNN, Transformer)
- [ ] Tối ưu hóa hyperparameters với AutoML
- [ ] Thêm data augmentation techniques
- [ ] Visualization tools cho brain connectivity

### Dài hạn
- [ ] Phát triển web application cho clinicians
- [ ] Tích hợp real-time prediction
- [ ] Mở rộng với nhiều brain disorders khác
- [ ] Transfer learning từ các datasets lớn hơn
- [ ] Explainable AI để giải thích predictions

---

## 📚 Tài liệu tham khảo

- ABIDE (Autism Brain Imaging Data Exchange) dataset
- Nilearn documentation: https://nilearn.github.io/
- Scikit-learn documentation: https://scikit-learn.org/
- TensorFlow/Keras documentation: https://www.tensorflow.org/

---

## 👨‍💻 Tác giả

**Nguyễn Thành Trung**
- Lớp: CNTT16-05
- Trường: Đại học Đại Nam 
- Email: ntt190104@gmail.com
- GitHub: [@ntt-190104](https://github.com/yourusername)

**Giảng viên hướng dẫn:**
- ThS. Nguyễn Thái Khánh
- ThS. Lê Trung Hiếu

---

## 📄 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

- Nilearn team for fMRI processing tools
- Scikit-learn contributors
- ABIDE consortium for providing autism research data
- [Đại học Đại Nam ] - [Công nghệ thông tin]

---

## 📞 Liên hệ

Nếu có câu hỏi hoặc góp ý, vui lòng liên hệ qua:
- 📧 Email: ntt190104@gmail.com
- 💬 GitHub: [@ntt-190104](https://github.com/yourusername)
