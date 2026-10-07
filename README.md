# 📚Scikit-learn-cookbook
Over 80 recipes for machine learning in Python with scikit-learn
# scikit-learn Cookbook — Bab 1–5

|           |                         |
| --------- | ----------------------- |
| **Nama**  | Zacky Yusup Hakim       |
| **NIM**   | 101032300183            |
| **Kelas** | BS1TK-47-REG-G13        |

Repositori ini berisi rangkuman dan implementasi kode Python untuk **Bab 1 sampai Bab 5** dari buku *scikit-learn Cookbook, Third Edition* (John Sukup, Packt Publishing, 2025). Setiap bab disajikan dalam satu Jupyter Notebook yang menggabungkan penjelasan konsep (Bahasa Indonesia) dengan contoh kode, visualisasi, ringkasan *Key Ideas*, dan latihan beserta solusinya (pada bab yang memiliki latihan).

Repositori ini dibuat untuk **Tugas 2 (Enrichment for Machine Learning Classes): Code Reproduction + Theoretical Deep-Dive**. Seluruh output (grafik dan hasil perhitungan) sudah tersimpan di notebook, sehingga isinya dapat dibaca tanpa dijalankan ulang.

---

## Daftar Isi

1. [Struktur Proyek](#struktur-proyek)
2. [Instalasi & Cara Menjalankan](#instalasi--cara-menjalankan)
3. [Bab 1 — Common Conventions and API Elements of scikit-learn](#bab-1--common-conventions-and-api-elements-of-scikit-learn)
4. [Bab 2 — Pre-Model Workflow and Data Preprocessing](#bab-2--pre-model-workflow-and-data-preprocessing)
5. [Bab 3 — Dimensionality Reduction Techniques](#bab-3--dimensionality-reduction-techniques)
6. [Bab 4 — Building Models with Distance Metrics and Nearest Neighbors](#bab-4--building-models-with-distance-metrics-and-nearest-neighbors)
7. [Bab 5 — Linear Models and Regularization](#bab-5--linear-models-and-regularization)
8. [Dataset](#dataset)
9. [Library yang Digunakan](#library-yang-digunakan)
10. [Referensi](#referensi)

---

## Struktur Proyek

```
scikit-learn Cookbook/
├── README.md
└── notebooks/
    ├── scikit-learn-Cookbook-Chapter1.ipynb
    ├── scikit-learn-Cookbook-Chapter2.ipynb
    ├── scikit-learn-Cookbook-Chapter3.ipynb
    ├── scikit-learn-Cookbook-Chapter4.ipynb
    └── scikit-learn-Cookbook-Chapter5.ipynb
```

## Instalasi & Cara Menjalankan

```
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
cd notebooks
jupyter notebook
```

Notebook juga dapat dibuka langsung di **Google Colab** (*File → Open notebook → GitHub*). Semua dataset berasal dari modul `sklearn.datasets` atau dibuat secara sintetis di dalam notebook. Satu-satunya yang memerlukan internet adalah `fetch_california_housing()` pada latihan akhir Bab 2, karena datasetnya diunduh saat pertama kali dijalankan.

---

## Bab 1 — Common Conventions and API Elements of scikit-learn

📓 [`scikit-learn-Cookbook-Chapter1.ipynb`](notebooks/scikit_learn-Cookbook_Chapter1.ipynb)

Bab fondasi: **konvensi dan pola API** yang dipakai hampir semua model scikit-learn. Bab ini tidak memiliki repositori kode terpisah di buku, sehingga seluruh contohnya diambil dari teks buku.

**Materi:**

1. Filosofi desain scikit-learn (konsistensi, kesederhanaan, modularitas, *reusability*)
2. *Estimator*: `fit()`, `predict()`, `fit_predict()` (`LinearRegression`, `KMeans`)
3. *Transformer*: `fit()`, `transform()`, `fit_transform()` (`StandardScaler`)
4. Estimator dan transformer kustom (`BaseEstimator`, *mixin class*)
5. Pipeline dan otomatisasi *workflow* (pengantar MLOps)
6. Atribut dan metode umum (`coef_`, `intercept_`, `score()`)
7. Penyetelan hyperparameter (`GridSearchCV`, `RandomizedSearchCV`, `set_params()`, `get_params()`)
8. Metadata: *tags* dan *metadata routing*
9. Praktik terbaik penggunaan API

**Rangkuman:**

- Hampir semua model mengikuti pola **`fit()` → `predict()`** (atau `transform()`), sehingga mengganti model cukup mengubah satu baris kode.
- **Data uji hanya di-`transform()`** dengan parameter dari data latih; memanggil `fit` ulang pada data uji menyebabkan inkonsistensi (*data leakage*).
- **Pipeline** membuat *workflow* ringkas, konsisten, dan dapat direproduksi.

---

## Bab 2 — Pre-Model Workflow and Data Preprocessing

📓 [`scikit-learn-Cookbook-Chapter2.ipynb`](notebooks/scikit_learn_Cookbook_Chapter2.ipynb)

*Garbage in, garbage out*: menyiapkan data mentah sebelum pemodelan.

**Materi:**

1. Dampak data mentah terhadap performa model
2. Data hilang: `SimpleImputer`, `KNNImputer`, `IterativeImputer`
3. Scaling: `StandardScaler`, `MinMaxScaler`, `Normalizer`
4. Encoding kategorikal: `OneHotEncoder`, `LabelEncoder`, `ColumnTransformer`
5. Pengantar pipeline & visualisasi pipeline (`set_config(display="diagram")`)
6. *Feature engineering*: `PolynomialFeatures`, `KBinsDiscretizer`, `RFE`, `SelectFromModel`
7. Latihan: pipeline komprehensif pada California Housing

**Rangkuman:**

- Imputasi bergerak dari yang sederhana (rata-rata) ke berbasis tetangga (KNN) dan berbasis model antarfitur (iteratif).
- `StandardScaler` (z-score), `MinMaxScaler` (rentang 0–1), dan `Normalizer` (norma satuan per **baris**) punya kegunaan berbeda; scaling penting untuk model berbasis jarak/gradien.
- `OneHotEncoder` untuk kategori nominal; `LabelEncoder` hanya untuk label target.
- Preprocessing sebaiknya dibangun sebagai **pipeline** agar konsisten dan bebas *data leakage*.

---

## Bab 3 — Dimensionality Reduction Techniques

📓 [`scikit-learn-Cookbook-Chapter3.ipynb`](notebooks/scikit_learn-Cookbook_Chapter3.ipynb)

Mengurangi jumlah fitur sambil mempertahankan informasi penting.

**Materi:**

1. Pengantar: *curse of dimensionality*, *feature selection* vs *feature extraction*
2. PCA: variansi maksimum, komponen ortogonal, *explained variance ratio*
3. LDA: separasi kelas (*supervised*)
4. Perbedaan PCA dan LDA
5. t-SNE untuk visualisasi dan pengaruh `perplexity`
6. Panduan memilih teknik reduksi dimensi
7. Latihan: PCA + regresi logistik (Digits) dan t-SNE + K-means

**Rangkuman:**

- **PCA** (*unsupervised*) pada dataset Wine: dua komponen pertama menjelaskan ≈ 55% variansi (PC1 ≈ 36,2%, PC2 ≈ 19,2%); butuh ≈ 10 komponen untuk 95%.
- **LDA** (*supervised*) memisahkan tiga kelas anggur lebih jelas dari PCA; komponen maksimum = jumlah kelas − 1.
- **t-SNE** bersifat non-linear dan hanya untuk visualisasi (tidak punya `transform()` untuk data baru).
- Pada Digits, PCA mereduksi 64 fitur menjadi 39 dengan akurasi regresi logistik turun tipis dari ≈ 97,5% ke ≈ 96,4%.

---

## Bab 4 — Building Models with Distance Metrics and Nearest Neighbors

📓 [`scikit-learn-Cookbook-Chapter4.ipynb`](notebooks/scikit_learn_Cookbook_Chapter4.ipynb)

Model **KNN**: prediksi dari *k* tetangga terdekat, sehingga hasilnya bergantung pada definisi "jarak".

**Materi:**

1. Memahami KNN: cara kerja, pengaruh nilai *k*, kelemahan
2. Metrik jarak: Euclidean, Manhattan, Minkowski
3. Penyetelan hyperparameter dengan `GridSearchCV`
4. Evaluasi: *cross-validation*, *learning curve*, *confusion matrix*, precision/recall/F1
5. Tambahan: efek scaling, memilih *k*, KNN untuk regresi
6. Latihan: classifier KNN (Breast Cancer), *grid search*, dan evaluasi pada data sintetis

**Rangkuman:**

- *k* kecil → *overfitting*; *k* besar → *underfitting*. Nilai terbaik dipilih lewat *cross-validation*.
- Pada Iris, kombinasi terbaik hasil grid search: `n_neighbors=9`, `metric='euclidean'`, `weights='uniform'` (skor CV ≈ 0,99).
- **Scaling sangat penting**: pada Wine, akurasi KNN naik dari ≈ 63,9% (tanpa scaling) menjadi ≈ 94,4% (dengan scaling).
- Pada Breast Cancer, KNN yang di-*scale* mencapai akurasi ≈ 94,7%.

---

## Bab 5 — Linear Models and Regularization

📓 [`scikit-learn-Cookbook-Chapter5.ipynb`](notebooks/scikit_learn_Cookbook_Chapter5.ipynb)

Regresi linear, masalahnya pada data berkorelasi, dan solusi **regularisasi**.

**Materi:**

1. Regresi linear (OLS, MSE, R²) dan asumsinya
2. Ridge (L2) dan Lasso (L1)
3. ElasticNet dan *coefficient path plot*
4. Regresi polinomial (`PolynomialFeatures`)
5. Spline (`SplineTransformer`)
6. Tambahan: memilih `alpha` dengan `RidgeCV` dan `LassoCV`
7. Latihan: Ridge, Lasso, dan ElasticNet

**Rangkuman:**

- **Ridge** mengecilkan koefisien; **Lasso** dapat menolkan koefisien (seleksi fitur otomatis); **ElasticNet** menggabungkan keduanya lewat `l1_ratio`.
- Pada data multikolinear, ElasticNet memberi R² terbaik (≈ 0,15) dibanding Ridge (≈ 0,08), Lasso (≈ 0,06), dan regresi linear (≈ 0,06). Nilai R² rendah karena sinyal pada dataset sintetis ini lemah.
- Dengan `alpha` dipilih lewat *cross-validation*, `LassoCV` hanya mempertahankan 14 dari 100 fitur dan menaikkan R² dari ≈ 0,065 menjadi ≈ 0,205.
- Polinomial derajat 5 mencapai R² ≈ 0,983 pada data bergelombang; pada latihan Ridge, penambahan fitur polinomial menaikkan R² uji dari ≈ −0,04 menjadi ≈ 0,89.

---

## Dataset

| Bab | Dataset                                                                | Sumber                                       |
| --- | ---------------------------------------------------------------------- | -------------------------------------------- |
| 1   | Data mainan, Iris (pada contoh tambahan)                               | Didefinisikan di notebook / `sklearn.datasets` |
| 2   | Data sintetis acak; California Housing (latihan)                       | NumPy / `fetch_california_housing()`         |
| 3   | Wine, Digits                                                           | `sklearn.datasets`                           |
| 4   | Iris, Wine, Breast Cancer; `make_circles`, `make_classification`       | `sklearn.datasets`                           |
| 5   | `make_regression` (multikolinear), data polinomial/sinus sintetis      | `sklearn.datasets` / NumPy                   |

## Library yang Digunakan

`numpy` · `pandas` · `scikit-learn` · `matplotlib` · `seaborn`

## Referensi

- Sukup, J. (2025). *scikit-learn Cookbook, Third Edition*. Packt Publishing.
- Repositori kode resmi: <https://github.com/PacktPublishing/scikit-learn-Cookbook-Third-Edition>
- Dokumentasi scikit-learn: <https://scikit-learn.org/stable/>
