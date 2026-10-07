# scikit-learn-cookbook
Over 80 recipes for machine learning in Python with scikit-learn
scikit-learn Cookbook
Reproduksi kode dan ringkasan teori dari buku scikit-learn Cookbook, Third Edition (John Sukup, Packt Publishing), dikerjakan sebagai Tugas 2 (Enrichment for Machine Learning Classes): Code Reproduction + Theoretical Deep-Dive.
> **Penulis:** _(isi nama dan NIM)_
> **Cakupan saat ini:** Chapter 1 sampai 5 (batas pengumpulan: 10 Oktober 2026, 23:59)
---
Tujuan Repositori
Repositori ini bertujuan memperdalam pemahaman dan keterampilan praktis dalam menerapkan konsep inti Machine Learning melalui:
Reproduksi kode dari setiap chapter buku,
Ringkasan setiap subbab (Key Ideas) pada tiap notebook, dan
Penjelasan teori dalam Bahasa Indonesia untuk setiap konsep yang dipraktikkan.
---
Struktur Repositori
```
scikit-learn-Cookbook/
├── README.md
├── scikit-learn-Cookbook-Chapter1.ipynb
├── scikit-learn-Cookbook-Chapter2.ipynb
├── scikit-learn-Cookbook-Chapter3.ipynb
├── scikit-learn-Cookbook-Chapter4.ipynb
└── scikit-learn-Cookbook-Chapter5.ipynb
```
Chapter	Judul	Notebook
1	Common Conventions and API Elements of scikit-learn	Chapter 1
2	Pre-Model Workflow and Data Preprocessing	Chapter 2
3	Dimensionality Reduction Techniques	Chapter 3
4	Building Models with Distance Metrics and Nearest Neighbors	Chapter 4
5	Linear Models and Regularization	Chapter 5
---
Ringkasan Isi Setiap Chapter
Chapter 1: Konvensi Umum dan Elemen API scikit-learn
Chapter ini adalah fondasi konsep untuk seluruh buku. Fokusnya pada filosofi desain scikit-learn (konsistensi, kesederhanaan, modularitas, reusability) dan pola API yang sama pada hampir semua model.
Estimator (`fit()`, `predict()`, `fit_predict()`) dan transformer (`fit()`, `transform()`, `fit_transform()`).
Aturan penting: data uji hanya di-`transform()` dengan parameter dari data latih, untuk mencegah data leakage.
Membuat estimator/transformer kustom dengan `BaseEstimator` dan mixin class.
Pipeline untuk mengotomatisasi workflow dan pengantar singkat MLOps.
Atribut dan metode umum (`coef_`, `intercept_`, `score()`), pencarian hyperparameter (`GridSearchCV`, `RandomizedSearchCV`, halving search), serta metadata (tags dan metadata routing).
Chapter 2: Alur Pra-Model dan Preprocessing Data
Chapter ini menunjukkan bahwa kualitas model dimulai dari kualitas data. Data mentah hampir selalu bermasalah, sehingga perlu dibersihkan dan disiapkan sebelum pemodelan.
Data hilang: `SimpleImputer`, `KNNImputer`, `IterativeImputer`.
Scaling: `StandardScaler`, `MinMaxScaler`, `Normalizer`.
Encoding kategorikal: `OneHotEncoder`, `LabelEncoder`, `ColumnTransformer`.
Pipeline dan visualisasinya (`set_config(display="diagram")`).
Feature engineering: `PolynomialFeatures`, `KBinsDiscretizer`, serta seleksi fitur dengan `RFE` dan `SelectFromModel`.
Latihan: pipeline komprehensif (imputasi, scaling, `RandomForestRegressor`) pada dataset California Housing.
Chapter 3: Teknik Reduksi Dimensi
Chapter ini membahas cara mengurangi jumlah fitur sambil mempertahankan informasi penting, untuk mengatasi curse of dimensionality, overfitting, dan kebutuhan visualisasi.
PCA: unsupervised, memaksimalkan variansi, komponen ortogonal, dan explained variance ratio.
LDA: supervised, memaksimalkan separasi kelas, dengan maksimum komponen = jumlah kelas − 1.
t-SNE: non-linear, mempertahankan struktur lokal, untuk visualisasi, serta pengaruh `perplexity`.
Tabel perbandingan dan panduan memilih PCA, LDA, dan t-SNE.
Latihan: PCA + regresi logistik pada dataset Digits (64 → 39 dimensi dengan akurasi tetap tinggi) dan t-SNE + K-means.
Chapter 4: Model dengan Metrik Jarak dan Nearest Neighbors
Chapter ini membahas KNN, model yang memprediksi berdasarkan k tetangga terdekat, sehingga hasilnya sangat bergantung pada definisi "jarak".
Cara kerja KNN, pengaruh nilai k (bias–variance trade-off), dan kelemahannya.
Metrik jarak: Euclidean, Manhattan, Minkowski (dan Cosine sebagai teori).
Penyetelan hyperparameter (`n_neighbors`, `weights`, `metric`) dengan `GridSearchCV`.
Evaluasi: skor cross-validation, learning curve, confusion matrix, serta precision, recall, dan F1.
Tambahan: efek scaling, memilih k, dan KNN untuk regresi.
Latihan: classifier KNN (Breast Cancer), grid search, dan evaluasi pada data sintetis 3 kelas.
Chapter 5: Model Linear dan Regularisasi
Chapter ini membahas regresi linear, masalahnya pada data berdimensi tinggi atau berkorelasi (multikolinearitas), dan solusinya berupa regularisasi.
Regresi linear (OLS, MSE, R²) beserta asumsinya.
Ridge (penalti L2), Lasso (penalti L1 yang memilih fitur), dan ElasticNet (gabungan L1 + L2) dengan coefficient path plot.
Regresi polinomial dan spline (`SplineTransformer`) untuk hubungan non-linear.
Tambahan: memilih `alpha` dengan `RidgeCV` dan `LassoCV`.
Latihan: Ridge, Lasso, dan ElasticNet pada data sintetis.
---
Cara Menjalankan
Clone repositori ini, atau buka notebook langsung di Google Colab (File → Open notebook → GitHub).
Instal dependensi (Colab sudah menyediakan sebagian besar):
```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```
Jalankan sel dari atas ke bawah. Seluruh output sudah tersimpan di notebook, sehingga isinya dapat dibaca tanpa dijalankan ulang.
Catatan
Chapter 2, sel latihan terakhir memakai `fetch_california_housing()` yang mengunduh dataset dari internet. Jalankan sel itu dengan koneksi aktif agar outputnya muncul.
Beberapa dataset sintetis pada kode buku dibuat tanpa `random_state`, sehingga angka tertentu dapat sedikit berbeda saat dijalankan ulang.
Versi library yang dipakai: scikit-learn 1.x (terbaru), Python 3.
---
Konvensi Penulisan Notebook
Setiap notebook diawali pengantar dan daftar isi, dan diakhiri ringkasan chapter berisi tabel konsep dan fungsi.
Bagian Key Ideas merangkum poin utama tiap subbab.
Sel kode yang berlabel "Tambahan (tidak ada di buku)" adalah contoh pelengkap yang bukan bagian dari kode asli buku.
Kode buku bersumber dari teks buku dan repositori resmi: https://github.com/PacktPublishing/scikit-learn-Cookbook-Third-Edition.
---
Referensi
Sukup, J. (2025). scikit-learn Cookbook, Third Edition. Packt Publishing.
Dokumentasi resmi scikit-learn: https://scikit-learn.org/stable/
Integritas Akademik
Kode direproduksi dari buku sebagai bagian dari tugas pembelajaran. Penjelasan teori disusun dengan bantuan LLM sebagaimana diizinkan pada ketentuan tugas, lalu ditinjau dan disesuaikan oleh penulis repositori.
