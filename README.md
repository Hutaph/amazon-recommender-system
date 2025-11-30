# Amazon Recommender System
> Hệ gợi ý Amazon Beauty chạy thuần NumPy, kiểm soát trọn vẹn pipeline dữ liệu và hai model chính: baseline phổ biến và Matrix Factorization cho bài toán dự đoán rating.

## Mục lục
1. [Giới thiệu](#giới-thiệu)
2. [Dataset](#dataset)
3. [Method](#method)
4. [Installation & Setup](#installation--setup)
5. [Usage](#usage)
6. [Results](#results)
7. [Project Structure](#project-structure)
8. [Challenges & Solutions](#challenges--solutions)
9. [Future Improvements](#future-improvements)
10. [Contributors](#contributors)
11. [Thông tin tác giả](#thông-tin-tác-giả)
12. [Contact](#contact)
13. [License](#license)

## Giới thiệu
### Bài toán
Xây dựng hệ gợi ý cho danh mục Amazon Beauty dựa trên dữ liệu rating (1–5 sao). Hệ thống phải dự đoán chính xác mức độ hài lòng và chuẩn bị danh sách gợi ý top-K phù hợp ngay cả khi dữ liệu cực kỳ thưa (sparsity > 99.9%).

### Động lực & ứng dụng
- Cải thiện trải nghiệm mua sắm với đề xuất cá nhân hóa.
- Hạn chế bias do sản phẩm nổi tiếng và xử lý tốt cold-start khi mở rộng danh mục.
- Làm nền cho các mô hình nâng cao (neural CF, session-based) bằng cách cung cấp pipeline sạch.

### Mục tiêu cụ thể
- Làm sạch dữ liệu và áp dụng k-core k=5 để đảm bảo tối thiểu 5 tương tác cho mỗi người dùng/sản phẩm.
- Chuẩn hoá trọng số thời gian (`TimeWeight`) hoàn toàn bằng NumPy để dùng cho 2 model: baseline và MF.
- Cung cấp baseline Popularity Recommender và Matrix Factorization (MF) với đánh giá RMSE.
- Trình bày bộ câu hỏi phân tích chuyên sâu trong `01_data_exploration.ipynb` nhằm dẫn dắt các ý tưởng khi modeling.

## Dataset
### Nguồn dữ liệu
- `data/raw/ratings_Beauty.csv`: tập rating Amazon Beauty (UserId, ProductId, Rating, Timestamp).
- Dữ liệu được tải từ bộ [Amazon Ratings trên Kaggle](https://www.kaggle.com/datasets/skillsmuggler/amazon-ratings), lọc theo danh mục Beauty và không kèm metadata khác.

### Mô tả các features
- `UserId`: định danh người dùng (chuỗi).
- `ProductId`: định danh sản phẩm Amazon (ASIN).
- `Rating`: mục tiêu (1–5, integer).
- `Timestamp`: Unix epoch theo giây.
- `TimestampDT`: chuyển đổi sang `datetime64[s]` phục vụ phân tích thời gian.
- `TimeWeight`: trọng số duy nhất (min–max timestamp về [0.2, 1.0]) được dùng cho cả baseline và MF.
- `ProductMean`: trung bình rating của từng sản phẩm (sau k-core), phục vụ Popularity baseline mà không cần tính lại trong notebook.
- Các bias (`UserMean`, `ItemMean`) hay residual được tính lại trực tiếp trong notebook từ dữ liệu train chứ không lưu trong CSV.

#### Giới thiều về các features
| Feature | Kiểu dữ liệu | Vai trò mô hình | Diễn giải |
| --- | --- | --- | --- |
| `UserId`, `ProductId` | string → int index | Khóa chính | Được encode liên tục để phục vụ ma trận thưa |
| `Rating` | int8 | Mục tiêu regression | Giá trị gốc 1–5, được chuẩn hoá (z-score) trong notebook trước khi train để khớp repo |
| `Timestamp` | int64 | Sắp xếp thời gian | Dùng cho temporal split + tính `TimeWeight` |
| `TimestampDT` | datetime64[s] | EDA thời gian | Thân thiện khi vẽ biểu đồ theo tháng/quý |
| `TimeWeight` | float32 | Loss weighting | Min-max thời gian (0.2–1.0) theo timestamp toàn cục |
| `ProductMean` | float32 | Baseline feature | Trung bình rating item, lưu sẵn để Popularity baseline dùng trực tiếp |
| (Bias/residual) | tính động | Chuẩn hóa baseline | Được tính lại trong notebook từ tập train, không lưu trong CSV |

### Kích thước & đặc điểm dữ liệu
- Sau k-core k=5: **198 502** rating, **22 363** người dùng, **12 101** sản phẩm, mật độ 0.073%.
- Khoảng thời gian: từ 2002-06 đến 2014-07 (`Timestamp` 1 023 840 000 – 1 406 073 600).
- 77.7% rating ≥4 ⇒ dữ liệu nghiêng mạnh về phản hồi tích cực.
- Temporal split 80/20:
  - Train: 158 801 rating, 20 833 người dùng, 11 776 sản phẩm.
  - Test: 39 701 rating, 10 219 người dùng, 8 834 sản phẩm.
  - Mức hoạt động trung bình: 7.62 rating/user ở train vs 3.89 ở test; 13.5 rating/item ở train vs 4.49 ở test.
  - Test coverage: 69.7% người dùng và 82.4% sản phẩm xuất hiện trước đó trong train.

#### Workflow của tiền xử  lý dữ liệu
| Giai đoạn | #Rating | #User | #Item | Ghi chú |
| --- | ---: | ---: | ---: | --- |
| CSV gốc `ratings_Beauty.csv` | 2 023 070 | 1 210 271 | 249 274 | Tập Amazon Beauty chưa làm sạch (mean rating 4.15) |
| Sau làm sạch + dedup (giữ rating mới nhất) | 2 023 070 | 1 210 271 | 249 274 | Dữ liệu hợp lệ vì log đã sạch sẵn nhưng vẫn được kiểm chứng bằng NumPy |
| Sau k-core k=5 (trước feature engineering) | 198 502| 22 363 | 12 101 | Loại 90% tương tác để đảm bảo mỗi entity ≥5 rating |
| Sau feature engineering & split | 198 502 | 22 363 | 12 101*| Bộ `ratings_processed.csv` dùng cho modeling |

## Method
### Quy trình xử lý dữ liệu
1. **Load & Clean (`load_raw_data`, `clean_raw`)**  
   - Đọc file CSV bằng `np.genfromtxt`, ánh xạ thủ công header.  
   - Vector hóa kiểm tra rỗng, ép kiểu rating sang `int`, bỏ timestamp âm ⇒ đảm bảo đầu vào thuần NumPy.
2. **Deduplicate (`deduplicate_latest`)**  
   - Ghép khóa `UserId_ProductId`, sắp xếp theo timestamp giảm dần và giữ bản ghi đầu tiên.  
   - Tránh hiện tượng user sửa rating nhiều lần khiến mô hình học trùng lặp.
3. **K-core filtering (`kcore_filter`, k=5)**  
   - Dùng `np.unique` và `np.isin` để đếm lượt tương tác mỗi vòng, lọc user/item <5 tương tác tới khi hội tụ hoặc đủ 10 vòng.  
   - Log mỗi vòng giúp tái lập quyết định trade-off giữa coverage và độ nhiễu.
4. **Datetime & Encoding (`to_datetime_minimal`, `encode_ids`)**  
   - Chuyển timestamp sang `datetime64[s]`, bổ sung Year/Month/Day phục vụ EDA.  
   - Mã hóa `UserId`/`ProductId` sang index liên tục bằng `np.unique(return_inverse=True)` để sẵn sàng xây CSR.
5. **Feature engineering (`save_processed`)**  
   - Chuẩn hoá `TimeWeight` bằng min–max timestamp (0.2 → 1.0) giống repo tham chiếu.  
   - Tính trung bình rating theo từng `ProductId` và ghi vào cột `ProductMean` để Popularity baseline chỉ cần đọc dữ liệu.  
   - Không còn lưu Recency/Bias/Residual; các thống kê này được notebook tính lại từ tập train để tránh lệ thuộc vào artifact cũ.  
   - Xuất `ratings_processed.csv` gọn nhẹ (7 cột) phục vụ mọi notebook/modeling.
6. **Temporal split (`split_train_test`)**  
   - Sắp xếp toàn bộ mapped interaction theo timestamp và cắt 80/20; trả về cả mask để tái sử dụng trong notebook khác.  
   - Giữ nguyên sự dịch chuyển thời gian nên tránh leakage giữa train/test.
7. **Standardize ratings (trong notebook modeling)**  
   - Sau khi split, rating train được chuẩn hoá (z-score), test dùng cùng mean/std ⇒ RMSE ở thang chuẩn hoá.  
   - Product mean baseline cũng được xây lại trên rating chuẩn hoá train trước khi feed vào `BaselineModel`.
8. **Visualization (`src/visualization.py`)**  
   - Hỗ trợ các biểu đồ chuyên sâu (Lorenz, k-core trade-off, timestamp integrity, bias violin, train/test distribution,...).


### Thuật toán sử dụng
#### Popularity Recommender (Baseline Model)
- Dự đoán: $\hat r_{ui} = \bar r_i$, với $\bar r_i$ là trung bình rating item $i$ trong tập train (fallback về global mean khi item mới).
- Thực hiện bằng vector hóa `np.add.at` và `np.unique` để tính tổng rating và đếm cho từng sản phẩm.

#### Matrix Factorization
- Mô hình: $\hat r_{ui} = \mu + b_u + b_i + p_u^\top q_i$.
- Hàm mất mát có trọng số:
  $$
  \mathcal{L} = \sum_{(u,i)} w_{ui} \left(r_{ui} - \hat r_{ui}\right)^2 + \lambda\left(\|p_u\|^2 + \|q_i\|^2 + b_u^2 + b_i^2\right)
  $$
- Thực thi SGD với NumPy: truy cập vector `P[u]`, `Q[i]`, cập nhật tại chỗ cho từng tương tác; $w_{ui}$ chính là `TimeWeight`.
- Mô hình làm việc trên thang rating gốc; product-mean baseline được tái tính mỗi lần chạy.

### Giải thích triển khai NumPy
- Pipeline không dùng pandas/os/pathlib để giữ tính thuần NumPy, ngay cả I/O CSV cũng dựa `np.genfromtxt`.
- `kcore_filter` dùng `np.unique` + `np.isin` để lọc theo batch, đảm bảo hiệu năng với ~200k dòng.
- MF sử dụng broadcasting và phép nhân dot (`np.dot`) cho cả forward lẫn backward; không phụ thuộc thư viện chuyên dụng (scipy, torch).
- Bộ nhớ được quản lý bằng cách tái sử dụng structured array và tránh copy không cần thiết (`np.take` thay vì slicing sâu).
- Tất cả seed random (`np.random.seed`) được thiết lập trong notebook modeling để kết quả lặp lại 1:1.

### Metric đánh giá models
- **RMSE**:  
  $$
  \text{RMSE} = \sqrt{\frac{1}{N} \, \sum_{(u,i)} (r_{ui} - \hat r_{ui})^2}
  $$
  Được dùng cho cả baseline Popularity và MF nhằm đo lường sai số dự đoán rating tuyệt đối.

## Tải và cài đặt
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
- Dữ liệu thô đặt tại `data/raw/ratings_Beauty.csv`.
- Các notebook sử dụng JupyterLab/Notebook; đảm bảo `ipykernel` đã được cài trong môi trường.
- Nếu tải dataset từ nguồn khác, đảm bảo giữ đúng header `UserId,ProductId,Rating,Timestamp` và mã hóa UTF-8 để `np.genfromtxt` đọc chính xác.

## Hướng dẫn sử dụng
> Sau khi kích hoạt môi trường và cài `requirements.txt`, hãy mở `jupyter lab` (hoặc `jupyter notebook`) tại thư mục gốc rồi thực hiện lần lượt ba notebook dưới đây để tái hiện toàn bộ pipeline.

1. **`01_data_exploration.ipynb` – Khám phá dữ liệu chuyên sâu**  
   Chạy `Run → Run All Cells` để  tìm hiểu sâu về dataset, trả lời 5 câu hỏi EDA và rà soát chất lượng dữ liệu sau k-core. Notebook cũng giải thích rõ hai feature mới `TimeWeight`, `ProductMean`, lưu ý lý do tồn tại và cách chúng được dùng ở bước modeling nhằm tránh hiểu nhầm khi tái chạy pipeline.  
   Ngoài phần biểu đồ, cuối notebook còn liệt kê kết luận chi tiết (bao gồm mức độ sparse, mức lệch rating, sự ổn định của dòng thời gian) để làm cơ sở cho mọi quyết định ở bước 02 và 03.

2. **`02_preprocessing.ipynb` – Tiền xử lý dữ liệu**  
   Sử dụng cell cấu hình đầu notebook để điều chỉnh tham số (ví dụ: `kcore_k`, phạm vi timestamp, lựa chọn chuẩn hoá thời gian). Sau đó chạy từng cell theo thứ tự để tải dữ liệu thô, làm sạch, k-core, encode và ghi `data/processed/ratings_processed.csv` cùng các mapping đi kèm.  
   Notebook còn in log từng giai đoạn (số vòng k-core, tỉ lệ giữ lại, kích thước mask train/test) nên rất hữu ích khi bạn muốn so sánh giữa các lần chạy hoặc chứng minh pipeline thuần NumPy hoạt động đúng.

3. **`03_modeling.ipynb` – Huấn luyện mô hình**  
   Đọc dữ liệu đã xử lý từ bước 02, tạo train/test theo thời gian, chuẩn hoá rating (z-score) rồi huấn luyện `BaselineModel` và `MatrixFactorizationModel` với cấu hình (100 epoch + early stopping).  
   Các cell cuối trình bày biểu đồ đường học, scatter giữa giá trị thật và dự đoán, đồng thời có phần “Nhận xét cuối” tóm tắt ưu/nhược điểm của từng mô hình bằng tiếng Việt để tiện đưa vào báo cáo.


## Results
| Model | RMSE (ratings) | Ghi chú |
| --- | --- | --- |
| Baseline Model | **1.0049** | Mean rating theo sản phẩm, tính trên 39 700 rating test |
| Matrix Factorization Model | **0.9547**| RMSE & đường học lưu trong notebook do phụ thuộc cấu hình epoch |

### Evaluation protocol
- Sắp xếp toàn bộ interaction theo timestamp và cắt 80/20 ⇒ mọi metric đều phản ánh kịch bản “huấn luyện quá khứ, dự đoán tương lai”.
- Tập test chỉ được sử dụng sau khi hoàn tất tuning hyperparameter trên validation (tách ra từ train hoặc dùng early stopping).
- Mọi đánh giá đều bỏ qua user/item hoàn toàn mới trong test (không có trong train); những trường hợp này nhận giá trị fallback (global mean hoặc 0).

#### Train/Test snapshot
| Tập | #Rating | #User | #Item | Avg rating/user | Avg rating/item | Coverage so với train |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Train (80% đầu tiên) | 158 801 | 20 833 | 11 776 | 7.62 | 13.49 | — |
| Test (20% cuối cùng) | 39 701 | 10 219 | 8 834 | 3.89 | 4.49 | User 69.7% · Item 82.4% |

#### Insight trực quan hoá
- **Biểu đồ minh họa (nằm trong `01_data_exploration.ipynb`):**
  - `plot_interaction_lorenz`: Lorenz cho user vs product trước/sau k-core, cho thấy 20% user tạo ra ~78% rating.
  - `plot_kcore_tradeoff_curve`: đường trade-off giữa k và số rating/coverage giúp justify k=5.
  - `plot_timestamp_integrity`: kiểm tra sạch timestamp, phát hiện không có gap bất thường.
  - `plot_bias_violin`: residual sau khi trừ bias bó trong ±1, xác nhận pipeline bias-removal hoạt động.
  - `plot_train_test_distribution`: chứng minh test vẫn đa dạng nhưng ít dense hơn (user avg 3.9 rating), phù hợp đánh giá gần thực tế.

### So sánh & phân tích
- Baseline RMSE **1.0049** phản ánh sai số gần với trung bình rating theo sản phẩm; dữ liệu vẫn lệch mạnh về 4–5 sao nên mô hình này chưa nắm rõ sự khác biệt giữa nhóm người dùng.
- Matrix Factorization (100 epoch + early stopping, dừng ở epoch 32) đạt RMSE **0.9547**, giảm ~5% so với baseline nhờ chuẩn hóa z-score và học nhân tố tiềm ẩn cho cả user/item.
- Trọng số thời gian (0.2–1.0) giúp SGD chú ý tương tác mới, giữ đường học train/test nằm sát nhau mà không cần regularization phức tạp.
- Temporal split 80/20 giữ nguyên dòng thời gian nên ~30% user trong test hoàn toàn mới; cần kết hợp popularity hoặc metadata để xử lý cold-start.
- Coverage user/item ở test đạt 69.7%/82.4%, đủ cho đánh giá ổn định nhưng vẫn còn dư địa cải thiện bằng mô hình hybrid hoặc bổ sung feature nội dung.

## Cấu trúc Project
```
amazon-recommender-system/
├── data/
│   ├── raw/
│   │   └── ratings_Beauty.csv
│   └── processed/
│       ├── ratings_processed.csv
│       ├── user_mapping.csv
│       └── product_mapping.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data_processing.py
│   ├── models.py
│   └── visualization.py
├── requirements.txt
└── README.md
```


**Giải thích **
- `data/raw/`: chứa file CSV gốc.
- `data/processed/`: 
- `notebooks/`: workflow theo trình tự 01 → 02 → 03; mỗi notebook đều sử dụng chung processed data để đảm bảo tính nhất quán.
- `src/data_processing.py`: hiện thân pipeline thuần NumPy, bao gồm load, clean, k-core, encode, split, save.
- `src/models.py`: Popularity Recommender (mean theo item) và Matrix Factorization SGD kèm early stopping.
- `src/visualization.py`: bộ sưu tập hàm vẽ phục vụ Section 9 của notebook EDA.
- `requirements.txt`: danh sách thư viện tối thiểu (numpy, pandas, matplotlib, seaborn, notebook…).

## Vấn đề và giải pháp
- **Sparsity 0.07%:**  
  - *Vấn đề:* ma trận user-item gần như rỗng → MF khó hội tụ.  
  - *Giải pháp:* áp dụng k-core k=5 (loại bỏ 90% rating nhiễu) và phân tích Lorenz để chứng minh trade-off chấp nhận được cho business.
- **Triển khai thuần NumPy:**  
  - *Vấn đề:* môi trường giới hạn thư viện nên không thể dùng pandas/scikit.  
  - *Giải pháp:* mọi phép tính đều dùng `np.genfromtxt`, `np.unique`, `np.bincount`, `np.add.at`; kể cả serialization cũng viết tay nhằm đảm bảo khả năng chạy ở môi trường tối giản.
- **Độ tin cậy metric:**  
  - *Vấn đề:* RMSE thấp có thể là ảo nếu validation không đúng thời gian.  
  - *Giải pháp:* luôn cố định seed, log timestamp cut-off và lưu mask; README này cũng mô tả quy trình đánh giá để hạn chế hiểu sai kết quả.
- **Tối ưu IO 2M dòng:**  
  - *Vấn đề:* `np.genfromtxt` dễ tốn bộ nhớ khi đọc file lớn.  
  - *Giải pháp:* đọc từng cột rồi ghép structured array, đồng thời chuyển bớt cột sang `int32/float32` để giảm dung lượng ~35%.

## Cải thiện trong tương lai
- Xây pipeline implicit-feedback (BPR/ALS) để phục vụ top-K khi thiếu rating rõ ràng.
- Khai thác metadata (category, price, brand) nếu có để giảm popularity bias.
- Tích hợp kiểm tra fairness/coverage theo phân khúc sản phẩm để đảm bảo model không chỉ ưu tiên sản phẩm hot của một vài hãng.

## Contributors
### Thông tin tác giả
- **Họ và tên:** Huỳnh Tấn Phước
- **MSSV:** 23120334
- **Môn học:** Lập trình cho Khoa học Dữ liệu
- **Lớp:** CQ2023/21

### Contact
- **Email cá nhân:** huynhtanphuoc164@gmail.com
- **Email sinh viên:** 23120334@student.hcmus.edu.vn

## License
- Dự án hiện phục vụ mục đích học tập. Vui lòng xin phép tác giả trước khi sử dụng cho các mục đích khác.
