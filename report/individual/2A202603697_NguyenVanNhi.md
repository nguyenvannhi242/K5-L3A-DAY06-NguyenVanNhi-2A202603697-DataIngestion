# Báo cáo cá nhân — Day 6: Data Ingestion Engineering

- **Họ và tên:** Nguyễn Văn Nhi
- **MSSV:** `2A202603697`
- **Lớp:** `L3A`
- **Ngày thực hiện:** 2026-11-13

---

## 1. Dữ liệu và Pipeline

### Chủ đề, nguồn/quyền, số trang và quy tắc tính trang
- **Corpus:** **NIST AI 100-1 — *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*** (phiên bản 1.0, tháng 1/2023).
- **Nguồn tải:** `https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf` (lưu tại `starter/data/NIST.AI.100-1.pdf`).
- **Quyền sử dụng:** *Public Domain* (ấn phẩm của Chính phủ Hoa Kỳ — NIST, được phép tải, xử lý và phân phối lại).
- **Số trang và quy tắc tính:**
  - Tổng số trang vật lý: **48 trang**.
  - Trang bị loại (6 trang): `P01–P02` (bìa/tiêu đề), `P03` (tuyên bố miễn trừ/cảm ơn), `P04` (mục lục), `P05` (danh sách hình), `P48` (bìa sau/DOI).
  - **Số trang nội dung hợp lệ:** **42 trang** (`P06–P47`, tổng **101.495 ký tự** sau chuẩn hóa), vượt yêu cầu tối thiểu 30 trang. Chi tiết từng trang được ghi tại [data/manifest.json](../../data/manifest.json).

### Chuẩn hóa, lỗi và giới hạn
- Chuẩn hóa Unicode `NFKC`, chuẩn hóa dấu xuống dòng `\r\n` -> `\n`, ghép từ bị ngắt gạch nối cuối dòng (`(\w)-\s*\n\s*(\w)` và `(\w)-\s+(\w)`), giữ nguyên cấu trúc đoạn văn (`\n\n`) cho chiến lược chunking theo đoạn.
- Toàn bộ 42 trang nội dung đều trích xuất text thành công bằng `pypdf 7.0.0`, không có trang scan/lỗi OCR; 6 trang bìa/mục lục/bìa sau được ghi log loại trừ trong `results/summary.json`.
- **Giới hạn xử lý:** Việc ghép gạch nối tự động có thể nối cả một số từ ghép thật bị ngắt dòng (ví dụ `context-specific` thành `contextspecific`); bảng và hình vẽ (`Fig. 1–3`) chỉ giữ text phẳng, mất bố cục cột.

### Embedding Model, Vector Store, Metric, Lệnh chạy và Chi phí
- **Embedding model:** `BAAI/bge-small-en-v1.5` (chạy local qua `sentence-transformers 6.3.0`, **384 chiều**, `normalize_embeddings=True`). Với query, áp dụng prefix chuẩn của BGE: `"Represent this sentence for searching relevant passages: "`; document giữ nguyên text.
- **Vector store:** `qdrant-client 1.20.0` (chế độ local persistent lưu tại `starter/results/qdrant_db/`, metric **Cosine**, exact search). *(Ghi chú: Docker Desktop trên máy không khởi động được engine Linux nên sử dụng Qdrant local persistent theo đúng tùy chọn thay thế hợp lệ trong `GUIDE.md` và `starter/README.md`.)*
- **Idempotency (nạp lại không trùng):** Dùng UUIDv5 tất định từ `chunk_id` và xóa chunk cũ theo `document_id` trước khi upsert. Kiểm chứng nạp lại Chiến lược A: số bản ghi giữ nguyên **239 → 239** (`no_duplicates: true`).
- **Lệnh chạy:**
  ```bash
  cd starter
  python -m pip install -r requirements.txt
  python -m pip install sentence-transformers qdrant-client
  python -m src.run
  ```
- **Chi phí:** **$0.00** (chạy 100% local).

### Metadata bắt buộc và trường lọc bổ sung
- **Metadata bắt buộc:** `document_id` (`"nist-ai-100-1"`), `source` (`"starter/data/NIST.AI.100-1.pdf"`), `page` (`int`, 6–47).
- **Trường bổ sung:**
  - `section` (`str`): ánh xạ trực tiếp từ mục lục tài liệu gồm 4 giá trị phân biệt: `executive_summary` (P06–P08), `part1_foundational` (P09–P24), `part2_core_profiles` (P25–P39), `appendix` (P40–P47).
  - `year` (`int`: `2023`), `document_type` (`str`: `"framework"`), `language` (`str`: `"en"`).
- **Lý do chọn `section`:** Corpus là một tài liệu dài có cấu trúc phân tầng rõ; người dùng thường hỏi giới hạn trong phần khái niệm (*Part 1*), bảng hướng dẫn hành động (*Part 2 Core*), hoặc phần phụ lục (*Appendix*).

---

## 2. Cấu hình và Benchmark

### Ba chiến lược A / B / C (kiểm soát biến)
| Chiến lược | Chunking | Tham số | Metadata Filter | Điểm khác biệt có kiểm soát |
|---|---|---|---|---|
| **A** | `fixed_size` | `size=500`, `overlap=50` | Không | Baseline cắt cứng theo ký tự |
| **B** | `by_paragraph` | `max_size=800`, `overlap=50` (chỉ khi câu/đoạn vượt 800) | Không | **A → B:** chỉ thay thuật toán chunking |
| **C** | `by_paragraph` | Giống hệt B (dùng chung chunk & vector) | Lọc `section` khi câu hỏi cung cấp ngữ cảnh phần | **B → C:** chỉ bật metadata filter |

### Bộ câu hỏi Benchmark (20 câu: 15 dev / 5 test)
- Lưu tại [benchmark/questions.json](../../benchmark/questions.json), chốt **trước khi chạy thí nghiệm**.
- Phân chia: **15 câu `dev` (`q01–q15`)** và **5 câu `test` (`q16–q20`)**, phủ 4 nhóm: trực tiếp (`truc_tiep`), diễn đạt lại (`dien_dat_lai`), điều kiện/ngoại lệ (`dieu_kien_ngoai_le`), và kiểm tra filter (`metadata_filter`).
- **Câu hỏi metadata filter (4 câu):** `q09` (`part1_foundational`), `q12` (`part2_core_profiles`), `q15` (`appendix`) trong `dev`; và `q16` (`executive_summary`) trong `test`. Giá trị filter được lấy từ ngữ cảnh câu hỏi trước khi truy xuất, không rò rỉ đáp án/trang gold.
- **Kiểm chứng evidence:** Đã chạy script kiểm tra 20/20 `evidence_quote` tồn tại nguyên văn trên trang `gold_page` sau chuẩn hóa.
- **Tiêu chí chọn chiến lược (chốt trước khi xem test):** Ưu tiên `Evidence Hit@3` cao nhất trên `dev`, sau đó đến Grounding/Coherence, và cuối cùng là thời gian/số chunk.

---

## 3. Kết quả

### Bảng 1: Hiệu năng truy xuất và tài nguyên
| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian ingest | Chi phí |
|---|---|---:|---:|---:|---:|---:|
| **A** | `dev` | 8/15 (53.3%) | 9/15 (60.0%) | 239 | 7.27s (embed 6.49s + upsert 0.78s) | \$0.00 |
| **A** | `test` | 4/5 (80.0%) | 4/5 (80.0%) | 239 | 7.27s | \$0.00 |
| **B** | `dev` | **11/15 (73.3%)** | **12/15 (80.0%)** | **170** | **5.44s** (embed 4.92s + upsert 0.52s) | \$0.00 |
| **B** | `test` | 4/5 (80.0%) | **5/5 (100.0%)** | **170** | **5.44s** | \$0.00 |
| **C** | `dev` | **11/15 (73.3%)** | **12/15 (80.0%)** | 170 | 5.49s (embed 4.92s + upsert 0.57s) | \$0.00 |
| **C** | `test` | 4/5 (80.0%) | **5/5 (100.0%)** | 170 | 5.49s | \$0.00 |

### Bảng 2: Đánh giá thủ công Coherence, Metadata và Grounding
| Strategy | Coherence TB /2 (n=10) | Metadata đủ & đúng (n/N) | Filter tuân thủ (n/N hoặc N/A) | Grounding 0/1/2 (số câu) & TB /2 (n=5 test) |
|---|---:|---|---|---|
| **A** | 1.00 / 2.00 (10×1) | 15/15 (100%) | N/A | `0: 0, 1: 1, 2: 4` → **1.80 / 2.00** |
| **B** | **1.80 / 2.00** (8×2, 2×1) | 15/15 (100%) | N/A | `0: 0, 1: 0, 2: 5` → **2.00 / 2.00** |
| **C** | **1.80 / 2.00** (8×2, 2×1) | 15/15 (100%) | **3/3 (dev), 1/1 (test)** | `0: 0, 1: 0, 2: 5` → **2.00 / 2.00** |

**Đường dẫn artifacts & cách chọn mẫu:**
- Trace đầy đủ: [results/traces.json](../../results/traces.json)
- Tổng hợp tự động: [results/summary.json](../../results/summary.json)
- Mẫu Coherence (10 trang đầu `P06–P15`, chốt trước khi xem điểm): [results/coherence_samples.json](../../results/coherence_samples.json)
- Nhãn audit thủ công (tác giả **Nguyễn Văn Nhi** tự chấm; có nguy cơ thiên lệch người chấm): [results/manual_audit.json](../../results/manual_audit.json)

---

## 4. Phân tích và Kết luận

### So sánh A–B (Tác động của Chunking)
- Chuyển từ fixed-size 500 ký tự (A) sang paragraph-aware tối đa 800 ký tự (B) giúp:
  - **Tăng mạnh độ chính xác truy xuất:** `dev` Hit@1 tăng từ **8/15 (53.3%) → 11/15 (73.3%)** (+20 điểm %), `dev` Hit@3 tăng **9/15 (60.0%) → 12/15 (80.0%)**, và `test` Hit@3 đạt **5/5 (100%)** so với **4/5 (80%)** của A.
  - **Cải thiện độ mạch lạc (Coherence):** tăng từ **1.00/2 → 1.80/2**, vì A liên tục cắt ngang giữa từ/câu ở mốc 500 ký tự (`"communities, so"`, `"are diff"`, `"use-"`), trong khi B giữ trọn vẹn đoạn văn.
  - **Giảm tài nguyên:** số chunk giảm **28.9%** (239 → 170), thời gian ingest giảm từ **7.27s → 5.44s**.

### So sánh B–C (Tác động của Metadata Filter)
- Trên 4 câu có filter (`q09`, `q12`, `q15`, `q16`), chiến lược C đạt **tuân thủ filter 100%** (3/3 dev, 1/1 test) và giữ nguyên Hit@1/Hit@3 bằng B (12/15 dev, 5/5 test).
- Dù chưa tạo cách biệt điểm số trên tập 20 câu nhỏ này (vì ngữ nghĩa câu hỏi đã đủ đặc trưng để B tìm đúng `section`), kiểm tra trace cho thấy filter giúp **loại bỏ nhiễu ngoài phạm vi**: ví dụ ở `q09` (hỏi về *Part 1*), B trả về Rank 2 là `P07 (executive_summary, 0.756)`, còn C loại bỏ được `executive_summary` và thay bằng `P09`, `P14` đều thuộc `part1_foundational`.
- **Hạn chế của metadata `section`:** Chỉ có ích khi người dùng biết trước hoặc nêu rõ phần tài liệu; nếu đoán sai hoặc suy diễn filter thì có thể lọc nhầm mất trang đúng.

### Hai ví dụ lỗi cụ thể
1. **`q17` (test) — Lỗi cắt ngang bằng chứng của Chiến lược A:**
   - **Nguồn gold:** `nist-ai-100-1`, Trang `19` (`part1_foundational`).
   - **Hiện tượng:** Chiến lược A truy xuất đúng `P19` ở cả Rank 1 (`0.828`) và Rank 3 (`0.787`), nhưng vẫn **trượt Hit@1 và Hit@3** (Grounding chỉ đạt 1/2).
   - **Nguyên nhân:** Đoạn văn về *Accuracy measurements* bị cắt làm đôi ở mốc 500 ký tự ngay trước/trong câu điều kiện (`"Accuracy measurements should always be paired with clearly defined and realistic test sets..."`), khiến không một chunk đơn lẻ nào chứa đủ bằng chứng. Chiến lược B/C giữ trọn đoạn văn này và đạt **Hit@1 (`0.853`)**.
2. **`q09` (dev) — Lỗi tách danh sách bullet ở Chiến lược B & C:**
   - **Nguồn gold:** `nist-ai-100-1`, Trang `24` (`part1_foundational`).
   - **Hiện tượng:** Cả B và C đều đưa `P24` lên **Rank 1 (`0.789`)**, nhưng vẫn **trượt Evidence Hit@3**.
   - **Nguyên nhân:** Trang `24` chứa danh sách gạch đầu dòng dài của mục *4. Effectiveness of the AI RMF*; hàm `by_paragraph` đã tách đoạn giới thiệu và các gạch đầu dòng thành hai chunk khác nhau do vượt ngưỡng/ranh giới dòng, làm `evidence_quote` bị chia qua 2 chunk liên tiếp.

### Chiến lược được chọn & Đánh đổi
- **Chọn Chiến lược C (`by_paragraph`, `max_size=800` + `section` filter khi có ngữ cảnh).**
- **Lý do:** Đạt hiệu quả cao nhất theo tiêu chí đã chốt trước (`dev` Hit@3 = 12/15, `test` Hit@3 = 5/5, Grounding = 2.0/2, Coherence = 1.8/2), số chunk ít hơn 28.9% so với A, và bổ sung cơ chế lọc metadata an toàn, tuân thủ 100% khi có yêu cầu.
- **Đánh đổi:** Tốn công bước phân tích cấu trúc tài liệu để ánh xạ `section` đáng tin cậy; và khi chỉ dùng lọc cứng `section`, nếu người dùng hỏi tổng quát không nêu phần thì C tự động quay về tìm toàn cục như B.

### Giới hạn và Hướng cải tiến
- **Giới hạn:** Tập test chỉ có 5 câu (chỉ mang tính minh họa, không thể suy rộng tổng quát cho mọi corpus); tác giả tự chấm Coherence/Grounding nên có thể tồn tại thiên lệch; chunking vẫn thực hiện theo từng trang PDF nên câu vắt qua ranh giới hai trang (như `P07 → P08`) bị mất ngữ cảnh đầu câu.
- **Công cụ hỗ trợ & Cách kiểm tra (theo `RULES.md`):** Sử dụng trợ lý AI (Antigravity) để hỗ trợ viết khung mã nguồn pipeline và định dạng bảng báo cáo. Toàn bộ dữ liệu, câu hỏi và kết quả được kiểm chứng thực tế bằng cách: (1) chạy script đối chiếu 20/20 `evidence_quote` khớp nguyên văn trên từng trang `gold_page` của file PDF gốc sau chuẩn hóa; (2) chạy trực tiếp `python -m src.run` với model thật `BAAI/bge-small-en-v1.5` và `QdrantVectorStore`; (3) kiểm tra thủ công từng bản ghi top-3 của tập `test` và 10 mẫu Coherence.
- **Cải tiến nếu làm lại:**
  1. Nối text liên trang trước khi chia chunk (nhưng vẫn lưu danh sách `pages` trong metadata để truy vết), giúp khắc phục lỗi câu bị ngắt ở cuối trang PDF.
  2. Gộp các danh sách bullet ngắn với câu dẫn phía trước thay vì tách rời tại `q09`.
  3. Mở rộng thêm **Hybrid retrieval (Dense + BM25)** để bắt chính xác các mã định danh như `GOVERN 1.7`, `MANAGE 1.4` khi ngữ nghĩa câu hỏi gần nhau.
