# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Phùng Gia Bảo
**Nhóm:** F1
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Cosine similarity cao (gần 1.0) có nghĩa là hai vector embedding có hướng gần giống nhau trong không gian nhiều chiều, tức là hai đoạn văn bản mang **ý nghĩa ngữ nghĩa tương đồng** — chúng nói về cùng một chủ đề hoặc truyền tải cùng một thông điệp, dù có thể dùng từ ngữ khác nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "Thoi gian lam viec la 8 gio moi ngay"
- Sentence B: "Nhan vien lam viec tu 8h sang den 5h chieu"
- Tại sao tương đồng: Cả hai câu đều nói về thời gian làm việc của nhân viên, chỉ diễn đạt khác nhau. Embedding model nhận ra ý nghĩa tương đồng nên vector hướng gần nhau. (Score thực tế: 0.5785)

**Ví dụ LOW similarity:**
- Sentence A: "Mon pho bo Ha Noi rat ngon"
- Sentence B: "Quy trinh danh gia hieu suat lam viec"
- Tại sao khác: Hai câu hoàn toàn khác chủ đề — một câu về ẩm thực, một câu về quy trình đánh giá nhân sự. Không có điểm chung về ngữ nghĩa nên vector hướng khác nhau. (Score thực tế: 0.4080)

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity đo **góc** giữa hai vector, không phụ thuộc vào độ dài (magnitude) của vector. Điều này quan trọng vì embedding của đoạn văn dài có thể có magnitude lớn hơn đoạn ngắn, nhưng ý nghĩa có thể giống nhau. Cosine similarity bỏ qua sự khác biệt về scale đó và chỉ tập trung vào **hướng** — đại diện cho ý nghĩa ngữ nghĩa.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Phép tính:*
> - Step size (bước nhảy) = chunk_size − overlap = 500 − 50 = 450
> - Số chunks = ⌈(10,000 − 500) / 450⌉ + 1 = ⌈9,500 / 450⌉ + 1 = ⌈21.11⌉ + 1 = 22 + 1 = **23 chunks**
>
> *Đáp án:* **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Với overlap=100: step = 500 − 100 = 400 → chunks = ⌈(10,000 − 500) / 400⌉ + 1 = ⌈23.75⌉ + 1 = 24 + 1 = **25 chunks** (tăng thêm 2 chunks). Overlap lớn hơn giúp **bảo toàn ngữ cảnh** tại vùng biên giữa các chunk — nếu một câu quan trọng nằm vắt ngang ranh giới cắt, overlap đảm bảo câu đó xuất hiện đầy đủ trong ít nhất một chunk, giúp retrieval chính xác hơn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Quản lý nhân sự & Nội quy doanh nghiệp (Employee Handbook — Sổ tay nhân viên EIV)

**Tại sao nhóm chọn domain này?**
> Domain quản lý nhân sự rất phù hợp cho bài toán RAG vì tài liệu nội bộ (sổ tay nhân viên, nội quy, quy tắc ứng xử, chính sách chế độ) thường dài và có cấu trúc phân mục rõ ràng. Nhân viên thường xuyên cần tra cứu nhanh các chính sách cụ thể (nghỉ phép, kỷ luật, lương thưởng, giờ làm việc...) mà không muốn đọc toàn bộ sổ tay. Hệ thống RAG giúp trả lời chính xác từ đúng phần tài liệu liên quan, tiết kiệm thời gian so với tìm kiếm thủ công.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | chung.txt | Sổ tay nhân viên — Chương Quy định chung | 2,694 | source, extension, doc_id |
| 2 | noi_quy_cong_ty.txt | Sổ tay nhân viên — Chương Nội quy công ty | 8,203 | source, extension, doc_id |
| 3 | quy_dinh_thoi_gian_lam_viec.txt | Sổ tay nhân viên — Quy định thời gian làm việc | 2,288 | source, extension, doc_id |
| 4 | quy_tac_ung_xu.txt | Sổ tay nhân viên — Quy tắc ứng xử | 12,715 | source, extension, doc_id |
| 5 | quy_trinh_chinh_sach_che_do.txt | Sổ tay nhân viên — Quy trình, chính sách, chế độ | 11,152 | source, extension, doc_id |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source | string | "noi_quy_cong_ty.txt" | Cho phép lọc kết quả theo chương/phần cụ thể của sổ tay, tránh trả về chunk từ phần không liên quan |
| extension | string | ".txt" | Phân loại theo định dạng file, đảm bảo tính nhất quán khi xử lý |
| doc_id | string | "noi_quy_cong_ty" | Dùng để xóa/cập nhật toàn bộ chunks của một chương khi nội dung chính sách thay đổi |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu (chunk_size=200):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| chung.txt | FixedSizeChunker (`fixed_size`) | 14 | 192.4 | ❌ Cắt giữa câu |
| chung.txt | SentenceChunker (`by_sentences`) | 8 | 335.8 | ✅ Giữ nguyên câu |
| chung.txt | RecursiveChunker (`recursive`) | 18 | 148.7 | ✅ Cắt theo đoạn/câu |
| noi_quy_cong_ty.txt | FixedSizeChunker (`fixed_size`) | 42 | 195.3 | ❌ Cắt giữa câu |
| noi_quy_cong_ty.txt | SentenceChunker (`by_sentences`) | 27 | 302.8 | ✅ Giữ nguyên câu |
| noi_quy_cong_ty.txt | RecursiveChunker (`recursive`) | 60 | 135.6 | ✅ Cắt theo đoạn |
| quy_dinh_thoi_gian_lam_viec.txt | FixedSizeChunker (`fixed_size`) | 12 | 190.7 | ❌ Cắt giữa câu |
| quy_dinh_thoi_gian_lam_viec.txt | SentenceChunker (`by_sentences`) | 7 | 325.9 | ✅ Giữ nguyên câu |
| quy_dinh_thoi_gian_lam_viec.txt | RecursiveChunker (`recursive`) | 16 | 142.0 | ✅ Cắt theo mục |

### Strategy Của Tôi

**Loại:** RecursiveChunker (chunk_size=500)

**Mô tả cách hoạt động:**
> RecursiveChunker sử dụng danh sách separator theo thứ tự ưu tiên: `["\n\n", "\n", ". ", " ", ""]`. Đầu tiên, nó cố gắng tách theo đoạn (`\n\n`), nếu đoạn vẫn quá dài thì tách theo dòng (`\n`), rồi theo câu (`. `), rồi theo từ (` `), cuối cùng theo ký tự (`""`). Các phần nhỏ được gộp lại cho đến khi đạt ngưỡng `chunk_size`, đảm bảo mỗi chunk vừa đủ lớn để chứa ngữ cảnh mà không vượt quá giới hạn.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu nhân sự (nội quy, quy tắc ứng xử, chính sách chế độ) có cấu trúc phân mục rõ ràng: các chương, điều, khoản được ngăn cách bằng dấu xuống dòng kép. RecursiveChunker tận dụng cấu trúc này để cắt theo ranh giới tự nhiên, giữ nguyên ngữ cảnh của từng chính sách/quy định. Điều này tốt hơn FixedSizeChunker (cắt cứng giữa câu) và linh hoạt hơn SentenceChunker (chunk có thể quá dài khi câu dài).

**Code snippet (nếu custom):**
```python

```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|-------------------|
| noi_quy_cong_ty.txt | best baseline (SentenceChunker) | 27 | 302.8 | Tốt — giữ câu, nhưng chunk dài có thể chứa nhiều chủ đề |
| noi_quy_cong_ty.txt | **RecursiveChunker (của tôi, size=500)** | ~17 | ~480 | Tốt hơn — cắt theo đoạn, mỗi chunk thường trọn 1 điều/khoản |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi (Phùng Gia Bảo) | RecursiveChunker (size=500) | 8/10 | Cắt theo cấu trúc tự nhiên, linh hoạt | Chunk tiếng Việt với model MiniLM (English) bị hạn chế |
| Hoàng Thanh Chiến | RecursiveChunker (size=400) | 9/10 | Cắt theo cấu trúc tự nhiên, linh hoạt | Chi phí embedding cao hơn do số lượng chunk có thể tăng đáng kể|
| Mạc Văn Thanh | RecursiveChunker (size=600, overlap=50) | 7/10 | Chia văn bản theo cấu trúc ngữ nghĩa tự nhiên, giữ được tính liên kết giữa các đoạn | Hiệu quả embedding tiếng Việt còn hạn chế khi kết hợp với model all-MiniLM-L6-v2 |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> RecursiveChunker với chunk_size=500 cho kết quả tốt nhất cho tài liệu nội quy nhân sự vì nó tôn trọng cấu trúc đoạn văn. Các chính sách HR thường được viết theo điều và khoản, RecursiveChunker giữ nguyên ranh giới này thay vì cắt ngang giữa chừng. Tuy nhiên, để cải thiện thêm, nên kết hợp với embedding model hỗ trợ tiếng Việt tốt hơn (ví dụ: `bkai-foundation-models/vietnamese-bi-encoder`).

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng regex `(?<=[.!?])\s` để phát hiện ranh giới câu — tách tại vị trí có khoảng trắng ngay sau dấu chấm/chấm hỏi/chấm than. Sau khi tách, strip whitespace thừa và nhóm các câu lại theo `max_sentences_per_chunk`. Edge case xử lý: text rỗng trả về `[]`, câu không có dấu chấm cuối vẫn được giữ nguyên trong chunk cuối cùng.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Algorithm đệ quy với danh sách separator giảm dần độ ưu tiên. Base case: nếu `len(text) <= chunk_size` thì trả về `[text]`. Recursive case: tách text bằng separator đầu tiên, cố gắng merge các phần nhỏ liền kề cho đến khi đạt `chunk_size`, phần nào quá lớn thì gọi đệ quy `_split()` với separator tiếp theo. Nếu hết separator, fallback cắt theo ký tự.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Mỗi document được tạo thành record chứa `{id, content, metadata, embedding}` qua hàm `_make_record()` và append vào `self._store` (list in-memory). Khi search, tính dot product giữa query embedding và tất cả stored embeddings, sort descending theo score, trả về top-k results. Nếu có ChromaDB thì đồng thời lưu vào collection, nhưng fallback gracefully nếu không cài.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` **lọc trước** (pre-filtering): duyệt `self._store`, chỉ giữ records có metadata khớp toàn bộ điều kiện trong `metadata_filter` (dùng `all()` + `dict.get()`), rồi mới chạy similarity search trên tập đã lọc. `delete_document` lọc bỏ tất cả records có `metadata["doc_id"] == doc_id` bằng list comprehension, trả về `True` nếu có record bị xóa.

### KnowledgeBaseAgent

**`answer`** — approach:
> RAG pattern 3 bước: (1) Gọi `store.search(question, top_k)` để lấy top-k chunks liên quan nhất. (2) Build prompt dạng: `"Based on the following context...\n\nContext:\n[Chunk 1]: ...\n[Chunk 2]: ...\n\nQuestion: ...\nAnswer:"` — inject context trực tiếp vào prompt. (3) Gọi `self.llm_fn(prompt)` để sinh câu trả lời. Prompt structure đảm bảo LLM chỉ trả lời dựa trên context được cung cấp.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.11.1, pytest-9.0.3, pluggy-1.6.0
rootdir: F:\AI thuc chien\Day-07-Lab-Data-Foundations-main
plugins: anyio-4.13.0

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 42 passed in 0.08s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Thoi gian lam viec la 8 gio moi ngay" | "Nhan vien lam viec tu 8h sang den 5h chieu" | high | 0.5785 | ✅ Đúng (cao nhất) |
| 2 | "Hom nay troi nang dep" | "Quy dinh ve dong phuc cong ty" | low | 0.4299 | ✅ Đúng (thấp) |
| 3 | "Nhan vien bi ky luat khi vi pham noi quy" | "Xu ly vi pham quy che lao dong" | high | 0.4401 | ❌ Thấp hơn kỳ vọng |
| 4 | "Che do bao hiem xa hoi cho nguoi lao dong" | "Quyen loi phuc loi cua nhan vien" | high | 0.4621 | ❌ Thấp hơn kỳ vọng |
| 5 | "Mon pho bo Ha Noi rat ngon" | "Quy trinh danh gia hieu suat lam viec" | low | 0.4080 | ✅ Đúng (thấp) |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Pair 3 và Pair 4 bất ngờ nhất: cả hai cặp câu đều nói về cùng chủ đề (kỷ luật lao động và phúc lợi nhân viên) nhưng score chỉ ở mức trung bình (~0.44-0.46). Nguyên nhân là model `all-MiniLM-L6-v2` được huấn luyện chủ yếu trên tiếng Anh, nên khi gặp tiếng Việt không dấu, nó không phân biệt tốt ngữ nghĩa. Điều này cho thấy **việc chọn đúng embedding model phù hợp với ngôn ngữ dữ liệu là cực kỳ quan trọng** — model đa ngôn ngữ hoặc model chuyên tiếng Việt sẽ cho kết quả chính xác hơn nhiều.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries trên implementation với RecursiveChunker (chunk_size=500) và local embedder (all-MiniLM-L6-v2). Dữ liệu sử dụng: 5 file nội quy/chính sách nhân sự (tổng 93 chunks).

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Quy định về thời gian làm việc của nhân viên? | Nhân viên làm việc từ 8h00 đến 17h30, nghỉ trưa 1h30 |
| 2 | Nhân viên cần tuân thủ những quy tắc ứng xử nào? | Tôn trọng luật pháp, đạo đức nghề nghiệp, ứng xử công bằng với đồng nghiệp và khách hàng |
| 3 | Chế độ nghỉ phép của nhân viên như thế nào? | Nhân viên làm việc đủ 12 tháng được 12 ngày nghỉ phép có hưởng lương |
| 4 | Quy trình xử lý vi phạm kỷ luật? | Tuân thủ luật pháp, xử lý theo quy chế lao động, có các mức kỷ luật tương ứng |
| 5 | Nội quy công ty quy định những gì? | Quy định chung về thẻ nhân viên, trang phục, giờ giấc, kỷ luật lao động |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Quy định về thời gian làm việc? | "Nhân viên sẽ được phát Thẻ nhân viên..." (noi_quy_cong_ty.txt) | 0.6577 | ⚠️ Liên quan một phần — đúng file nội quy nhưng chunk về thẻ nhân viên | Trả về nội quy chung, chưa đúng phần thời gian |
| 2 | Quy tắc ứng xử của nhân viên? | "Tôn trọng sự tuân thủ của nhân viên với bộ quy tắc ứng xử..." (quy_tac_ung_xu.txt) | 0.7389 | ✅ Chính xác — đúng file và đúng nội dung quy tắc ứng xử |
| 3 | Chế độ nghỉ phép? | "Nghỉ sinh con: theo quy định luật lao động..." (quy_dinh_thoi_gian_lam_viec.txt) | 0.6360 | ✅ Chính xác — đúng file, chunk về nghỉ phép/nghỉ sinh | Top-2 cũng relevant (12 ngày phép, score 0.6358) |
| 4 | Xử lý vi phạm kỷ luật? | "EIV luôn tôn trọng luật pháp, cam kết tuân thủ..." (quy_tac_ung_xu.txt) | 0.6222 | ⚠️ Liên quan — về tuân thủ luật pháp nhưng chưa cụ thể quy trình kỷ luật |
| 5 | Nội quy công ty quy định gì? | "CHƯƠNG I QUY ĐỊNH CHUNG..." (chung.txt) | 0.6676 | ✅ Chính xác — đúng file quy định chung, giới thiệu tổng quan sổ tay |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 4 / 5

**Phân tích kết quả:**
> Kết quả retrieval tốt hơn đáng kể so với trước đó (4/5 vs 2/5) nhờ việc sử dụng chỉ 5 file dữ liệu nhân sự cùng domain thay vì trộn lẫn nhiều chủ đề khác nhau. Các file có kích thước tương đối cân bằng (2,288 — 12,715 ký tự) giúp tránh tình trạng một file quá lớn "lấn át" kết quả. Query 2 cho kết quả tốt nhất (score 0.7389) vì từ khóa "quy tắc ứng xử" xuất hiện trực tiếp trong nội dung file. Để cải thiện hơn nữa, nên: (1) Dùng model hỗ trợ tiếng Việt tốt hơn, (2) Thiết kế metadata chi tiết hơn (thêm `chapter`, `topic`).

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Việc chia dữ liệu thành các file theo chủ đề rõ ràng (nội quy riêng, quy tắc ứng xử riêng, chính sách chế độ riêng) giúp retrieval chính xác hơn rất nhiều so với gộp tất cả vào một file lớn. Metadata `source` khi đó trở nên có ý nghĩa hơn vì mỗi file đại diện cho một chủ đề cụ thể.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Metadata schema thiết kế tốt (thêm trường `department`, `topic`, `language`) giúp search_with_filter hoạt động hiệu quả hơn nhiều so với search toàn bộ corpus. Kết hợp pre-filtering theo metadata + semantic search cho kết quả chính xác hơn đáng kể.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> (1) Sử dụng multilingual embedding model hoặc model chuyên tiếng Việt (ví dụ: `bkai-foundation-models/vietnamese-bi-encoder` hoặc `intfloat/multilingual-e5-base`). (2) Thiết kế metadata schema phong phú hơn — thêm trường `chapter`, `topic` để có thể filter theo chủ đề (ví dụ: filter `topic=nghi_phep` khi hỏi về nghỉ phép). (3) Tăng overlap cho RecursiveChunker để bảo toàn ngữ cảnh tốt hơn tại ranh giới giữa các điều/khoản.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
