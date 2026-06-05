# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Mạc Văn Thanh]
**Nhóm:** [F1]
**Ngày:** [6/6/2026]

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Viết 1-2 câu:* High cosine similarity (gần 1) có nghĩa là hai vector có hướng rất gần nhau trong không gian vector. Trong NLP, điều này chỉ ra rằng hai đoạn text có ý nghĩa ngữ nghĩa (semantic meaning) rất tương đồng.

**Ví dụ HIGH similarity:**
- Sentence A: "Con mèo đang ngủ trên ghế sofa."
- Sentence B: "Một chú mèo con đang nằm nghỉ trên chiếc ghế dài."
- Tại sao tương đồng: Cả hai câu đều miêu tả cùng một hành động (ngủ/nằm nghỉ) của cùng một chủ thể (con mèo/chú mèo con) tại cùng một địa điểm (ghế sofa/ghế dài).

**Ví dụ LOW similarity:**
- Sentence A: "Con mèo đang ngủ trên ghế sofa."
- Sentence B: "Thị trường chứng khoán hôm nay biến động mạnh."
- Tại sao khác: Hai câu hoàn toàn không liên quan về mặt ngữ nghĩa và chủ đề.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu:* Cosine similarity chỉ quan tâm đến góc (hướng) giữa hai vector, do đó nó đo lường độ tương đồng về ngữ nghĩa mà không bị ảnh hưởng bởi độ dài (magnitude) của văn bản. Khác với Euclidean distance, văn bản dài hơn không bị phạt nếu nội dung vẫn tương đồng.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* Kích thước mỗi bước di chuyển (step) = chunk_size - overlap = 500 - 50 = 450. Số lượng chunks = ceil((10,000 - 50) / 450) = ceil(9950 / 450) = ceil(22.11) = 23.
> *Đáp án:* 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:* Nếu overlap tăng lên 100, step giảm xuống 400, số lượng chunks sẽ tăng lên (ceil(9900/400) = 25 chunks). Overlap nhiều hơn giúp đảm bảo không bị mất ngữ cảnh giữa các đoạn cắt, đặc biệt khi chunk bị cắt ngang một ý quan trọng.

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

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Sample 1 | FixedSizeChunker (`fixed_size`) | 15 | ~100 | Bị cắt ngang câu, đôi khi mất ngữ cảnh |
| Sample 1 | SentenceChunker (`by_sentences`) | 5 | ~300 | Giữ trọn vẹn câu, ngữ cảnh rõ ràng |
| Sample 1 | RecursiveChunker (`recursive`) | 15 | ~90 | Giữ cụm từ/câu nguyên vẹn nếu có thể |

### Strategy Của Tôi

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?* Strategy cắt văn bản đệ quy dựa trên danh sách các ký tự phân cách từ đoạn văn đến từ vựng (ví dụ `\n\n`, `\n`, `. `, ` `). Thuật toán ban đầu cố gắng chia theo mảng rộng nhất, nếu đoạn chia vẫn lớn hơn kích thước chunk_size, nó sẽ đệ quy chia nhỏ tiếp bằng cách dùng ký tự phân cách ở cấp độ thấp hơn. Quá trình lặp lại cho đến khi mọi phần nhỏ hơn chunk_size.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?* Cách tiếp cận đệ quy bảo toàn tốt nhất cấu trúc tự nhiên của văn bản. Nó giúp tránh chia cắt ngẫu nhiên giữa chừng câu hay từ, mang lại các đoạn chunks có ngữ nghĩa trọn vẹn để đưa vào Embedding Model mà không lo vượt giới hạn token.

**Code snippet (nếu custom):**
```python
# Sử dụng trực tiếp class RecursiveChunker đã được thiết kế sẵn.
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|-------------------|
| noi_quy_cong_ty.txt | best baseline (SentenceChunker) | 27 | 302.8 | Tốt — giữ câu, nhưng chunk dài có thể chứa nhiều chủ đề |
| noi_quy_cong_ty.txt | **RecursiveChunker (của tôi, size=500)** | ~17 | ~480 | Tốt hơn — cắt theo đoạn, mỗi chunk thường trọn 1 điều/khoản |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
|(Phùng Gia Bảo) | RecursiveChunker (size=500) | 7/10 | Cắt theo cấu trúc tự nhiên, linh hoạt | Chunk tiếng Việt với model MiniLM (English) bị hạn chế |
| Tôi (Mạc Văn Thanh) | RecursiveChunker (size=600, overlap=50) | 7/10 | Chia văn bản theo cấu trúc ngữ nghĩa tự nhiên, giữ được tính liên kết giữa các đoạn | Hiệu quả embedding tiếng Việt còn hạn chế khi kết hợp với model all-MiniLM-L6-v2 |
| Tôi (Hoàng Thanh Chiến) | RecursiveChunker (size=400) | 9/10 | Cắt theo cấu trúc tự nhiên, linh hoạt | Chi phí embedding cao hơn do số lượng chunk có thể tăng đáng kể|

**Strategy nào tốt nhất cho domain này? Tại sao?**
> RecursiveChunker với chunk_size=500 cho kết quả tốt nhất cho tài liệu nội quy nhân sự vì nó tôn trọng cấu trúc đoạn văn. Các chính sách HR thường được viết theo điều và khoản, RecursiveChunker giữ nguyên ranh giới này thay vì cắt ngang giữa chừng. Tuy nhiên, để cải thiện thêm, nên kết hợp với embedding model hỗ trợ tiếng Việt tốt hơn (ví dụ: `bkai-foundation-models/vietnamese-bi-encoder`).

---


## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions
s
**`SentenceChunker.chunk`** — approach:
> *Viết 2-3 câu: dùng regex gì để detect sentence? Xử lý edge case nào?* Sử dụng regex `(?<=\.)\s+|(?<=!)\s+|(?<=\?)\s+|(?<=\.)\n` để tách câu dựa vào các dấu ngắt câu `. ! ?` đi liền với khoảng trắng hoặc ký tự xuống dòng. Nhờ lookbehind `(?<=...)`, các dấu kết thúc câu được giữ nguyên ở cuối từ mà không bị mất, giúp văn bản tự nhiên hơn. Sau đó nhóm các câu lại không vượt quá biến max_sentences_per_chunk.

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Viết 2-3 câu: algorithm hoạt động thế nào? Base case là gì?* Đệ quy duyệt qua các separators theo độ ưu tiên (từ `\n\n` đến ký tự rỗng `""`). Base case là khi chuỗi truyền vào nhỏ hơn `chunk_size` hoặc khi danh sách separators đã cạn. Ở mỗi hàm đệ quy, nếu ghép 2 phần bị vượt quá `chunk_size` thì sẽ tách hoặc đệ quy chuỗi đó với separators tiếp theo.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Viết 2-3 câu: lưu trữ thế nào? Tính similarity ra sao?* Hỗ trợ lưu trữ song song ChromaDB hoặc mảng dict in-memory (fallback). Trong bộ nhớ, thực hiện embed nội dung rồi lưu thành các objects dictionary. Hàm search gọi `_dot(query_emb, r["embedding"])` để tính độ tương tự Cosine (Cosine Similarity) sau đó xếp hạng giảm dần chọn K cao nhất.

**`search_with_filter` + `delete_document`** — approach:
> *Viết 2-3 câu: filter trước hay sau? Delete bằng cách nào?* Phép lọc (filter) phải được áp dụng *trước* phép similarity search để thu hẹp không gian tìm kiếm, chỉ tính khoảng cách trên các bản ghi phù hợp. Quá trình delete được cấu trúc để lặp qua danh sách bộ nhớ và gỡ bỏ các bản ghi có giá trị meta `doc_id` trùng khớp với id cung cấp.

### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu: prompt structure? Cách inject context?* Gọi trực tiếp phương thức `self.store.search` để lấy các chunk phù hợp nhất từ kiến thức. Trích xuất text trong trường `"content"` của chunks thành chuỗi Context, rồi nối chuỗi theo format template `Context: \n {context} \n\n Question: {question}` rồi pass vào hàm `llm_fn` để tạo sinh kết quả cuối cùng.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.13.13, pytest-9.0.3, pluggy-1.6.0
rootdir: C:\Users\Thanh\Desktop\lap-7\Day-07-Lab-Data-Foundations
plugins: anyio-4.12.1
collected 42 items

tests\test_solution.py ..........................................        [100%]

============================= 42 passed in 0.26s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)
| Pair | Sentence A                                      | Sentence B                                         | Dự đoán | Actual Score | Đúng?              |
| ---- | ----------------------------------------------- | -------------------------------------------------- | ------- | ------------ | ------------------ |
| 1    | "Nhan vien lam viec 8 gio moi ngay tai cong ty" | "Cong ty quy dinh nhan vien lam viec 8 tieng"      | high    | 0.5785       | Đúng (cao nhất)  |
| 2    | "Buoi sang hom nay thoi tiet rat dep"           | "Noi quy cong ty yeu cau mac dong phuc"            | low     | 0.4299       | Đúng (thấp)      |
| 3    | "Nhan vien se bi xu ly neu vi pham noi quy"     | "Cong ty ap dung hinh thuc xu phat lao dong"       | high    | 0.4401       | hấp hơn kỳ vọng |
| 4    | "Nguoi lao dong duoc tham gia bao hiem xa hoi"  | "Nhan vien duoc huong cac chinh sach phuc loi"     | high    | 0.4621       |Thấp hơn kỳ vọng |
| 5    | "Pho bo la mon an noi tieng cua Ha Noi"         | "Doanh nghiep danh gia hieu qua lam viec hang nam" | low     | 0.4080       | Đúng (thấp)      |

## Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?

> Pair 3 và Pair 4 là hai trường hợp đáng chú ý nhất vì nội dung giữa các câu có liên quan khá chặt chẽ về mặt ngữ nghĩa, đều thuộc lĩnh vực quản lý nhân sự và chính sách lao động. Tuy nhiên, similarity score chỉ đạt khoảng 0.44–0.46, thấp hơn mong đợi.
>
> Điều này cho thấy model `all-MiniLM-L6-v2` chưa tối ưu cho tiếng Việt, đặc biệt với dữ liệu không dấu. Model vẫn nhận diện được một phần ý nghĩa chung giữa các câu nhưng chưa biểu diễn tốt mức độ tương đồng ngữ nghĩa sâu. Qua đó có thể thấy rằng hiệu quả của embeddings phụ thuộc rất nhiều vào ngôn ngữ huấn luyện và chất lượng dữ liệu đầu vào. Nếu sử dụng model đa ngôn ngữ hoặc model được huấn luyện chuyên cho tiếng Việt, kết quả thường sẽ chính xác và ổn định hơn.


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
| 1 | Quy định về thời gian làm việc? | chunk_51: "CHƯƠNG IV ... THỜI GIAN LÀM VIỆC VÀ NGHỈ NGƠI ..." | 0.6226 | Có | Làm việc từ 8h00 - 17h30, có nghỉ trưa |
| 2 | Quy tắc ứng xử của nhân viên? | chunk_11: "thể thành viên EIV... Thưởng, chi động viên..." | 0.5549 | Không | Trả lời sai do lấy nhầm chunk về lương thưởng |
| 3 | Chế độ nghỉ phép? | chunk_21: "Đối với các trường hợp nghỉ từ 30 ngày trở lên thì phải nộp đơn..." | 0.6310 | Có | Cần sự đồng ý của Giám đốc và thông báo cho HCNS |
| 4 | Xử lý vi phạm kỷ luật? | chunk_2: "Chúc bạn luôn thành công trong công việc..." | 0.5358 | Không | Câu trả lời không liên quan đến kỷ luật |
| 5 | Nội quy công ty quy định gì? | chunk_2: "Chúc bạn luôn thành công trong công việc..." | 0.6562 | Không | Lấy nhầm đoạn kết của sổ tay, không trả lời được |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 2 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được cách chia chunk theo từng câu (Sentence Chunker) để tránh việc một chunk chứa quá nhiều ý khác nhau, giúp kết quả tìm kiếm chính xác hơn ở cấp độ câu hỏi hẹp.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Nhóm khác đã sử dụng metadata filter (lọc theo nguồn file) rất hiệu quả trước khi tìm kiếm vector, giúp loại bỏ hẳn những kết quả sai lệch thuộc các chương khác của sổ tay.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ không chỉ dùng RecursiveChunker thuần túy mà sẽ kết hợp đánh dấu Markdown (Header Chunker) để nhóm các điều khoản trong sổ tay lại với nhau. Đồng thời, tôi sẽ đổi mô hình embedding sang loại hỗ trợ tiếng Việt tốt hơn thay vì mô hình tiếng Anh (all-MiniLM-L6-v2) vì hiện tại mô hình bắt ngữ nghĩa tiếng Việt còn rất kém.

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
