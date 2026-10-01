# Day 14 — Exercises

## AI Evaluation & Benchmarking · Lab Worksheet

**Thời gian làm bài:** 9:15–12:00

**Domain:** OrbitTech Store Customer Support

Điền trực tiếp câu trả lời vào file này. Golden dataset 20 QA được viết một lần
duy nhất trong `golden_dataset.json`, không chép lại toàn bộ vào Markdown.

---

Từ 9:15–9:30, cài môi trường và chạy baseline tests theo `guide_lab.md`.

---

## Part 1 — Warm-up (9:30–9:45)

### Exercise 1.1 — RAGAS Metric Thresholds

Theo bài giảng:

- 0.8–1.0: Good — monitor, maintain.
- 0.6–0.8: Needs work — analyze failures, iterate.
- Dưới 0.6: Significant issues — investigate.

Với từng metric, xác định khi nào score thấp có thể chấp nhận và khi nào là
critical.

| Metric | Acceptable Low Score Scenario | Critical Low Score Scenario | Action Required |
|---|---|---|---|
| Faithfulness | Score ~0.6–0.8: câu trả lời đúng chính sách đổi trả (unopened 30 ngày) nhưng diễn đạt lại và thêm chi tiết phụ vô hại, ví dụ "vui lòng giữ lại hộp gốc" dù corpus không ghi rõ. Không gây thiệt hại tài chính. | Score < 0.6 (đặc biệt < 0.3): bịa chính sách có hậu quả, ví dụ khẳng định "AeroBuds Pro được bảo hành 24 tháng" trong khi corpus quy định 12 tháng, hoặc "máy đã mở hộp vẫn được trả trong 30 ngày và miễn 10% restocking fee". | Thêm faithfulness guardrail: bắt buộc citation/quote evidence cho mỗi claim về thời hạn, phí, điều kiện; reject hoặc gắn cờ câu trả lời có claim không grounded trước khi gửi cho khách. |
| Answer Relevance | Score ~0.6–0.8: trả lời đúng ý hỏi về đổi trả NovaBook 14 nhưng lan man thêm 1–2 câu giới thiệu PulsePhone X không được hỏi. Khách vẫn giải quyết được việc. | Score < 0.6 (đặc biệt < 0.3): lạc intent, ví dụ khách hỏi "mở hộp bị tính phí restocking bao nhiêu?" nhưng bot trả lời về quyền lợi OrbitPlus membership. Khách phải hỏi lại. | Cải thiện prompt clarity và intent detection: làm rõ system prompt OrbitTech, thêm few-shot cho câu hỏi đa nghĩa (return vs warranty vs repair), giới hạn độ dài phần mở rộng. |
| Context Recall | Score ~0.6–0.8: retriever lấy được evidence cốt lõi (window 30/14 ngày) nhưng thiếu 1 chunk phụ như "refund về phương thức gốc trong 5–7 business days". Answer vẫn đúng phần chính. | Score < 0.6: bỏ sót evidence quyết định, ví dụ không retrieve đoạn "opened device 14 ngày + 10% restocking fee" nên bot trả lời sai window/miễn phí. Lỗi thuộc retrieval. | Tăng coverage: tăng top-k, cải thiện chunking theo điều kiện chính sách, bổ sung synonym (return/refund/exchange), rewrite query cho câu hỏi có điều kiện mở hộp/khiếm khuyết. |
| Context Precision | Score ~0.6–0.8: chunk đúng nằm ở top-1/top-2 nhưng kèm 1–2 chunk noise, ví dụ hỏi warranty 24 tháng của NovaBook 14 nhưng kèm thêm chunk shipping. Generator vẫn dùng đúng evidence. | Score < 0.6: top-K toàn noise (ví dụ hỏi warranty nhưng top chunks đều về promotions), chunk liên quan bị đẩy xuống cuối nên generator trả lời dựa trên context sai. | Áp dụng reranking (lexical rerank hoặc cross-encoder), lọc chunk dưới relevance_threshold, cải thiện BM25/query rewriting để chunk liên quan lên đầu. |
| Completeness | Score ~0.6–0.8: thiếu chi tiết nhỏ không đổi quyết định, ví dụ nêu đúng window 30 ngày cho unopened nhưng quên "phần gift-card được hoàn vào gift card thay thế". | Score < 0.6: thiếu điều kiện/exception quyết định, ví dụ quên "defective device verified thì miễn restocking fee" hoặc quên "claim warranty cần order number hoặc proof of purchase". Khách có thể bị từ chối quyền lợi. | Thêm answer checklist và few-shot complete answer cho chính sách có điều kiện; tăng context window cho generator; kiểm tra completeness trước khi finalize. |

### Exercise 1.2 — Bias trong LLM-as-a-Judge

Ba bias thường gặp:

- Position bias: judge ưu tiên answer xuất hiện trước.
- Verbosity bias: judge ưu tiên answer dài hơn.
- Self-preference: judge ưu tiên output giống chính model đó.

**Câu 1: Thiết kế experiment phát hiện position bias với ít nhất hai conditions.**

> *Câu trả lời:* Dùng cùng 1 question OrbitTech (ví dụ "Opened NovaBook 14 được trả trong bao lâu, phí bao nhiêu?") và cùng 2 candidate answers A (đúng: 14 ngày + 10% restocking fee, có citation) và B (sai/thiếu: 30 ngày miễn phí). Tạo 2 conditions với nội dung giống hệt, chỉ đổi thứ tự: Condition A/B (A đứng trước, B đứng sau) và Condition B/A (B đứng trước, A đứng sau). Mỗi condition chạy N lần (ví dụ N ≥ 30), randomize thứ tự trình bày, blind label (judge không biết đâu là đáp án đúng), dùng cùng LLM-as-a-Judge prompt và rubric. Đo win-rate của từng answer và consistency rate (tỉ lệ judge chọn cùng answer khi đổi vị trí). Nếu judge đổi lựa chọn theo vị trí (ví dụ luôn chọn answer đứng trước dù nội dung không đổi, chênh lệch win-rate theo position > ngưỡng, consistency thấp) thì kết luận có position bias. Khắc phục: randomize order, chạy cả 2 chiều rồi average, dùng multiple judges.

**Câu 2: Làm thế nào giảm verbosity bias bằng rubric design?**

> *Câu trả lời:* Thiết kế rubric chấm theo evidence và checklist thay vì độ dài: mỗi claim về thời hạn/phí/điều kiện phải có citation từ corpus mới được điểm; claim không có evidence bị trừ điểm dù answer dài. Tách dimension riêng cho Completeness (đủ ý quyết định) và Clarity/Tone (ngắn gọn), đặt trọng số cho Correctness và Evidence cao hơn. Thêm quy tắc penalty explicit: "answer dài nhưng thêm thông tin ngoài corpus hoặc lặp ý không được cộng điểm; answer ngắn đủ ý được điểm cao hơn answer dài lan man". Có thể đặt length guideline và ví dụ response mẫu ngắn-đúng vs dài-sai để calibrate judge.

**Câu 3: Tại sao cần calibrate LLM judge với human labels?**

> *Câu trả lời:* Vì LLM judge có thể lệch hệ thống (leniency bias cho điểm quá cao, severity bias cho điểm quá thấp, self-preference ưu tiên văn phong giống chính nó) và thiếu am hiểu domain OrbitTech (ví dụ nhầm lẫn return window với warranty period). Calibrate bằng cách lấy một tập mẫu có human labels (chuyên gia hỗ trợ khách hàng chấm theo cùng rubric 1–5), so sánh với điểm của judge để đo agreement (ví dụ Cohen's kappa, correlation), từ đó chỉnh rubric/prompt, đặt threshold và phát hiện drift. Human labels là ground truth để xác nhận judge đáng tin trước khi dùng làm quality gate tự động.

### Exercise 1.3 — Evaluation trong CI/CD

**Câu 1: Chọn threshold để block deployment.**

> Lưu ý: các threshold dưới đây là hypothetical policy thresholds cho CI/CD quality gate (do team tự đặt theo rủi ro nghiệp vụ). Chúng khác với fixed Task 1 code contract trong `template.py` (pass rule `passed = all three scores >= 0.5`, `failure_type` khi `< 0.3`, `overall_score = mean 3 answer metrics`, regression khi drop `> 0.05`): code contract là logic cố định để unit tests kiểm tra, còn policy thresholds là ngưỡng vận hành có thể siết/nới theo từng release.

| Metric | Threshold | Lý do |
|---|---:|---|
| Faithfulness | Block nếu avg < 0.75 | Hallucination về chính sách (window, phí, bảo hành) gây thiệt hại tài chính và pháp lý trực tiếp nên cần ngưỡng chặn strict nhất. |
| Answer Relevance | Block nếu avg < 0.70 | Answer lạc intent khiến khách phải hỏi lại, tăng tải hỗ trợ; ngưỡng thấp hơn Faithfulness một chút vì ít rủi ro trực tiếp hơn bịa chính sách. |
| Completeness | Block nếu avg < 0.70 | Thiếu exception/điều kiện (miễn restocking fee khi defective, proof of purchase) cũng gây từ chối quyền lợi sai; chặn ở mức tương đương Relevance để đảm bảo answer đủ ý quyết định. |

**Câu 2: Khi nào dùng offline evaluation, online evaluation và human review?**

> *Câu trả lời:* Offline evaluation dùng trước khi deploy trên golden dataset cố định 20 QA (stratified sampling Easy/Medium/Hard/Adversarial): reproducible, so sánh được giữa các phiên bản prompt/retriever, làm quality gate chặn release khi metric drop > 0.05. Online evaluation dùng sau khi deploy trên production traffic: theo dõi live metrics, user feedback/thumbs-down, phát hiện drift khi chính sách hoặc hành vi khách thay đổi mà golden dataset chưa cover. Human review dùng cho các case judge không tin cậy: adversarial/prompt-injection, privacy/safety, failure cluster điểm thấp, và định kỳ audit mẫu để calibrate LLM judge với human labels. Quy trình chuẩn: offline gate chặn lỗi lớn → online monitor phát hiện sớm → human review quyết định các case nhạy cảm và bổ sung vào golden dataset (Augment) cho vòng lặp tiếp theo.

---

## Part 2 — Core Coding (9:45–10:40)

Hoàn thiện các TODO bắt buộc trong `template.py`.

### Task 1 — Data Models

- `QAPair`: question, expected answer, gold context, metadata và retrieved contexts.
- `EvalResult`: answer-side scores, optional retrieval scores, pass/failure fields.
- `overall_score()`: trung bình Faithfulness, Relevance và Completeness.

### Task 2 — RAGASEvaluator

Answer-side:

- `evaluate_faithfulness(answer, context)`
- `evaluate_relevance(answer, question)`
- `evaluate_completeness(answer, expected)`

Retrieval-side:

- `evaluate_context_recall(contexts, expected)`
- `evaluate_context_precision(contexts, expected)`

Full pipeline:

- `run_full_eval(..., contexts=None)` luôn tính ba answer metrics.
- Nếu có `contexts`, tính và lưu thêm Context Recall và Context Precision.
- Retrieval scores không làm thay đổi `overall_score()` và pass rule gốc.

### Task 3 — LLMJudge

- `score_response(question, answer, rubric)`
- `detect_bias(scores_batch)`

### Task 4 — BenchmarkRunner

- `run(qa_pairs, agent_fn, evaluator)`
- `generate_report(results)`
- `run_regression(new_results, baseline_results)`
- `identify_failures(results, threshold)`

`BenchmarkRunner.run()` phải truyền `pair.retrieved_contexts` vào
`run_full_eval()`. Report phải có average của hai retrieval metrics.

### Task 5 — FailureAnalyzer

- `categorize_failures(failures)`
- `find_root_cause(failure)`
- `generate_improvement_suggestions(failures)`
- `generate_improvement_log(failures, suggestions)`

Kiểm tra:

```bash
pytest tests/ -v
```

`rerank_by_overlap()` là TODO bonus của Exercise 3.5. Test tương ứng được skip
nếu bạn chưa làm bonus.

---

## Part 3 — Golden Dataset & Real Benchmark (10:40–11:35)

### Exercise 3.1 — Build the Golden Dataset

Thiết kế và validate dataset theo Mục 5–6 trong `guide_lab.md`. Nội dung 20 QA
được điền trực tiếp trong `golden_dataset.json`; phần dưới chỉ ghi lại kết quả
và quyết định thiết kế, không chép lại toàn bộ QA.

**Kết quả dataset**

| Hạng mục | Kết quả |
|---|---|
| Tổng số records | ____ / 20 |
| Easy | ____ / 5 |
| Medium | ____ / 7 |
| Hard | ____ / 5 |
| Adversarial | ____ / 3 |
| Source documents được sử dụng | ____ / 10 |
| Validator status | PASS / FAIL |

**Ba case đại diện cho quyết định thiết kế**

| ID | Difficulty | Source document(s) | Vì sao case phù hợp với difficulty/attack type? |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

**Điểm khó nhất khi xây dựng expected answer hoặc evidence là gì?**

> *Câu trả lời:*

**Xác nhận:**

- [ ] Mọi claim trong expected answer đều có evidence hỗ trợ.
- [ ] Không có questions trùng ý và không dùng kiến thức ngoài corpus.
- [ ] `python validate_golden_dataset.py` báo `PASS`.

### Exercise 3.2 — Benchmark Run

Chạy:

```bash
python domain_assistant.py
python evaluate_answers.py
```

Copy bảng terminal vào đây hoặc điền từ `artifacts/benchmark_results.json`.

| ID | Question (short) | Ctx Recall | Ctx Precision | Faithfulness | Relevance | Completeness | Overall | Passed? | Failure Type |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| E01 | | | | | | | | | |
| E02 | | | | | | | | | |
| E03 | | | | | | | | | |
| E04 | | | | | | | | | |
| E05 | | | | | | | | | |
| M01 | | | | | | | | | |
| M02 | | | | | | | | | |
| M03 | | | | | | | | | |
| M04 | | | | | | | | | |
| M05 | | | | | | | | | |
| M06 | | | | | | | | | |
| M07 | | | | | | | | | |
| H01 | | | | | | | | | |
| H02 | | | | | | | | | |
| H03 | | | | | | | | | |
| H04 | | | | | | | | | |
| H05 | | | | | | | | | |
| A01 | | | | | | | | | |
| A02 | | | | | | | | | |
| A03 | | | | | | | | | |

**Aggregate Report**

- Overall pass rate: ____%
- Avg Context Recall: ____
- Avg Context Precision: ____
- Avg Faithfulness: ____
- Avg Relevance: ____
- Avg Completeness: ____
- Failure type distribution: ____

**Ba cases có Overall Score thấp nhất**

1. ID: ____ | Score: ____ | Failure type: ____
2. ID: ____ | Score: ____ | Failure type: ____
3. ID: ____ | Score: ____ | Failure type: ____

**Nhận xét ngắn:** Metric nào yếu nhất? Kết quả gợi ý vấn đề nằm ở retrieval
hay generation?

> *Câu trả lời:*

### Exercise 3.3 — LLM-as-a-Judge Rubric Design

Thiết kế rubric domain-specific cho OrbitTech Customer Support. Mỗi mức phải
đủ cụ thể để hai người chấm độc lập có thể hiểu giống nhau.

Chọn 3–5 dimensions:

- [ ] Correctness
- [ ] Completeness
- [ ] Relevance
- [ ] Evidence/citation
- [ ] Actionability
- [ ] Safety/privacy
- [ ] Tone/clarity
- [ ] Dimension khác: __________

| Score | Tiêu chí domain-specific | Ví dụ response |
|---:|---|---|
| 5 | | |
| 4 | | |
| 3 | | |
| 2 | | |
| 1 | | |

**Ba edge cases khó chấm**

| Edge Case | Tại sao khó chấm? | Rubric xử lý thế nào? |
|---|---|---|
| | | |
| | | |
| | | |

**Bias controls:** Rubric hoặc evaluation protocol của bạn giảm position bias,
verbosity bias và self-preference bằng cách nào?

> *Câu trả lời:*

### Exercise 3.4 — Framework Comparison (Bonus +5)

Chỉ làm sau khi hoàn thành 3.1–3.3. Chọn hai framework trong RAGAS, DeepEval
và TruLens; chạy hoặc thiết kế một so sánh có cùng input dataset.

| Tiêu chí | Framework 1: ____ | Framework 2: ____ |
|---|---|---|
| Setup complexity | | |
| Metrics available | | |
| CI/CD integration | | |
| Kết quả trên cùng dataset | | |
| Insight rút ra | | |

- Scores có nhất quán không?
- Framework nào strict hơn và vì sao?
- Hai framework có tìm ra cùng failure cases không?

> *Phân tích:*

### Exercise 3.5 — Retrieval Reranking (Bonus +5)

Mục tiêu: kiểm tra việc đổi thứ tự chunks có tăng Context Precision mà không
thay đổi Context Recall hay không.

1. Chọn ít nhất 5 cases từ `artifacts/actual_answers.json`.
2. Tính Context Recall và Context Precision trước rerank.
3. Implement `rerank_by_overlap()` hoặc một reranker khác.
4. Rerank cùng tập chunks, không thêm hoặc xóa chunk.
5. Tính lại hai metrics và giải thích kết quả.

| ID | Recall before | Recall after | Precision before | Precision after | Delta Precision |
|---|---:|---:|---:|---:|---:|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| **Avg** | | | | | |

**Tại sao Recall dự kiến không đổi?**

> *Câu trả lời:*

**Khi nào reranking không đủ và cần sửa retriever/query/chunking?**

> *Câu trả lời:*

---

## Part 4 — Reflection (11:35–11:50)

Hoàn thành `reflection.md` bằng kết quả thật từ Exercise 3.2.

---

## Completion Checklist

Hoàn thành kiểm tra cuối trong khoảng 11:50–12:00.

- [ ] Tất cả required tests pass.
- [ ] `golden_dataset.json` validate thành công.
- [ ] Exercise 3.1 hoàn thành trong file JSON và bảng kết quả phía trên.
- [ ] Exercise 3.2 có năm metrics, aggregate report và ba cases thấp nhất.
- [ ] Exercise 3.3 có rubric 1–5 và bias controls.
- [ ] `reflection.md` có ba failure analyses và regression strategy.
- [ ] Đã copy `template.py` thành `solution/solution.py`.
- [ ] Exercise 3.4 và 3.5 chỉ làm nếu chọn bonus.
