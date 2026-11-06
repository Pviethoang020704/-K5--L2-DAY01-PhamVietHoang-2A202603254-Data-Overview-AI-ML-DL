# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** 3.13.18 / 2.11.0+cpu / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này là kết quả phân loại toàn bộ bức ảnh, không phải một vật thể riêng lẻ, bởi vì mô hình Classification dự đoán nhãn ở cấp độ ảnh mà không trả về vị trí của từng vật thể.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách lớp đi kèm taxonomy/checkpoint đã huấn luyện, không phải do AI tự nghĩ ra.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Cần lưu cả ba thông tin để xác định chính xác lớp, giúp con người đọc hiểu và tránh nhầm lẫn giữa các bộ nhãn khác nhau.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần quy định rõ tiêu chí lựa chọn một nhãn chính khi ảnh có nhiều vật thể, ví dụ ưu tiên chủ thể nổi bật, vật thể chiếm diện tích lớn hoặc chủ đề chính của ảnh. Với ảnh traffic có nhiều phương tiện khác nhau, người gán nhãn phải dựa trên quy tắc thống nhất thay vì tùy ý chọn một loại xe.
6. Vì sao không được lấy `score` của model làm ground truth?
Không thể sử dụng score của mô hình làm ground truth vì score chỉ thể hiện mức độ tin cậy của mô hình đối với dự đoán, không đảm bảo dự đoán chính xác. Ví dụ, mô hình dự đoán nhãn cab với score 0.510915 nhưng ảnh traffic có nhiều phương tiện khác nhau. Ground truth cần được xác định theo guideline và kiểm tra bởi con người.
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=bus ; `score`=0.912557 ; `bbox_xyxy`=[93.17, 187.95, 223.01, 320.91] ; `bbox_width`=129.84 ; `bbox_height`=132.96
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Bounding box bao quanh chiếc xe bus có góc trên bên trái tại tọa độ (93.17, 187.95), góc dưới bên phải tại (223.01, 320.91). Khung có chiều rộng 129.84 pixel và chiều cao 132.96 pixel. Các tọa độ được tính từ góc trên bên trái của ảnh.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Ngưỡng 0.20: 17 prediction.
Ngưỡng 0.60: 6 prediction.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Khi hạ ngưỡng confidence, số lượng bounding box được giữ lại thường tăng lên. Reviewer phải kiểm tra nhiều dự đoán hơn, xác định các hộp sai, nhãn sai, hộp trùng lặp hoặc không bao đúng vật thể, sau đó yêu cầu chỉnh sửa theo guideline.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Khi gán nhãn bounding box, phải vẽ hình chữ nhật nhỏ nhất bao sát toàn bộ phần nhìn thấy của vật thể, không cắt mất các phần nhìn thấy và hạn chế khoảng trống không cần thiết. Áp dụng cùng một nguyên tắc cho tất cả ảnh.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Trong ảnh kitchen, có một người ở mép trái ảnh (person, score 0.61), nhưng chỉ nhìn thấy một phần cánh tay và bàn tay, phần còn lại của cơ thể nằm ngoài khung hình.

Tôi đề xuất đưa trường hợp này vào guideline, quy định rằng khi vật thể bị cắt bởi mép ảnh, annotator chỉ vẽ bounding box bao quanh phần vật thể nhìn thấy trong ảnh, không suy đoán phần nằm ngoài khung hình. Nếu phần nhìn thấy quá ít và không thể xác định chắc chắn loại vật thể, cần hỏi Lab Coach (escalation).
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[
        446.0,
        70.0
      ],
      [
        445.0,
        71.0
      ],
      [
        444.0,
        71.0
      ],
      [
        443.0,
        72.0
      ],
      [
        442.0,
        72.0
      ],
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Polygon cho biết đường viền và hình dạng chi tiết của vật thể, thay vì chỉ dùng hình chữ nhật như bounding box. Nhờ đó có thể xác định chính xác hơn vùng pixel thuộc về từng vật thể, đặc biệt với những vật có hình dạng không đều.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id dùng để phân biệt các đối tượng riêng biệt trong cùng một kết quả phân đoạn. Ví dụ, nếu có 3 chiếc bát, chúng cùng thuộc class bowl nhưng mỗi chiếc có một instance_id khác nhau.

instance_id không phải class_id vì nó không biểu thị loại vật thể. Nó cũng không phải tracking_id vì không dùng để theo dõi cùng một vật thể qua nhiều khung hình video.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Khi gán nhãn segmentation, annotator phải vẽ đường biên mask bám sát đường viền phần nhìn thấy của vật thể, không lấn sang các vật thể khác hoặc vùng nền. Với phần bị che khuất, không tự suy đoán hình dạng nếu guideline không yêu cầu.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Trong ảnh kitchen, vùng cánh tay bên phải của người đầu bếp (person, score 0.90) nằm sát các vật dụng trong bếp, khiến đường biên giữa người và các vật thể xung quanh khó xác định chính xác.

Tôi đề xuất sử dụng guideline để quy định annotator phải vẽ mask bám sát phần nhìn thấy của cơ thể, không lấy cả vùng nền hoặc vật thể bên cạnh. Nếu ranh giới quá mờ, không thể xác định rõ theo guideline thì cần hỏi Lab Coach (escalation), tránh tự suy đoán.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
|---|---|---|---|---|
| Phân loại ảnh | Một nhãn lớp chính cho toàn bộ ảnh, gồm class_id và class_name theo taxonomy. | Ảnh traffic có nhiều loại phương tiện nhưng mô hình dự đoán cab với score 0.510915. Khó xác định một lớp đại diện cho toàn ảnh. | Quan sát toàn bộ ảnh, chọn nhãn chính theo guideline và taxonomy. | Kiểm tra nhãn có phù hợp với nội dung ảnh và quy tắc chọn lớp hay không. |
| Phát hiện vật thể | Mỗi vật thể có một nhãn lớp và bounding box xác định vị trí bằng 4 tọa độ. | Trong ảnh kitchen, người ở mép trái (person, score 0.61) bị cắt bởi mép ảnh, chỉ thấy một phần cơ thể. | Xác định từng vật thể, gán class và vẽ box bao sát phần nhìn thấy theo guideline. | Kiểm tra nhãn, tọa độ box, vật thể bị bỏ sót, hộp trùng hoặc hộp vẽ sai. |
| Instance segmentation | Mỗi đối tượng có nhãn lớp và mask/polygon xác định vùng pixel thuộc vật thể. | Trong ảnh kitchen, vùng cánh tay bên phải người đầu bếp (person, score 0.90) nằm sát khu vực bếp, gây khó khăn khi xác định đường biên mask. | Gán nhãn từng đối tượng, vẽ mask bám sát đường viền phần nhìn thấy. | Kiểm tra mask có lấn sang nền hoặc vật thể khác, có bỏ sót vùng hoặc sai đường biên không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ sử dụng dữ liệu được cung cấp trong phạm vi bài thực hành, không đưa thông tin cá nhân hoặc dữ liệu nhạy cảm vào báo cáo, không tự ý chia sẻ dữ liệu ra bên ngoài khi chưa được phép.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Tôi sẽ dừng xử lý dữ liệu đó và báo cho Lab Coach hoặc giảng viên phụ trách để được hướng dẫn, không tự ý sử dụng hay chỉnh sửa dữ liệu ngoài phạm vi bài học.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
