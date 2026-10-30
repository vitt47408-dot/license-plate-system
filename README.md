# Hệ thống nhận diện biển số và quản lý xe ra vào

## 1. Mô hình AI phù hợp

Với yêu cầu thiết kế trên Windows và CPU, lựa chọn thực tế nhất là kết hợp:

- Detected plate region: bộ phát hiện vùng biển số bằng YOLO plate detector nếu có trọng số riêng biệt (best option), hoặc fallback bằng OpenCV contour detection.
- OCR: EasyOCR hỗ trợ tiếng Việt/Anh và chạy trên CPU, tải model lần đầu chạy.

Lý do chọn cách này:

- Không có một mô hình biển số xe Việt Nam “one-click” có sẵn trong gói Python để chạy ngay trên mọi máy Windows với CPU.
- Cách an toàn và thực tiễn nhất là thử dùng mô hình plate detector chuyên dụng từ Roboflow Universe hoặc repo Ultralytics-compatible; nếu không có, ứng dụng vẫn hoạt động bằng bộ phát hiện contour OpenCV để cung cấp MVP thực tế.
- Điều này đáp ứng yêu cầu “không giả định YOLO COCO generic phát hiện biển số".

Nguồn đề xuất:

- https://github.com/ultralytics/ultralytics
- Roboflow Universe: tìm model "license plate detection" hoặc "vehicle number plate"; chỉ dùng file `.pt` tương thích YOLO.
- Dataset tham khảo cho fine-tuning tại Việt Nam: https://www.kaggle.com/datasets/tanhphp/vietnamese-license-plates. Kiểm tra license/điều khoản Kaggle và định dạng nhãn trước khi dùng.

Lưu ý: không đặt một file trọng số giả mạo hoặc tạo kết quả nhận diện giả; nếu không có mô hình chuyên dụng, ứng dụng sẽ cảnh báo rõ và chuyển sang fallback contour để tránh nhầm lẫn.

## 2. Cấu trúc dự án

```text
license_plate_system/
├── app.py
├── .streamlit/
│   └── config.toml   (giao diện tối và màu thương hiệu)
├── database.py
├── plate_detector.py
├── ocr_engine.py
├── requirements.txt
├── README.md
├── license_plate.db
├── models/
│   ├── plate_detector_yolov8n.pt  (tùy chọn)
│   └── README.txt
├── uploads/
├── evidence/
├── exports/
├── runs/
├── sample_data/
│   └── README.md
└── .venv/   (khi tạo môi trường ảo)
```

## 3. Tạo môi trường Python trên Windows (CMD)

```cmd
cd c:\Users\Admin\Desktop\license_plate_system
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

## 4. Cài đặt thư viện

```cmd
pip install -r requirements.txt
```

## 5. Tải mô hình pretrained

Mô hình plate detector chuyên dụng không được bundle sẵn trong repo vì phải tải từ nguồn bên ngoài và tùy thuộc vào license. Có 2 phương án:

### Phương án A: dùng mô hình YOLO plate detector chuyên dụng (khuyến nghị)

Tải file trọng số YOLO plate-specific từ nguồn đáng tin cậy, xem kỹ license, rồi đặt tên:

```text
models\plate_detector_yolov8n.pt
```

Không dùng trọng số YOLO COCO generic trực tiếp làm detector, vì bộ class COCO thông thường không có class biển số. Có thể dùng `yolov8n.pt` pretrained chính thức làm **điểm khởi tạo fine-tune** trên dữ liệu biển số; chỉ dùng trọng số `best.pt` sau fine-tune làm detector. Đặt trọng số đã huấn luyện vào đường dẫn trên rồi khởi động lại ứng dụng.

### Phương án B: không tải mô hình plate detector

Hệ thống sẽ tự động dùng fallback OpenCV contour detection và EasyOCR để nhận diện biển số. Trường hợp này vẫn chạy ở mức MVP nhưng cho độ chính xác thấp hơn nếu ảnh không rõ.

## 6. Chạy ứng dụng

```cmd
streamlit run app.py
```

Mở browser tại địa chỉ hiển thị trong terminal, ví dụ:

```text
http://localhost:8501
```

Giao diện mặc định dùng bảng màu tối trong `.streamlit\config.toml`. Nếu thay đổi cấu hình theme khi server đang chạy, hãy khởi động lại Streamlit để áp dụng.

### Mở app trên điện thoại

#### Dùng điện thoại và máy tính chung Wi-Fi

Chạy Streamlit cho phép thiết bị khác trong mạng truy cập:

```cmd
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Trên máy tính, chạy `ipconfig`, lấy địa chỉ IPv4 của adapter Wi-Fi, rồi mở `http://<IPv4-của-máy-tính>:8501` trên điện thoại. Ví dụ: `http://192.168.1.25:8501`. Nếu Windows Firewall hỏi, chỉ cho phép trên mạng Private.

Trong trang **Giám sát trực tiếp**, chọn **Camera trực tiếp (tự động)** và bấm **Bật camera** để cấp quyền. App nhận ảnh định kỳ từ chính thiết bị mở trang, không yêu cầu TURN/STUN.

#### Deploy lên máy chủ có HTTPS

Với Streamlit Community Cloud:

1. Đưa mã nguồn lên một repository GitHub; không đưa `.venv`, `datasets`, `runs`, `license_plate.db`, ảnh trong `evidence`/`uploads` hoặc thông tin bí mật lên repository. File `.gitignore` trong dự án đã loại các dữ liệu này khỏi những lần thêm file mới.
2. Mở [share.streamlit.io](https://share.streamlit.io/), đăng nhập bằng GitHub, chọn **Create app**, rồi chọn repository, branch và `app.py`.
   Repository có `packages.txt` để cài thư viện hệ thống OpenCV cần trên Linux; sau khi thay đổi file này, dùng **Reboot app** để Cloud cài lại dependency.
3. Bấm **Deploy** và đợi dịch vụ cấp URL HTTPS. Mở URL trên điện thoại, vào **Giám sát trực tiếp > Camera trực tiếp (tự động)**, chọn hướng camera và bấm **Bật camera**.

Trong triển khai này, file trọng số `models/plate_detector_yolov8n.pt` được giữ lại để detector YOLO hoạt động; thư mục dữ liệu huấn luyện và checkpoint phụ không được đưa lên. Các mô hình AI chỉ được khởi tạo sau khi bật camera để trang giám sát không phải tải chúng ngay khi mở. Nếu repository không thể chứa trọng số, tải nó lên nơi lưu trữ phù hợp và cấu hình bước tải trọng số trước khi chạy app.

**Lưu ý triển khai:** app không có đăng nhập/mật khẩu. Ai biết URL Cloud có thể truy cập, xem và thao tác dữ liệu (bao gồm hồ sơ xe, lịch sử, ảnh, backup/restore và huấn luyện); chỉ chia sẻ URL với người được phép. Camera trình duyệt cần HTTPS và quyền camera. Hình ảnh được gửi đến ứng dụng qua kết nối HTTPS/WebSocket hiện có, không dùng WebRTC, STUN hoặc TURN; máy chủ nhận khung hình khoảng mỗi 1,1 giây và xử lý nhận diện trên app. Điều này tốn băng thông và tài nguyên máy chủ hơn truyền video trực tiếp. Chọn hướng theo vị trí camera vì app không suy ra chiều xe chỉ từ một khung hình. Khung xanh phân biệt **CAR/MOTORCYCLE/BUS/TRUCK** (xe) và **BIEN SO** (biển số). Khi không có `models/yolov8n.pt`, ứng dụng yêu cầu Ultralytics tự tải trọng số xe chính thức `yolov8n.pt` vào máy chủ; lần khởi động đầu cần Internet. Tự ghi nhận dựa trên biển số đúng định dạng; OCR từ 65% cần hai lần đọc, OCR 40–64% cần ba lần đọc khớp để hạn chế ghi sai. Model xe chỉ vẽ khung tham khảo nên không bắt buộc phải thấy toàn thân xe trong khung hình. Cùng biển số/hướng bị lọc trùng 25 giây.

#### Lưu dữ liệu bền vững trên Streamlit Cloud (tùy chọn)

SQLite và filesystem vẫn được dùng mặc định khi không cấu hình dịch vụ bên ngoài. Để dữ liệu không mất khi Cloud khởi động lại, thêm các Secrets sau tại **App settings > Secrets**. Không đưa secrets vào Git hoặc gửi qua chat:

```toml
DATABASE_URL = "PASTE_YOUR_PROVIDER_POSTGRES_CONNECTION_URI"

S3_BUCKET = "ten-bucket"
S3_ACCESS_KEY_ID = "access-key"
S3_SECRET_ACCESS_KEY = "secret-key"
S3_ENDPOINT_URL = "https://endpoint-cua-nha-cung-cap"
S3_REGION = "auto"
S3_ADDRESSING_STYLE = "path"
S3_PREFIX = "license-plate-system"
```

`DATABASE_URL` bật PostgreSQL; app tự tạo/cập nhật các bảng khi khởi động. PostgreSQL cần cho phép kết nối SSL từ Streamlit Cloud. Để dùng AWS S3, có thể bỏ `S3_ENDPOINT_URL` và đặt region AWS phù hợp; với R2/MinIO hoặc dịch vụ tương thích S3, dùng endpoint, region và addressing style của nhà cung cấp. `S3_ENDPOINT_URL` và `S3_REGION` phụ thuộc dịch vụ. Cần cả ba giá trị bucket/access key/secret key; nếu chỉ cấu hình một phần app sẽ báo lỗi thay vì âm thầm quay về lưu local. App lưu ảnh bằng chứng, mẫu OCR và bản sao tệp tải lên vào bucket; ảnh được tải qua server app và bucket không cần public.

Nếu cấu hình chỉ PostgreSQL, ảnh vẫn lưu local nên Cloud có thể làm mất ảnh sau khi khởi động lại. Nếu cấu hình chỉ S3, thông tin xe/lịch sử vẫn ở SQLite tạm thời. Để bền vững đầy đủ cần cấu hình cả PostgreSQL và object storage. Mỗi lần đổi `DATABASE_URL`/S3 credentials cần **Reboot app**.

Để chuyển dữ liệu hiện có: trước khi cấu hình backend mới, tải ZIP tại **Cấu hình > Sao lưu và khôi phục**; cấu hình PostgreSQL/S3 rồi Reboot; tải ZIP đó lên cùng mục và khôi phục. Khôi phục sẽ thay toàn bộ dữ liệu database đích, đồng thời tải ảnh/file trong ZIP lên object storage nếu đã cấu hình. Tạo/tải backup hiện tại trước khi khôi phục. Sau đó dùng **Tạo bản sao lưu** định kỳ và lưu ZIP ở nơi riêng; backup chứa thông tin cá nhân, lịch sử và ảnh nên cần được bảo vệ. Các khóa S3 cần quyền list/get/put trong bucket/prefix đã cấu hình; thao tác rollback ảnh OCR cần quyền delete.

Camera không cần cấu hình tài khoản TURN/STUN. Mở app bằng HTTPS, cho phép quyền camera của trình duyệt rồi bấm **Bật camera**. Nếu không mở được camera, kiểm tra quyền camera của trang/trình duyệt và thử tải lại app. Khi dùng tính năng ghép điện thoại với máy tính, hình ảnh đi qua máy chủ ứng dụng nên cả hai thiết bị phải truy cập cùng deployment đang hoạt động; luồng video không được lưu thành file.

## 7. Kiểm thử MVP

### 7.1. Giám sát trực tiếp và nhận diện
- Vào **Giám sát trực tiếp**, chọn **Camera trực tiếp (tự động)**, chọn hướng **XE VÀO/XE RA**, bấm **Bật camera** và cấp quyền camera. Ảnh từ webcam/camera điện thoại được gửi qua HTTPS khoảng mỗi 1,1 giây để nhận diện.
- App xử lý nhận diện nền để video preview không phải chờ YOLO/OCR; kết quả mới nhất được dùng để vẽ khung xanh thân xe và biển số. App nhận diện gần mỗi 1 giây và tự lưu khi biển số đúng định dạng, OCR đạt từ 65% trong hai lần đọc hoặc 40–64% trong ba lần đọc khớp; không cần xác nhận từng biển số hoặc phát hiện khung thân xe. Với kết quả OCR rõ, app dừng sớm sau lượt nhận dạng đầu; nếu chưa rõ, app vẫn thử các ảnh tăng cường để giữ độ nhạy. Kiểm tra danh sách sự kiện tự động để rà lại kết quả vì OCR vẫn có thể đọc sai.
- Luồng video giới hạn ở 960×540, tối đa 15 FPS; mô hình xe dùng ảnh suy luận 480 px và giới hạn 5 kết quả để giảm tải khi chạy CPU. Biển số được OCR trên vùng crop gốc để giữ chi tiết.
- Để ghép điện thoại với máy tính, chọn **Liên kết camera từ xa** trên cả hai thiết bị. Trên điện thoại, chọn **Phát camera từ điện thoại**, tạo mã rồi bấm **Bật camera**; trên máy tính, chọn **Xem camera trên máy tính** và nhập mã. Mã ghép có hiệu lực 30 phút; khởi động lại ứng dụng sẽ ngắt liên kết và cần tạo mã mới. Khung hình được chuyển tiếp trong bộ nhớ tiến trình ứng dụng, không lưu thành video.
- Camera trực tiếp cần HTTPS và kết nối WebSocket tới ứng dụng; không cần mở kết nối STUN/TURN riêng.
- Nếu camera không mở, kiểm tra quyền camera của trình duyệt/Windows tại **Settings > Privacy & security > Camera**, rồi tải lại trang và bấm **Bật camera** lần nữa.
- Có thể chọn **Ảnh/video (thủ công)** nếu cần quét riêng tệp; đây là luồng riêng, không thay cho camera trực tiếp.
- Tại **Giám sát trực tiếp** cũng có ô **Tải video để liệt kê và phân tích biển số** độc lập; chọn MP4/AVI/MOV/MKV để xem video ngay trên màn hình rồi bấm **Phát video và phân tích biển số**. Video phát trong trình duyệt trong khi ứng dụng quét nền các khung hình được lấy mẫu. Mỗi kết quả hiển thị khung hình gốc, ảnh biển số phóng to, nội dung OCR và độ tin cậy để kiểm tra trước khi xác nhận.
- Với video, thanh trượt chỉ nhận diện khung hình đang chọn. Để tìm nhiều biển số trong clip, bấm **Quét tìm biển số trong toàn video**; ứng dụng lấy mẫu tối đa 2 khung hình/giây (không quá 300 khung hình), gộp kết quả trùng nhau rồi yêu cầu kiểm tra từng biển số, chọn hướng và xác nhận trước khi lưu. Video dài hoặc biển số nhỏ/mờ có thể cần thử một khung hình khác.
- OCR thử ảnh màu, grayscale và tăng tương phản; nếu crop nhỏ hơn khoảng 80×32 px, hãy chọn khung rõ hơn hoặc đưa camera lại gần/lấy nét. Phóng lớn bằng phần mềm không thể khôi phục chi tiết không có trong ảnh; confidence của detector cũng không xác nhận rằng khung đó thực sự chứa biển số.
- Hệ thống hiển thị:
  - ảnh camera có khung phát hiện và ảnh zoom vùng biển số ở cột trái
  - bảng lịch sử xe ra/vào ở giữa
  - thẻ **Chi tiết sự kiện** bên phải với biển số, thời gian, hướng di chuyển, độ tin cậy OCR và thông tin đăng ký
  - ô xác nhận/sửa biển số trước khi lưu
- Với **Ảnh/video (thủ công)**, kết quả OCR chỉ là bản xem trước và chưa có trong lịch sử. Chọn hướng XE VÀO/XE RA, xác nhận/sửa biển số, tích **Tôi đã kiểm tra và xác nhận biển số chính xác**, rồi bấm **Ghi nhận vào lịch sử**. Camera trực tiếp tự lọc/lưu theo ngưỡng, không yêu cầu xác nhận từng lượt.
- Dashboard tính **Tổng xe trong bãi** từ lượt vào/ra đã xác nhận và **Xe vào hôm nay** từ lịch sử trong ngày. Đặt sức chứa tại **Cấu hình > Sức chứa khu vực** để tính **Chỗ trống**; nếu chưa đặt, ứng dụng không tự bịa sức chứa.

### 7.2. Tải video
- Tải video MP4/AVI và chọn một khung hình để xem nhanh.
- Bấm **Quét tìm biển số trong toàn video** để nhận diện nhiều biển số trên các khung hình được lấy mẫu; video được đọc tuần tự (không seek riêng từng mẫu), khung lớn được thu nhỏ và detector dùng ảnh 640 px để hỗ trợ biển số nhỏ. Tối đa 25 kết quả OCR khác nhau được giữ làm ảnh xem trước. Nếu có nhiều hơn, app giữ kết quả có độ tin cậy cao nhất và thông báo đã giới hạn kết quả; kiểm tra/sửa từng biển số và chỉ xác nhận các xe cần ghi nhận.
- Ô tải video độc lập cũng liệt kê kết quả OCR duy nhất và hiển thị ảnh crop phóng to cạnh khung hình đã phát hiện; hỗ trợ MP4, AVI, MOV và MKV.
- Mỗi biển số khác nhau chỉ xuất hiện một lần trong danh sách quét; các lần đọc trùng được gộp lại. Quét không thay thế việc người vận hành xác minh hướng và nội dung biển số.

### 7.3. Lưu lịch sử xe vào/ra
- Với camera trực tiếp, chọn hướng camera IN/OUT; ứng dụng tự lọc OCR và ghi biển số hợp lệ cùng ảnh bằng chứng.
- Ảnh/video tải lên chỉ được ghi vào lịch sử sau khi người vận hành kiểm tra, chọn hướng và xác nhận lưu.
- Hệ thống lưu vào SQLite và ngăn ghi trùng khung hình liên tiếp.
- Lịch sử và thẻ chi tiết sự kiện hiển thị ở màn hình giám sát; các sự kiện cần xác minh không được tính vào trạng thái xe trong khu vực.

### 7.4. Xuất báo cáo
- Vào **Báo cáo**.
- Chọn khoảng thời gian; biểu đồ và tổng số lượt được tính trên đúng khoảng ngày đó.
- Tải CSV hoặc Excel về máy; bản sao cũng được lưu trong thư mục `exports`.

### 7.5. Xe bất thường và xác minh

- Dashboard hiển thị sự kiện biển số chưa có hồ sơ, OCR confidence thấp, sự kiện chờ xác minh và xe có lượt vào nhưng chưa có lượt ra tương ứng.
- Đây là cảnh báo để người vận hành kiểm tra, không phải kết luận vi phạm.
- Vào trang Lịch sử ra vào để xác nhận/sửa biển số trong hàng chờ xác minh. Thao tác được ghi vào nhật ký.
- Ảnh thiếu, biển số chưa đăng ký và hồ sơ bị hạn chế cũng hiển thị rõ để rà soát.

### 7.6. Ghi chú huấn luyện và đánh giá mô hình

- Giao diện có mục **Huấn luyện model** để tải dataset ZIP và fine-tune YOLO trực tiếp trong app. Trên CPU/Streamlit Cloud, bắt đầu với vài epoch vì lượt train dài có thể bị giới hạn tài nguyên hoặc bị ngắt khi app khởi động lại.
- Để huấn luyện YOLO cần ảnh biển số **đã gán nhãn bounding box**, nhãn YOLO `.txt`, tách tập `train` và `val`, cùng file YAML dataset. Ảnh/video chưa gán nhãn không thể dùng để train detector.
- `yolov8n.pt` có thể làm checkpoint khởi tạo khi fine-tune; bản pretrained COCO chưa phải detector biển số. Trang huấn luyện chỉ thay model đang dùng sau khi lượt train tạo được `best.pt`; có thể tải model về để giữ bản sao.
- Dataset ZIP cần chứa đúng một `data.yaml`, `images/train`, `images/val`, `labels/train` và `labels/val`. App giới hạn ZIP tải lên 512 MB, tổng dung lượng giải nén 2 GB, kiểm tra đường dẫn an toàn và tự chỉnh `path` trong YAML.
- Dataset tham khảo: [Vietnamese license plates trên Kaggle](https://www.kaggle.com/datasets/tanhphp/vietnamese-license-plates). Kiểm tra điều khoản/giấy phép và xem nội dung tải xuống; nếu không có nhãn YOLO, cần gán nhãn trước (ví dụ bằng CVAT/Roboflow) rồi export YOLO format.
- Dataset Kaggle này đã được tải vào `datasets\vietnamese-license-plates`; metadata Kaggle ghi giấy phép Apache 2.0. Archive được xác minh có `train/val/test`; bản tải hiện tại có 2.036 ảnh train, 146 ảnh validation và 73 ảnh test. Ghi chú nguồn/kiểm tra nằm ở `datasets\vietnamese-license-plates\DATASET_SOURCE.md`.
- Đã chạy một lượt pilot CPU để xác nhận pipeline: YOLOv8n, 3 epoch, 320 px, batch 4, fraction 5%. Checkpoint thử nghiệm ở `models\plate_detector_pilot.pt`, đồng thời hiện được cấu hình làm detector tại `models\plate_detector_yolov8n.pt`; kết quả test ở `runs\train\vietnamese_plates_cpu_pilot_3ep\test_metrics.json`. Đây chưa phải huấn luyện toàn bộ; xem phần ghi chú nguồn dữ liệu để biết các chỉ số và giới hạn.
- Các bước tải bằng Kaggle CLI trên Windows:

  1. Tạo tài khoản Kaggle, mở trang dataset và chấp thuận điều khoản nếu được yêu cầu. Trong Kaggle Account, tạo API token; đặt `kaggle.json` tại `%USERPROFILE%\.kaggle\kaggle.json`. Không chia sẻ token này.
  2. Mở CMD tại thư mục dự án và chạy:

     ```cmd
     .venv\Scripts\activate
     python -m pip install kaggle
     mkdir datasets\raw
     kaggle datasets download -d tanhphp/vietnamese-license-plates -p datasets\raw --unzip
     ```

  3. Kiểm tra file tải về. Nếu không có cặp ảnh/nhãn bounding box thì mở ảnh trong CVAT hoặc Roboflow, gán một class `license_plate` cho mỗi biển số, chia train/validation theo ảnh riêng biệt rồi export định dạng Ultralytics YOLO. Không lấy ảnh OCR/crop biển số làm ảnh detection nếu mục tiêu là phát hiện khung biển số trên toàn ảnh xe.
- Cấu trúc kỳ vọng:

  ```text
  datasets/plates/
  ├── images/train/
  ├── images/val/
  ├── labels/train/
  ├── labels/val/
  └── dataset.yaml
  ```

- Ví dụ `dataset.yaml` trên Windows (đổi đường dẫn theo vị trí dataset):

  ```yaml
  path: C:/Users/Admin/Desktop/license_plate_system/datasets/plates
  train: images/train
  val: images/val
  names:
    0: license_plate
  ```

- Có thể huấn luyện CPU từ CMD bằng helper mới, tại thư mục dự án:

  ```cmd
  .venv\Scripts\activate
  python train_plate_detector.py --data datasets\plates\dataset.yaml --epochs 5 --imgsz 416 --batch 4 --model yolov8n.pt
  ```

  Điều chỉnh dataset, số epoch và kích thước ảnh trực tiếp trong lệnh. Huấn luyện toàn bộ dữ liệu trên CPU có thể mất nhiều thời gian.

- Helper triển khai tự động chép best checkpoint vào `models\plate_detector_yolov8n.pt`. File gốc nằm dưới `runs\plate_detect\train\weights\best.pt`:

  ```cmd
  copy runs\plate_detect\train\weights\best.pt models\plate_detector_yolov8n.pt
  ```

- Đánh giá trên tập validation:

  ```cmd
  yolo detect val model=models\plate_detector_yolov8n.pt data=datasets\plates\dataset.yaml device=cpu
  ```

- Kết quả mAP chỉ được tính khi chạy `model.val` trên tập validation thật. Chỉ số OCR không thay thế mAP phát hiện.
- Giữ tập test tách biệt để đánh giá cuối cùng; không dùng test làm train hoặc validation.

### 7.7. Sao lưu

- Trong app, mở **Cấu hình > Sao lưu và khôi phục**, chọn **Tạo bản sao lưu**, rồi tải file ZIP về nơi lưu trữ an toàn.
- File ZIP bao gồm database, ảnh bằng chứng và file đã tải lên. Để khôi phục, chọn file ZIP, xác nhận thao tác thay thế dữ liệu hiện tại rồi bấm **Khôi phục bản sao lưu**.
- App mặc định lưu tại thư mục dự án: `license_plate.db` và `evidence/`. Khi chạy trên hosting có ổ đĩa tạm thời, hãy tải backup ra ngoài app thường xuyên; dữ liệu trên hosting có thể mất khi khởi động lại hoặc triển khai lại.

## 8. Dữ liệu ảnh/video mẫu

Thư mục `sample_data` được giữ trống để tránh tạo dữ liệu giả. Bạn có thể đặt các file thực tế ở đây để kiểm thử:

```text
sample_data/
  - license_plate_01.jpg
  - license_plate_02.jpg
  - traffic_01.mp4
```

## 9. Vị trí dữ liệu

Khi chạy trên máy tính, ứng dụng lưu dữ liệu trong thư mục dự án:

```text
license_plate.db   # Hồ sơ xe, lịch sử ra/vào, cài đặt và nhật ký
evidence/          # Ảnh bằng chứng và ảnh mẫu OCR
uploads/           # Ảnh/video tải lên để quét
exports/           # Báo cáo và các file backup
```

Các file database và ảnh nằm trên máy chủ chạy Streamlit. Với bản chạy cục bộ, chúng nằm trên máy tính đó; với dịch vụ hosting dùng ổ đĩa tạm thời, dữ liệu có thể mất khi ứng dụng khởi động lại hoặc triển khai lại. Hãy tạo ZIP backup trong mục **Cấu hình** và tải xuống một nơi lưu trữ bền vững.

## 10. Các lỗi thường gặp và cách sửa

### Lỗi: `ModuleNotFoundError: No module named 'streamlit'`

```cmd
pip install -r requirements.txt
```

### Lỗi: `OpenCV` không đọc được video

- Kiểm tra codec video hoặc đổi định dạng file sang MP4/H.264.
- Dùng camera index hợp lệ, ví dụ `0` cho webcam tích hợp.

### Lỗi: OCR không nhận diện chữ

- Ảnh biển số phải rõ, sáng đủ, không bị méo.
- Cắt vùng biển số rõ hơn.
- Người dùng cần xác nhận lại biển số thủ công nếu OCR không chắc chắn.

### Lỗi: `EasyOCR` tải model chậm

- Bước đầu tiên sẽ tải model từ internet, cần kết nối mạng tốt.
- Nếu tải thất bại, kiểm tra firewall / proxy và thử lại.

### Lỗi: `Could not find model` hoặc `No module named easyocr`

```cmd
pip install easyocr
```

### Camera máy tính không mở

- Camera được mở qua OpenCV trên chính máy đang chạy Streamlit; không phụ thuộc quyền camera của trình duyệt.
- Chọn đúng chỉ số camera (`0` thường là webcam tích hợp), đóng ứng dụng khác đang dùng camera và kiểm tra quyền camera của Windows.

### Điện thoại không mở được camera

- Mở trang bằng HTTPS khi truy cập từ máy chủ deploy; cấp quyền camera cho trình duyệt.
- Khi chạy trong Wi-Fi nội bộ bằng HTTP, một số trình duyệt/thiết bị có thể không cho truy cập camera. Dùng URL HTTPS.
- Camera trực tiếp dùng camera của thiết bị đang mở trang và gửi ảnh qua HTTPS; camera không phải thiết bị gắn với máy chủ. Ảnh truyền khoảng mỗi 1,1 giây thay vì video FPS cao để giới hạn băng thông và tải xử lý.

## 11. Ghi chú về khả năng triển khai thực tế

- Đây là phiên bản MVP chạy được trên CPU với dữ liệu thực.
- Nếu cần hiệu suất cao hơn và nhận diện chính xác hơn với biển số xe Việt Nam, nên bổ sung mô hình plate detector chuyên dụng và kiểm thử trên tập dữ liệu thực tế.
- Chức năng quản lý phương tiện, lịch sử ra vào và báo cáo đã được xây dựng đúng như yêu cầu.
