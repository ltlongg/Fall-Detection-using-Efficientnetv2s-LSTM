# Fall Detection System using EfficientNetV2-S & LSTM

Dự án này là một hệ thống phát hiện hành động té ngã (Fall Detection) sử dụng kiến trúc mạng lai ghép (Hybrid Network) kết hợp giữa **EfficientNetV2-S** (CNN) để trích xuất đặc trưng hình ảnh và **Bi-LSTM** (RNN) để xử lý thông tin chuỗi thời gian. Hệ thống được tích hợp vào một giao diện web trực quan sử dụng **Streamlit**, hỗ trợ cả phân tích video có sẵn và giám sát thời gian thực (Real-time).

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow)

## 📋 Tính năng chính

1.  **Dự đoán qua Video (Video Analysis):**
    * Cho phép tải lên video (mp4, avi, mov, mkv).
    * Phân tích video dựa trên chuỗi frame để xác định có hành động ngã hay không.
    * Hiển thị độ tin cậy (Confidence score).
    * Lưu lịch sử các lần phân tích.

2.  **Giám sát Thời gian thực (Real-time Monitoring):**
    * Sử dụng Webcam để giám sát trực tiếp.
    * Tích hợp **YOLOv8** để phát hiện người (Person Detection) giúp giảm nhiễu nền.
    * Cảnh báo ngay lập tức khi phát hiện ngã ("Nguy hiểm" vs "An toàn").
    * **Tự động ghi hình:** Hệ thống tự động cắt và lưu lại đoạn video khi phát hiện té ngã để xem lại sau.
    * Nhật ký (Log) hiển thị các sự kiện theo thời gian thực.

## 🧠 Kiến trúc Mô hình (Model Architecture)

Hệ thống sử dụng mô hình **EfficientNetLSTM**:

* **Backbone (CNN):** `EfficientNetV2-S` (Pre-trained trên ImageNet) dùng để trích xuất đặc trưng (features) từ từng khung hình (frame). Các lớp của EfficientNet sẽ được đóng băng (freeze) khi huấn luyện huấn luyện.
* **Temporal Processing (RNN):** `Bidirectional LSTM` (2 lớp, hidden size 256) tiếp nhận chuỗi đặc trưng từ CNN để học mối quan hệ thời gian giữa các frame liên tiếp.
* **Classifier:** Các lớp Fully Connected (Linear) kết hợp Dropout để đưa ra xác suất té ngã (Binary Classification).
* **Input:** Chuỗi 32 frames, kích thước ảnh resize về 288x288.

## 🛠️ Cài đặt

### 1. Yêu cầu hệ thống
* Python 3.8 trở lên.
* GPU (Khuyến nghị để đạt FPS tốt khi chạy Real-time), nhưng có thể chạy trên CPU.

### 2. Cài đặt thư viện
Tạo môi trường ảo (khuyến khích) và cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
