HƯỚNG DẪN SỬ DỤNG:

1. sentiment_analysis.py:

- Đây Chương trình viết dưới dạng dòng lệnh, hỗ trợ 2 tham số bắt buộc là
  result_file và câu đầu vào, tên tập tin chạy là sentiment_analysis.py và
- Ví dụ:
"...\MSSV>python sentiment_analysis.py --input "Môn học rất bổ ích" --result sentiment.txt"

2. train_model.py:

- Đặt file này cùng cấp với các training data và file sentiment_analysis
- Nếu có thay đổi training data thì mở mã nguồn và điều chỉnh các '_dir' bên trong
- Vì chưa tìm hiểu cách save/load model nên chương trình không có model sẵn, khi chạy file
  sentiment_analysis.py, mã nguồn sẽ tự động sử chạy file train_model và sử dụng kết quả để
  tính score cho câu input