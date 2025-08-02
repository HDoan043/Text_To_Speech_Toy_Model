# **XỬ LÝ DỮ LIỆU**
<br><br>

## **I. Tổng quan**

### **I.1. Dataset**

Sử dụng bộ Dataset **VIVOS CORPUS**. Folder dataset tên là `vivos`, có cấu trúc như sau:
```bash
\vivos
    |_ \test
    |   |_ \waves
    |   |   |_ \VIVOSDEV01
    |   |   |   |_VIVOSDEV01_R001.wav
    |   |   |   |_VIVOSDEV01_R002.wav
    |   |   |   |_ ...
    |   |   |_ \VIVOSDEV02
    |   |   |   |_VIVOSDEV02_R001.wav
    |   |   |   |_VIVOSDEV02_R002.wav
    |   |   |   |_ ...
    |   |   |_ \VIVOSDEV03
    |   |   |   |_ ...
    |   |   |_ ...
    |   |_ genders.txt
    |   |_ prompts.txt
    |_ \train
    |   |_ \waves
    |   |   |_ \VIVOSSPK01
    |   |   |   |_VIVOSSPK01_R001.wav
    |   |   |   |_VIVOSSPK01_R002.wav
    |   |   |   |_ ...
    |   |   |_ \VIVOSSPK02
    |   |   |   |_VIVOSSPK02_R001.wav
    |   |   |   |_VIVOSSPK02_R002.wav
    |   |   |   |_ ...
    |   |   |_ \VIVOSSPK03
    |   |   |   |_ ...
    |   |   |_ ...
    |   |_ genders.txt
    |   |_ prompts.txt  
    |_ COPYING
    |_ README    
```

**Giải thích:** Bộ dataset gồm hơn 15 giờ dữ liệu được thu thập từ các ứng viên, bao gồm cả nam cả nữ. Dataset chia rõ tập `train` và `test`: 
* Tập `train` nằm trong folder `train`
* Tập `test` nằm trong folder `test`
* Folder `train` và `test` có cấu trúc giống nhau, đều gồm có:
    * `genders.txt` : file text chỉ rõ giới tính của từng ứng viên với mã số tương ứng
    * `prompts.txt` : file text chứa transcript của từng ứng viên với mã số tương ứng
    * folder `waves`: gồm các folder con, mỗi folder con ứng với 1 ứng viên, bên trong là các file `.wav` đọc từng câu.

### **I.2. Xử lý dữ liệu**
Sử dụng tập `train` để huấn luyện, tập `test` để kiểm tra. 

Nhiệm vụ xử lý dữ liệu:
* Xử lý transcript: Có vô số từ trong tiếng Việt, để học được cách phát âm của từng từ thì rất phức tạp. Tuy nhiên, số phoneme( âm vị) trong tiếng Việt lại là hữu hạn, do đó có thể cho mô hình học cách phát âm các âm vị và ghép lại thành phát âm từ, điều này dễ dàng hơn, khả thi hơn, tương tự như cách con người học ghép vần. Do đó, nhiệm vụ đầu tiên là chuyển các đoạn transcript về các phoneme tương ứng.

* Xử lý wave: Khi huấn luyện, cần phải chỉ rõ cho mô hình biết cách đọc 1 câu như thế nào.

## **II. Xử lý transcripts**
### **II.1. Nhiệm vụ**
Ánh xạ 1 bản transcript ngôn ngữ tự nhiên thành 1 bản chỉ gồm các phoneme. 

Tuy nhiên, các phoneme chỉ là các kí tự, để đưa vào mô hình thì cần các số. Do đó ta sẽ thống kê từ điển các phoneme đã xuất hiện, và thay thế bản phoneme dạng chuỗi kí tự thành dạng số. Các số này là chỉ số của phoneme trong từ điển.

Đầu ra của bước này là 1 file `pickle` chứa các số như trên.

### **II.2. Công cụ**
Sử dụng thư viện `phonemizer` để chuyển 1 từ thành phoneme tương ứng

**Cài đặt**
```bash
pip install phonemizer
```

**Import**
```python
# Import hàm chuyển đổi phonemize từ gói phonemizer

# Hàm chuyển đổi phonemize sẽ chuyển 1 câu thành dãy chuỗi các phoneme
from phonemizer import phonemize

# Để định nghĩa cách mà các phoneme phân cách với nhau trong chuỗi phoneme trả về từ hàm phonemizer, các từ phân cách với nhau thì cần đối tượng Separator

# Import lớp Separator từ gói separator
from phonemizer.separator import Separator
```

**Sử dụng**
```python
# Định nghĩa đối tượng Separator
separator = Separator(
    phone = ",", # giữa 2 phoneme cách nhau bởi kí tự ","
    word = "|"   # giữa 2 từ cách nhau bởi kí tự "|"
)

# Chuyển đổi câu thành phoneme -> trả về chuỗi
string_result = phonemizer(
    string_sentence,       # câu( string) cần chuyển đổi
    language,              # để "vi" nếu là tiếng việt, 
                           # "en" nếu là tiếng anh
    backend = "espeak",
    with_stress,           # để True nếu có đánh trọng âm từ, 
                           # False nếu không
    separator = separator  # đối tượng Separator vừa khởi tạo
)
```

### **II.3. Pipeline**
Duyệt file `prompts.txt` trong folder `train`:
* Với speaker( cùng 1 Id): Lấy tất cả transcripts của người đó, chuyển về phoneme và tổng hợp thành 1 file, tên là `Id.txt` trong folder `Phoneme` trong folder `Data`
* Ở đây chỉ huấn luyện mạng trên dữ liệu của 1 speaker( thử nghiệm trên tập dữ liệu nhỏ 1 người nói)

## **III. Xử lý wave**
### **III.1. Mục tiêu**
Ánh xạ các đoạn `.wav` thành danh sách các số, mỗi số đại diện cho 1 đặc trưng năng lượng cho 1 giai đoạn trong đoạn `wav`. 

Danh sách các số được biểu diễn bởi cấu trúc dữ liệu `np_array` và được lưu trong file `.npy` để cải thiện tốc độ truy xuất.

Mỗi file `.npy` tương ứng với 1 file `.wav`, mỗi file `.wav` ứng với 1 câu script trong transcript.

### **III.2. Pipeline**
* Loại bỏ khoảng lặng đầu và cuối đoạn sóng âm( Khi mô hình nhận đầu vào là các phoneme, nó sẽ ngay lập tức dự đoán phát âm của các phoneme, nếu đưa cả khoảng lặng vào thì sẽ gây nhiễu vì mô hình sẽ phải học cách dự đoán khoảng lặng trước khi dự đoán phoneme thực sự, mà khoảng lặng có thể dài ngắn khác nhau với mỗi câu).

* Chuẩn hóa đoạn sóng âm về tần số `22050 Hz`, kênh `mono`, độ rộng `16-bit`.

* Chuyển đổi đoạn sóng âm thành **Mel spectrogram** và lưu vào file `.npy`.

### **III.3. Công cụ**

Sử dụng backend `sox` để cắt khoảng lặng đầu và cuối.

Sử dụng thư viện `torchaudio` để chuẩn hóa đoạn âm thanh `.wav` và chuyển thành **Mel spectrogram**.

#### **III.3.1. `sox`**
**Cài đặt**
Cài đặt backend `sox`
* Download `sox` từ trang https://sourceforge.net/projects/sox/files/sox/
* Chạy file `.exe` vừa tải về để cài đặt -> 1 folder `sox -...` được tải về
* Thêm đường dẫn của `sox -...` vừa tải về vào biến môi trường:
    * Mở **System Properties**, mở **Environment Variables**
    * Tìm biến tên *Path* -> **Edit**
    * Chọn **New** và thêm đường dẫn tới folder `sox-...` vào danh sách biến môi trường
    * OK -> OK -> OK
* Kiểm tra cài đặt thành công:
    * Mở terminal mới
    * Chạy lệnh:
     ```bash
     sox --version
     ```
     * Nếu ra `Sox v...` thì cài thành công
     
`sox` cung cấp lệnh thao tác thông qua **cmd**, nhưng có thể sử dụng `subprocess` để sử dụng `sox` với code( bản chất vẫn là thay người dùng tương tác với **cmd**). `subprocess` có sẵn trong `Python`, không cần phải cài đặt.

**Import**
```python
import subprocess
```

**Sử dụng**

```python
subprocess.run(
    arg # danh sách lệnh + tham số theo đúng thứ tự như nhập lệnh trong cmd, tất cả các phần tử trong danh sách phải là chuỗi string
)
```

Ở đây: lệnh trim khoảng lặng trong file `input.wav`, và trả ra kết quả trong file `output.wav` với ngưỡng `top_db=30` thì:
* Lệnh `cmd` là:
    ```bash
    sox input.wav output.wav silence 1 0.1 30 reverse silence 1 0.1 30d reverse
    ``` 
* Lệnh `python` tương ứng là:
    ```python
    subprocess.run([
        "sox", "input.wav", "output.wav", "silence", "1", "0.1", "30d", "reverse", "silence", "1", "0.1", "30d", "reverse"
    ])

#### **III.3.2. `torchaudio`**

**Cài đặt**
```bash
pip install torchaudio
```

**Import**
```python
import torchaudio
```

**Sử dụng**
```python
### LOAD WAVE ###
'''
    Đoạn sóng âm sẽ được lấy mẫu theo tần số lấy mẫu sample_rate, có nghĩa là: cứ sau mỗi 1/sample_rate s, đoạn âm thanh sẽ được lấy mẫu 1 lần

    Đặc trưng âm thanh tại thời điểm đó sẽ được chuyển thành 1 giá trị số tương ứng. Cứ 1s, đoạn âm thanh sẽ được lấy mẫu sample_rate lần, chuyển thành sample_rate số. Thời lượng âm thanh n s, sẽ được lấy mẫu n*sample_rate lần, chuyển thành n*sample_rate số.

    Khi load audio, hàm torchaudio.load sẽ trả về tensor và sample_rate:
        _ sample_rate là tần số lấy mẫu mà hệ thống sử dụng để lấy mẫu trong đoạn âm thanh
        _ tensor có kích thước [channels, n*sample_rate] là kết quả của quá trình lấy mẫu. tensor là 1 dãy gồm channels( số kênh) phần tử. Mỗi phần tử là kết quả lấy mẫu trên 1 kênh của âm thanh, là 1 dãy gồm n*sample giá trị lấy mẫu. 

    Kết hợp với vosk để tìm thời điểm bắt đầu phát âm của câu và kết thúc của câu: cần truyền vào hàm rec.AcceptWaveForm 1 chuỗi byte, do đó khi có kết quả lấy mẫu dưới dạng tensor, cần chuyển về byte để truyền vào hàm.
'''
waveform, sample_rate = torchaudio.load(
    filepath, # đường dẫn file wave
    normalize # mặc định là False
              # Nếu là True: các giá trị sẽ được chuẩn 
              # hóa về [-1; 1]
)

### RESAMPLE ###
# Khởi tạo đối tượng resample
resampler = torchaudio.transforms.Resample(
    orig_freq, # tần số lấy mẫu gốc
    new_freq, # tần số lấy mẫu mong muốn 
)
# Resample đoạn âm thanh
# Trả về 1 tensor đại diện âm thanh mới gồm các giá trị 
#   được thay đổi dựa trên việc lấy mẫu theo tần số mới
waveform_resample = resampler(waveform)

### CHUYỂN KÊNH SANG MONO ###
if waveform.shape[0] >1:
    waveform = torch.mean(
        waveform, 
        dim = 0,
        keepdim = True 
        )

### TRÍCH XUẤT AUDIO CON TỪ ĐOẠN GỐC ###
# Tính index của mẫu được lấy tại thời điểm start_time
start_sample = int(start_time*sample_rate)
# Tính index của mẫu được lấy tại thời điểm end_time
end_sample = int(end_time*sample_rate)
# Trích xuất audio con từ start_time đến end_time
sub_audio = waveform[:, start_sample: end_sample]

### TÍNH MEL SPECTROGRAM ###
'''
    Để tính mel spectrogram từ đoạn wave:
    _ Thực hiện lấy mẫu đoạn wave
    _ Thực hiện tính mel spectrogram theo các tham số truyền vào
'''
# Khởi tạo đối tượng tính mel spectrogram với các giá trị tham số đề xuất cho bài toán TTS
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=22050, # Đây là tần số lấy mẫu để tính mel
                       # Không phải tần số lấy mẫu của đoạn âm thanh
                       # Vẫn cần phải resample đoạn âm thanh về 22050 Hz
    n_fft=1024,
    hop_length=256,
    n_mels=80,
    f_min=0.0,
    f_max=8000.0,
    power=2.0
)
# Chuyển đổi âm thanh về mel spectrogram
mel_spectrogram = mel_transform(waveform)
```


