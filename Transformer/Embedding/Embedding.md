# **Embedding**
<br><br>


## **I. Tổng quan**

**Mục tiêu**

Với 1 phoneme đơn lẻ trong chuỗi, cần chuyển đổi thành 1 vector embedding số tương ứng.

Chuỗi phoneme của câu cần chuyển thành ma trận gồm các vector embedding.

**Yêu cầu**

Vector embedding của 1 phoneme đơn trong chuỗi phải mã hóa được các thông tin sau:
* Phân biệt được phoneme này với phoneme khác

* Thể hiện được vị trí của phoneme trong chuỗi

## **II. Phương pháp**

### **II.1. Embedding phoneme**

Để phân biệt được phoneme này với phoneme khác, ta có thể đơn giản chỉ cần sử dụng **one-hot encoding** để mã hóa từng phoneme. Tuy nhiên, nếu chỉ mã hóa bằng **one-hot encoding** thì có 2 nhược điểm:
* Tạo thành các **sparse vector**, nghĩa là các vector thưa. Điều này gây ra việc kém hiệu quả trong quá trình huấn luyện vì gồm nhiều phần tử `0`. Để mô hình học hiệu quả thì cần các vector **dense**( tức là các vector có ít thành phần bằng 0)
* Các phoneme có cách phát âm tương tự nhau không thể biểu diễn được gần nhau hơn do hệ vector khi embedding bằng **one-hot** là hệ vector **trực chuẩn** ( bất kì cặp vector nào cũng đều trực giao với nhau)

Do đó, để khắc phục được nhược điểm trên, sử dụng 1 tầng `Embedding` để tự học cách mã hóa phoneme thành 1 vector. Vector này sẽ được học và cập nhật dần dần trong quá trình huấn luyện để giảm hàm loss. 

Gói `torch.nn` của **Pytorch** có hỗ trợ lớp `Embedding` để thực hiện việc này.

**`nn.Embedding`**

Lớp `Embedding` hoạt động như 1 bảng ánh xạ: 
* Lớp `Embedding` có 1 ma trận trọng số có thể cập nhận được `W` có kích thước `num_embedding x embedding_dim`. Với `num_embedding` là số lượng từ hoặc phoneme trong từ điển, `embedding_dim` là số chiều của vector embedding.

* Phoneme thứ `i` trong từ điển sẽ được ánh xạ 1 thành vector: `[W[i][0], W[i][1], ... , W[i][embedding_dim -1]]`. Các vector này có thể học được, mỗi thành phần của vector có thể được cập nhật theo hướng làm giảm hàm **LOSS**, để sau cùng các vector có thể embedding tốt nhất các từ.

Trong từ điển có càng nhiều từ, hoặc càng nhiều phoneme thì kích thước của bảng tra cứu càng lớn, lớp `Embedding` cần phải quản lý càng nhiều.

**Khởi tạo**

```python
import torch.nn as nn

embedding = nn.Embedding(
    num_embeddings, 
    embedding_dim # nên để 256 hoặc 512
)
```

**Forwarding**

Tầng `Embedding` nhận vào 1 **tensor** có kích thước `batch_size x seq_len`:
* `batch_size` là số mẫu trong 1 lô dữ liệu
* Mỗi vector trong lô dữ liệu là 1 tensor có kích thước `seq_len` ( số phoneme trong 1 mẫu dữ liệu), với: thành phần thứ `i` trong vector là chỉ số `index` của phoneme thứ `i` của mẫu trong từ điển. 

    Ví dụ: 

    Mẫu là 1 chuỗi phoneme: `a z c`, từ điển gồm 26 phoneme với chỉ số của phoneme `a` là `0`, chỉ số của phoneme `z` là `25`, của `c` là `2`

    Thì vector được sử dụng là `[0, 25, 2]` 

    1 batch sẽ bao gồm `batch_size` vector như vậy tập hợp lại thành 1 tensor. Tuy nhiên, để hợp lại thành 1 batch thì các vector phải có kích thước bằng nhau, nhưng các mẫu có thể dài ngắn do số lượng các phoneme khác nhau. Do đó, cần phải **padding** các vector để có cùng kích thước. Do đó ta sẽ **padding** sao cho các vector **embedding** sẽ có kích thước là kích thước `max_seq_len` của embedding dài nhất trong batch.

    Sau đó sử dụng 1 lớp mask để lấy ra những phần tử thực sự tương ứng với từng mẫu thay vì lấy cả phần padding vô giá trị.

Do đó, cần phải có 1 danh sách thống kê tất cả các phoneme trong từ điển để biết từng phoneme có chỉ số bao nhiêu trong từ điển.

Lớp `Embedding` sẽ sử dụng chỉ số của phoneme để ánh xạ thành 1 vector thông qua ma trận `W`. Đầu ra là 1 tensor có kích thước `batch_size x max_seq_len x embedding_dim`

### **II.2. Positional Encoding**

Các khối bên trong mô hình không nhận biết thứ tự xuất hiện của các phoneme trong mẫu, dẫn đến tình huống: mẫu gồm phoneme `a t` và mẫu gồm phoneme `t a` sẽ được phát âm như nhau. Vì vậy cần phải có 1 cơ chế giúp mã hóa sự tuần tự của các phoneme.

Ý tưởng là mã hóa vị trí của từng phoneme trong câu. **Pytorch** không hỗ trợ lớp này nên phải tự định nghĩa.

Để có thể kết hợp với vector **Embedding** thì tốt nhất nên có sự thống nhất về kích thước đầu ra giữa vector **Embedding** và vector **Positional Encoding**
* Input: Dùng chung 1 tensor đầu vào với lớp `Embedding` nên cũng có kích thước `batch_size x max_seq_len`

* Output: Để thống nhất kích thước với tensor đầu ra của lớp `Embedding`, nên để kích thước đầu ra của lớp `Positional Encoding` cũng là `batch_size x max_seq_len x embedding_dim`. Có nghĩa là: mỗi phoneme trong 1 mẫu phải được mã hóa `Positional Encoding` thành 1 vector có kích thước `embedding_dim`.

    Công thức như sau: vị trí `pos` thì khi mã hóa thành vector **embedding** thì 
    thành phần thứ `k` trong vector **positional encoding** sẽ được tính khác nhau với `k` chẵn và `k` lẻ:

    * Với `k` chẵn:
    ```python
    embedding[k] = sin(pos / (1000**(k/embedding_dim)))
    ```

    * Với `k` lẻ:
    ```python
    embedding[k] = cos(pos / (1000**(k/embedding_dim)))
    ```
### **II.3. Tổng hợp kết quả mã hóa**

Tensor tổng hợp thông tin phát âm và vị trí của các phoneme để đưa vào `encoder` sẽ được tính như sau:

```python
input = embedding + positional_encoding
# 2 tensor này cộng được với nhau vì có cùng kích thước batch_size x max_seq_len x embedding_dim
```