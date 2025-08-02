from phonemizer import phonemize
from phonemizer.separator import Separator
import torch
import pickle
import os


data = os.path.join("..", "vivos", "train", "prompts.txt")

phoneme_file = os.path.join("..", "ProcessedData", "Tem", "phoneme.txt")

# Chỉ lấy dữ liệu của 1 người để huấn luyện thử nghiệm
# Đọc file
with open(data, "r", encoding="utf-8") as f:
    separator = Separator(
        phone = ",",
        word = "|"
    )
    num = 1
    while True:
        # Đọc từng dòng
        line = f.readline()

        # Nếu có Id của người đầu tiên thì phonemize câu nói đó
        if "VIVOSSPK01" in line:
            # Lấy nội dung câu nói từ dòng bằng cách bỏ từ đầu tiên( từ Id của speaker)
            split_line = line.split()
            sequence = " ".join(split_line[1:])
            
            # Chuyển câu thành phoneme và ghi vào file đích
            phonemes_string = phonemize(
                sequence,
                language = "vi",
                backend = "espeak",
                with_stress = False,
                separator = separator
            )

            # tách các từ bởi dấu "|"
            phonemes_list_word = phonemes_string.split("|")

            new_phonemes_list_word = []
            # mỗi từ, tách các phoneme bằng dấu ","
            for phone_word in phonemes_list_word:
                phones_in_word = phone_word.split(",") # tách từng phoneme, các phoneme cách nhau bởi dấu ","
                phones_not_empty = [each for each in phones_in_word if each != ""] # với mỗi phoneme, chọn nó nếu nó khác rỗng
                new_phonemes_list_word.append(" ".join(phones_not_empty))

            final_phonemes_string = " | ".join(new_phonemes_list_word)[:-3]
            with open(phoneme_file, "a", encoding="utf-8") as g:
                g.writelines(final_phonemes_string + "\n")
            print(f"\rProcessed {num} sequences ...", end="")
            num += 1
        else: 
            break

# Đọc file phoneme dạng chuỗi vừa tạo ra và thống kê các phoneme thành 1 từ điển
phoneme_dict = {}
index = 0
print()
print("Listing phonemes ...")
with open(phoneme_file, "r", encoding="utf-8") as f:
    while True:
        line = f.readline()
        if line:
            phones = line.split()
            for phone in phones:
                if phone not in phoneme_dict:
                    phoneme_dict[phone] = index
                    index +=1

        else: break

# Lại mở file phoneme chuỗi và ánh xạ thành các số dựa trên từ điển phoneme thống kê được
seq_list = []
mask = []
max_seq_len = 0
print(f"Mapping phonemes ...")
with open(phoneme_file, "r", encoding="utf-8") as f:
    while True:
        line = f.readline()
        if line:
            phones = line.split()
            max_seq_len = max(max_seq_len, len(phones))
            index_phone = []
            for phone in phones:
                index_phone.append(phoneme_dict[phone])
            
            seq_list.append(index_phone)

        else: break

# padding
print(f"Padding ...")
for i in range(len(seq_list)):
    mask_seq = [1 for _ in range(len(seq_list[i]))]
    if len(seq_list[i]) < max_seq_len:
        padding = [0 for _ in range(max_seq_len - len(seq_list[i]))]
        seq_list[i].extend(padding)
        mask_seq.extend(padding)
    
    mask.append(mask_seq)

seq_list = torch.tensor(seq_list)
mask = torch.tensor(mask)

print(f"Saving ...")
# lưu file
with open(os.path.join("..", "ProcessedData", "Phoneme", "phoneme.pkl"), "wb") as f:
    pickle.dump(seq_list, f)
with open(os.path.join("..", "ProcessedData", "Phoneme", "mask_phoneme.pkl"), "wb") as g:
    pickle.dump(mask, g)

print("Finish !!!")