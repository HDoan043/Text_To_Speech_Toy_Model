import subprocess
import os

'''
Hàm cắt khoảng lặng đầu và cuối
'''
def trim(
        input_wavefile, 
        output_wavefile, 
        top_db = -50):
    subprocess.run([
        "sox", input_wavefile, output_wavefile, 
        "silence", "1", "0.1", f"{top_db}d", "reverse", 
        "silence", "1", "0.1", f"{top_db}d", "reverse"
    ])

if __name__ == "__main__":
    # Liệt kê các file wave cần đọc
    root = os.path.join("..","vivos", "train", "waves", "VIVOSSPK01")
    wavefile_list = os.listdir(root)

    # Nơi lưu file sau khi trim
    des = os.path.join("..", "ProcessedData", "Tem")

    # cắt khoảng lặng và lưu
    num = 0
    for wavefile in wavefile_list:

        print(f"\rSuccessfully process {num} wave files", end= "")
        input_file = os.path.join(root, wavefile)
        output_file = os.path.join(des, wavefile)

        trim(input_file, output_file)
        num += 1

    print(f"\rSuccessfully process {num} wave files")

