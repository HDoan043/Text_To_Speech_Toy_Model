import os, torchaudio, torch, pickle

'''
Hàm load file .wav
'''
def load_wave(file_path):
    waveform, sampling_rate = torchaudio.load(file_path, normalize = True)
    return waveform, sampling_rate

'''
Chuẩn hóa đoạn âm thanh về tần số 22050 Hz, số kênh Mono
'''
def norm(waveform, sampling_rate, new_sampling_rate = 22050):
    # Chuẩn hóa số kênh về mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(
            waveform,
            dim = 0,
            keepdim = True
        )
    # Khởi tạo đối tượng resample đoạn wave
    resampler = torchaudio.transforms.Resample(
        sampling_rate,
        new_sampling_rate
    )
    # Resample đoạn âm thanh
    waveform = resampler(waveform)

    # trả về đoạn wave sau khi được chuẩn hóa
    return waveform

'''
Tính mel spectrogram
'''
def mel_spectrogram(waveform):
    mel_transformer = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050,
        n_fft = 1024,
        hop_length = 256,
        n_mels = 80,
        f_min = 0.0,
        f_max = 8000.0,
        power = 2.0
    )
    mel = mel_transformer(waveform)
    mel_spec = torch.log(mel + 1e-6)
    return mel_spec

if __name__ == "__main__":
    # liệt kê các file wave
    root = os.path.join("..","ProcessedData", "Tem")
    waveList = os.listdir(root)

    # kết quả
    mel_list = []

    index = 0
    # tính mel spectrogram của từng file và lưu vào file npy
    for wavefile in waveList:
        print(f"\rProcessing file {index} ...", end="")
        file_path = os.path.join(root, wavefile)
        # đọc file wave
        waveform, sampling_rate = load_wave(file_path)
        
        # Chuẩn hóa âm thanh về 22050 Hz và Mono
        waveform = norm(waveform, sampling_rate)

        # Tính mel spectrogram
        mel = mel_spectrogram(waveform)

        # Chuyển thành numpy và lưu vào mảng kết quả
        mel_list.append(mel)
        index += 1

    print()
    print(f"Finish processed {index} wave files!!!")
    # Sau cùng lưu mảng kết quả vào 1 file pickle
    with open(os.path.join("..", "ProcessedData", "Mel_spectrogram","mel_spectrogram.pkl"), "wb") as f:
        pickle.dump(mel_list, f)


    