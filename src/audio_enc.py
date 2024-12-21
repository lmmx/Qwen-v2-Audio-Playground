import time
from io import BytesIO
from urllib.request import urlopen

import librosa
import torch
from transformers import AutoProcessor, Qwen2AudioEncoder


class Timer:
    def __init__(self, action: str):
        self.action = action

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        print(f"{self.action} took {self.elapsed_time:.2f} seconds")


# Use Qwen2AudioEncoder instead of Qwen2AudioForConditionalGeneration
model = Qwen2AudioEncoder.from_pretrained("Qwen/Qwen2-Audio-7B")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)

prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>"
# url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/glass-breaking-151256.mp3"
url = "https://github.com/drichert/touchy/raw/refs/heads/master/media/Sundown.mp3"

with Timer("Download") as bt:
    buf = BytesIO(urlopen(url).read())

with Timer("Load") as lt:
    audio, sr = librosa.load(buf, sr=processor.feature_extractor.sampling_rate)

inputs = processor(text=prompt, audios=audio, sampling_rate=sr, return_tensors="pt")
encoded_features = inputs.input_features

with torch.no_grad(), Timer("Encode") as et:
    encoder_outputs = model(encoded_features)

final_encoded_features = encoder_outputs.last_hidden_state

print(f"Encoded features shape: {final_encoded_features.shape}")
