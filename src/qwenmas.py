import time
from io import BytesIO
from pathlib import Path
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
model = Qwen2AudioEncoder.from_pretrained("Qwen/Qwen2-Audio-7B").cuda()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)

prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>"
# url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/glass-breaking-151256.mp3"
url = "https://github.com/drichert/touchy/raw/refs/heads/master/media/Sundown.mp3"
data_path = Path(__file__).parents[1] / "data"
embs_path = data_path / "embs"
embs_path.mkdir(exist_ok=True)

for file in data_path.iterdir():
    filename = file.stem
    enc_emb = embs_path / f"{filename}.pt"
    if file.suffix == ".mp3" and not enc_emb.exists():
        buf = BytesIO(file.read_bytes())
    else:
        continue

    with Timer("Load") as lt:
        audio, sr = librosa.load(buf, sr=processor.feature_extractor.sampling_rate)

    inputs = processor(text=prompt, audios=audio, sampling_rate=sr, return_tensors="pt")
    encoded_features = inputs.input_features

    with torch.no_grad(), Timer("Encode") as et:
        encoder_outputs = model(encoded_features)

    final_encoded_features = encoder_outputs.last_hidden_state
    if final_encoded_features.isnan().all():
        raise ValueError(f"Error: NaN in the result for {filename}")
    else:
        torch.save(final_encoded_features.cpu(), enc_emb)

    print(f"Processed {filename}")
    print(f"Encoded features shape: {final_encoded_features.shape}")
