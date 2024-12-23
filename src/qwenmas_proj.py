import time
from io import BytesIO
from pathlib import Path

import librosa
import torch
from transformers import (
    AutoProcessor,
    Qwen2AudioEncoder,
    Qwen2AudioForConditionalGeneration,
)


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


# Load the audio encoder and projector
full_model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B")
audio_encoder = full_model.audio_tower.cuda()
# audio_encoder = Qwen2AudioEncoder.from_pretrained("Qwen/Qwen2-Audio-7B").cuda()
projector = full_model.multi_modal_projector.cuda()
# projector = Qwen2AudioMultiModalProjector.from_pretrained("Qwen/Qwen2-Audio-7B").cuda()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)

prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>"
data_path = Path(__file__).parents[1] / "data"
sep_path = data_path / "split"
embs_dir = data_path / "embs"
embs_dir.mkdir(exist_ok=True)


def encode_and_project_file(file, split: bool = False):
    filename = file.stem
    if split:
        filename = f"{file.parent.stem}_separated_{filename}"
    emb_path = embs_dir / f"{filename}.pt"
    if file.suffix == ".mp3":
        if not emb_path.exists():
            buf = BytesIO(file.read_bytes())
        else:
            return
    else:
        return

    with Timer("Load") as lt:
        audio, sr = librosa.load(buf, sr=processor.feature_extractor.sampling_rate)

    inputs = processor(text=prompt, audios=audio, sampling_rate=sr, return_tensors="pt")
    encoded_features = inputs.input_features

    with torch.no_grad(), Timer("Encode and Project") as et:
        # Step 1: Encode the audio features
        encoder_outputs = audio_encoder(encoded_features.cuda())
        final_encoded_features = encoder_outputs.last_hidden_state

        # Step 2: Apply the projector
        projector_outputs = projector(final_encoded_features)

    if projector_outputs.isnan().all():
        raise ValueError(f"Error: NaN in the result for {filename}")
    else:
        save_data = {
            "inputs": inputs,
            "encoded_outputs": final_encoded_features.cpu(),
            "projected_outputs": projector_outputs.cpu(),
        }
        torch.save(save_data, emb_path)

    print(f"Processed and projected {filename}")
    return


for file in data_path.iterdir():
    encode_and_project_file(file)

for file in sep_path.glob("*/*.mp3"):
    encode_and_project_file(file, split=True)
