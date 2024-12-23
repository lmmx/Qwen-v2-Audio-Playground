from io import BytesIO
from pathlib import Path

import librosa
from torch import nn
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)

prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Describe the music:"
# mariah = Path(__file__).parents[1] / "data" / "youtube_yXQViqx6GMY_audio.mp3"
mariah = Path(__file__).parents[1] / "data" / "youtube_yXQViqx6GMY_audio.mp3"
audio, sr = librosa.load(
    BytesIO(mariah.read_bytes()), sr=processor.feature_extractor.sampling_rate
)
inputs = processor(text=prompt, audios=audio, return_tensors="pt", sampling_rate=sr)

generated = model.generate(**inputs, max_length=256)
generated_ids = generated[:, inputs.input_ids.size(1) :]
response = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print("Unablated audio encoder:")
print(response)
# # This is a soulful pop track in G major with a tempo of 120 BPM. It features a female vocalist,
# # singing over a progression of chords including D major, E minor, and C major. The song has a
# # Christmas vibe, possibly from its use of bells and chimes.


def debug_audio_tower(module, input, output):
    print("Completed audio tower")
    return output


def debug_projector(module, input, output):
    print("Completed projector")
    return output


def debug_lm(module, input, output):
    print("Completed language model")
    return output


# Add hooks to just the main components
model.audio_tower.register_forward_hook(debug_audio_tower)
model.multi_modal_projector.register_forward_hook(debug_projector)
model.language_model.register_forward_hook(debug_lm)


class SequenceAverager(nn.Module):
    def forward(self, x):
        # x shape: [1, 750, 1280]
        means = x.mean(dim=1, keepdim=True)  # Shape: [1, 1, 1280]
        return means.expand(-1, x.shape[1], -1)  # Shape: [1, 750, 1280]


# Insert it after layer_norm
model.audio_tower.averager = SequenceAverager()
original_layer_norm = model.audio_tower.layer_norm
model.audio_tower.layer_norm = nn.Sequential(
    model.audio_tower.averager, original_layer_norm
)
print("Modified model:\n", model)

print("With audio encoder sequence averaging:")
generated = model.generate(**inputs, max_length=128)
generated_ids = generated[:, inputs.input_ids.size(1) :]
response = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(response)
