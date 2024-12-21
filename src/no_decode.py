from io import BytesIO
from urllib.request import urlopen

import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B", output_hidden_states=True, return_dict_in_generate=True
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)

prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Describe the music:"
# url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/glass-breaking-151256.mp3"
url = "https://github.com/drichert/touchy/raw/refs/heads/master/media/Sundown.mp3"
audio, sr = librosa.load(
    BytesIO(urlopen(url).read()), sr=processor.feature_extractor.sampling_rate
)
inputs = processor(text=prompt, audios=audio, return_tensors="pt", sampling_rate=sr)

generated = model.generate(**inputs, max_length=256)
# Slice out the audio tokens (comes after the prompt)
generated_ids = generated.sequences[:, inputs.input_ids.size(1) :]
response = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
# >>> print(response)
# A calm, ambient, cinematic string piece. Perfect as a background to create a peaceful
# and relaxing atmosphere.
