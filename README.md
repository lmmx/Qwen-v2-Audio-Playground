# Qwen Audio Playground

## Demo

`src/demo.py` shows the regular usage: it works as advertised!

```
>>> print(response)
```

> A calm, ambient, cinematic string piece. Perfect as a background to create a peaceful
> and relaxing atmosphere.

## Audio encoding

`src/audio_enc.py` gives a script to extract the audio encoding (the 1.2B audio encoder model, without the 7B encoder-decoder model)

- Refer to [the source](https://github.com/huggingface/transformers/blob/504c4d36929b6bb8a8c2ecfad0f2625f4075f22a/src/transformers/models/qwen2_audio/#L851)
  for what this involves code-wise and [page 3, section 2, Methodology](https://arxiv.org/pdf/2407.10759), for the technical explanation:

  > each frame of the encoder output approximately corresponds to a 40ms segment of the original audio signal

There is an audio feature extractor (in the sense of audio features, not learnt features) which
involves resampling to 16 kHz and preparing a mel-spectrogram from the raw waveform

> using a window of 25ms and a hop size of 10ms

It is also pooled with a stride of 2 (i.e. every other window gets dropped), halving the length of the audio representation.

