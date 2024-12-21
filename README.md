# Qwen Audio Playground

## Demo

`src/demo.py` shows the regular usage: it works as advertised!

```
>>> print(response)
```

> - A calm, ambient, cinematic string piece. Perfect as a background to create a peaceful
> - and relaxing atmosphere.

## Audio encoding

`src/audio_enc.py` gives a script to extract the audio encoding (the 1.2B audio encoder model, without the 7B encoder-decoder model)

- Refer to [the source](https://github.com/huggingface/transformers/blob/504c4d36929b6bb8a8c2ecfad0f2625f4075f22a/src/transformers/models/qwen2_audio/#L851)
  for what this involves code-wise and [page 3, section 2, Methodology](https://arxiv.org/pdf/2407.10759), for the technical explanation:

  > each frame of the encoder output approximately corresponds to a 40ms segment of the original audio signal

- Note: encoding is faster on GPU (it involves Conv1D and Transformer layers)

There is an audio feature extractor (in the sense of audio features, not learnt features) which
involves resampling to 16 kHz and preparing a mel-spectrogram from the raw waveform

> - using a window of 25ms and a hop size of 10ms

It is also pooled with a stride of 2 (i.e. every other window gets dropped), halving the length of the audio representation.

The embeddings could also be extracted before they get aligned with the text space (if this is not of use)

The dimensionality of the result here is 1280 which comes from the value of `d_model` in the
[config](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_audio/configuration_qwen2_audio.py)

> - num_mel_bins (`int`, *optional*, defaults to 128):    
>      Number of mel features used per input features. Should correspond to the value used in the   
>      `Qwen2AudioProcessor` class.   
> - encoder_layers (`int`, *optional*, defaults to 32):   
>      Number of encoder layers.    
> - encoder_attention_heads (`int`, *optional*, defaults to 20):    
>      Number of attention heads for each attention layer in the Transformer encoder.   
> - encoder_ffn_dim (`int`, *optional*, defaults to 5120):   
>      Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.   
> - encoder_layerdrop (`float`, *optional*, defaults to 0.0):  
>      The LayerDrop probability for the encoder. See the [LayerDrop paper](https://arxiv.org/abs/1909.11556)  
>      for more details.  
> - d_model (`int`, *optional*, defaults to 1280):  
>      Dimensionality of the layers.  
> - dropout (`float`, *optional*, defaults to 0.0):  
>      The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.  
> - attention_dropout (`float`, *optional*, defaults to 0.0):  
>      The dropout ratio for the attention probabilities.  
> - activation_function (`str`, *optional*, defaults to `"gelu"`):  
>      The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,  
>      `"relu"`, `"silu"` and `"gelu_new"` are supported.  
> - activation_dropout (`float`, *optional*, defaults to 0.0):  
>      The dropout ratio for activations inside the fully connected layer.  
> - scale_embedding (`bool`, *optional*, defaults to `False`):  
>      Scale embeddings by diving by sqrt(d_model).  
> - init_std (`float`, *optional*, defaults to 0.02):  
>      The standard deviation of the truncated_normal_initializer for initializing all weight matrices.  
> - max_source_positions (`int`, *optional*, defaults to 1500):  
>      The maximum sequence length of log-mel filter-bank features that this model might ever be used with.  

> - Download took 0.95 seconds
> - Load took 1.26 seconds
> - Encode took 5.11 seconds
> - Encoded features shape: torch.Size([1, 750, 1280])

## Post-audio encoding Qwen LM encoding

The file `src/no_decode.py` interferes with the demo by accessing hidden states, to investigate what
you'd get with just the LLM encoding.

The audio gets combined with the text prompt and passed into the LLM, and the shape changes:

```py
>>> inputs["input_ids"].shape
torch.Size([1, 7])
>>> inputs["input_features"].shape
torch.Size([1, 128, 3000])
```

- **7 tokens** in the text prompt `"<|audio_bos|><|AUDIO|><|audio_eos|>Describe the music:"` for a batch size of 1.
- **128 features** (e.g., frequency bins or channels) in the encoded audio.
- **3000 frames** in the input, corresponding to 40ms chunks sampled at 16kHz with gaps as described.

```py
>>> len(generated.hidden_states[0])
33
>>> len(generated.hidden_states[-1])
33  # 33 hidden states for one batch
```

- Each layer contains **33 sets of hidden states** (batch size 1 in this example).
- **First layer**:  
  - **Each hidden state** contains **756 vectors**, each in **4096-dimensional space**.
- **Final layer**:
  - **Each hidden state** contains a **single vector (1 x 4096)** summarising the sequence.


```python
>>> generated.hidden_states[0][0].shape
torch.Size([1, 756, 4096])  # First layer
>>> generated.hidden_states[-1][0].shape
torch.Size([1, 1, 4096])  # Final layer
>>> len(generated.hidden_states[-1][0])
1  # Single vector in the final layer
```

- **First Layer**:
  - Shape: `[1, 756, 4096]` → 1 batch, 756 tokens (7 text + 749 audio), 4096-dimensional vectors.
- **Final Layer**:
  - Shape: `[1, 1, 4096]` → 1 batch, 1 summary vector, 4096 dimensions.

The audio and text prompt are combined, processed, and condensed into a single deep representation in the final hidden states.
