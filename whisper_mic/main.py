from whisper_mic import WhisperMic
import torch
model_size = "large-v3"

mic = WhisperMic(model=model_size)
result = mic.listen_loop()
# print(result)

# print("cuda" if torch.cuda.is_available() else "cpu")