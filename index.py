import torch
from transformers import CsmForConditionalGeneration, AutoProcessor

model_id = "sesame/csm-1b"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and processor
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

# Prepare inputs and generate
text = "[0]Hello from Sesame.[1]I feel good"  # `[0]` specifies speaker ID 0
inputs = processor(text, add_special_tokens=True).to(device)
audio = model.generate(**inputs, output_audio=True)

# Save the audio
processor.save_audio(audio, "basic_example.wav")
