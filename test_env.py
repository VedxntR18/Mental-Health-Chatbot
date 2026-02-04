import torch
from transformers import pipeline

# Check if DL library is loaded
print(f"PyTorch Version: {torch.__version__}")

# Test the NLP Brain
bot = pipeline("sentiment-analysis")
print(bot("I am so excited to start my project!"))