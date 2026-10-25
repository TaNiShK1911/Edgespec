from __future__ import print_function
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edgespec.model.loader import load_model

print("Starting OPT-1.3B load test...")
filepath = 'data/weights/opt-1.3b-4bit.edgespec'
model = load_model(filepath, device='cuda')
print("Model loaded successfully!")
gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)
print("GPU Memory Allocated: %.2f MiB" % gpu_mem)

# Run a quick dummy forward
x = torch.randint(0, 50272, (1, 8), device='cuda')
model.init_kv_caches(max_seq_len=64, device='cuda')
print("Running forward pass...")
logits = model(x)
print("Forward pass successful! Logits shape:", logits.shape)
gpu_mem_after = torch.cuda.memory_allocated() / (1024 * 1024)
print("GPU Memory After Forward: %.2f MiB" % gpu_mem_after)
print("ALL CHECKS PASSED!")
