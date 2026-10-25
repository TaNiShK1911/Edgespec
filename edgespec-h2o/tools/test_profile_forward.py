from __future__ import print_function
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgespec.model.loader import load_model

def p(msg):
    print(msg)
    sys.stdout.flush()

p("[1] Loading model...")
t0 = time.time()
model = load_model('data/weights/opt-1.3b-4bit.edgespec', device='cuda')
p("[1] Model loaded in %.2fs. GPU Allocated: %.1f MiB" % (
    time.time() - t0, torch.cuda.memory_allocated() / (1024 * 1024)))

x = torch.randint(0, 50272, (1, 8), device='cuda')
model.init_kv_caches(max_seq_len=64, device='cuda')
positions = torch.arange(2, 10, device='cuda').unsqueeze(0)

with torch.no_grad():
    p("[2] Running full forward pass across all 24 layers...")
    h = model.embed_tokens(x) + model.embed_positions(positions)
    h = h.to(model.dtype)

    t1 = time.time()
    for l_idx, layer in enumerate(model.layers):
        h, _ = layer(h, model.kv_caches[l_idx], positions)
        if (l_idx + 1) % 6 == 0:
            p("    Layer %d/24 done (elapsed: %.1fs, GPU alloc: %.1f MiB)" % (
                l_idx + 1, time.time() - t1,
                torch.cuda.memory_allocated() / (1024 * 1024)))
    
    # Final layer norm and LM head
    if model.final_layer_norm is not None:
        h = model.final_layer_norm(h)
    logits = torch.nn.functional.linear(h.to(model.lm_head_weight.dtype), model.lm_head_weight)
    p("[3] FULL 24-LAYER FORWARD PASS SUCCESS in %.2fs!" % (time.time() - t1))
    p("    Final GPU Memory: %.1f MiB (Reserved: %.1f MiB)" % (
        torch.cuda.memory_allocated() / (1024 * 1024),
        torch.cuda.memory_reserved() / (1024 * 1024)))
    p("    Logits shape: %s" % str(logits.shape))

p("[4] SUCCESSFUL LAYER TESTS!")
