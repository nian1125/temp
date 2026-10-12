"""Find which samples were used at generation step 169 in v8c-short training.

Run on training machine:
    cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
    python scripts/find_step169_samples.py

The GRPO trainer uses a RepeatRandomSampler that:
1. Creates a random permutation of dataset indices (seed=42)
2. Yields indices in that permuted order
3. With 8 GPUs, DistributedSampler further shards the indices

Each generation cycle samples global_batch_size=8 prompts.
Step 169 = generation cycle 21 (0-indexed).
"""

import json
import torch
from torch.utils.data import RandomSampler

# Load the filtered data
anno_path = "/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores_quick/filtered_10k_gt0.3.jsonl"
with open(anno_path) as f:
    samples = [json.loads(l) for l in f if l.strip()]

print(f"Total samples: {len(samples)}")

# Replicate the sampling with seed=42
# HF Trainer sets: seed_worker with self.args.seed
# The RandomSampler uses torch.Generator with seed
generator = torch.Generator()
generator.manual_seed(42)

# Get the full permutation
sampler = RandomSampler(range(len(samples)), generator=generator)
all_indices = list(sampler)

# Each generation cycle = batch_size=8 prompts
# With 8 GPUs and DDP, the DistributedSampler gives each GPU len/8 indices
# But for GRPO, the generation is done collectively - global_batch_size=8
# means 8 prompts total per generation cycle

# Show samples for generation cycles 18-24 (steps 145-193)
print("\n" + "="*80)
for cycle in range(25):
    step = cycle * 8 + 1
    batch_start = cycle * 8
    batch_end = batch_start + 8
    batch_indices = all_indices[batch_start:batch_end]
    
    if 140 <= step <= 200:
        print(f"\n=== Generation cycle {cycle} (step {step}) ===")
        for j, idx in enumerate(batch_indices):
            s = samples[idx]
            print(f"  Prompt {j+1} [idx={idx}]:")
            print(f"    query: {s['query']}")
            print(f"    span: {s['span']}, duration: {s['duration']:.1f}s")
            print(f"    base_model_iou: {s.get('iou', 'N/A')}")
            print(f"    video: {s['video_path'].split('/')[-1]}")
