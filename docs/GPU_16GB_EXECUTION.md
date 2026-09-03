# 16GB / 24GB GPU Execution Notes

## Do not force a fake batch

DynamicWAM's released `DynamicWAMPolicy.sample()` and `HeadFlowRunner.sample_chunk()` are written for batch size 1: input tensors are explicitly unsqueezed to batch 1, diffusion timesteps expand to 1, and action/video noise tensors are allocated with leading dimension 1. DOMINO evaluation is also a closed-loop simulator rollout.

Changing this to batch multiple independent simulators is a runtime/benchmark modification, not a harmless efficiency flag. Do not do it in the causal diagnostic.

## 16GB safe mode

Use one DynamicWAM process per GPU.

1. Keep released BF16 precision.
2. Apply `patches/dynamicwam-cpu-t5.patch`. The frozen UMT5 encoder remains BF16 on CPU; only the cached instruction embedding is transferred to CUDA.
3. Recommended environment variables:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
```

4. Keep `CUDA_VISIBLE_DEVICES` pinned to exactly one device for each run.
5. Do not launch a second model process on a 16GB GPU.
6. Reuse FULL trajectories/results instead of recomputing them when provenance is identical.

This maximizes usable memory without changing policy inputs or benchmark semantics.

## 24GB mode

The default remains one model process. A second process duplicates the video model, action expert and caches; do not assume 24GB is sufficient merely because one process leaves some free memory. Only consider two-process concurrency after measuring peak allocated/reserved memory over a complete L3 episode and leaving several GB of safety margin for SAPIEN/rendering allocations.

## Where batching *is* appropriate

If a later method trains a small validity gate from cached low-dimensional features, that offline training should use the largest batch that fits (typically tens to hundreds of feature examples). That is a separate phase. The current phenomenon experiment is inference-only closed-loop evaluation and remains batch 1 by design.

## Throughput priority

For the v3 smoke, efficiency comes from reducing redundant runs rather than forcing batch size:

- reuse valid FULL runs;
- omit no-op L1 `reset_change` and `stale_hold`;
- run only L1 `full/random_reset` and L3 `full/reset_change/random_reset/stale_hold`;
- stop immediately if the preregistered v3 gate fails.
