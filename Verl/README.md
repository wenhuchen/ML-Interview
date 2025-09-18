Here I listed some different scripts to start 32b model training

Some arguments to notice:

1. alwasy use `ppo_micro_batch_size_per_gpu` instead of `ppo_micro_batch_size` because the underlying logic for conversion is complicated. `ppo_micro_batch_size_per_gpu` always means the per-gpu batch size when the model is doing updating.

2. if you want really on-policy learning, you should set GLOBAL_BATCH_SIZE==MINI_BATCH_SIZE. The GLOBAL_BATCH_SIZE is actually the global prompt size, which will be multiplied by the `actor_rollout_ref.rollout.n` to rollout a group of responses.

3. Using fsdp partition is actually partitioning the devices to both data+model dimensions. So if you are using fsdp, the data parallelism == devices.