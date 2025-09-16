#!/usr/bin/env python3
"""
Ray-based SFT training script for Qwen3-32B model with FSDP size 8.
This script trains the model on synthetic data with 4K sequence length using Ray for distributed training.
"""

import os
import argparse
from typing import Dict, List
from dataclasses import dataclass
import json
import torch
from transformers import (
    AutoModelForCausalLM,
)
from datasets import Dataset
import ray
from omegaconf import DictConfig, OmegaConf, open_dict
from verl.utils.device import get_device_id
from verl.trainer.constants_ppo import PPO_RAY_RUNTIME_ENV
from verl.trainer.ppo.ray_trainer import Role
from verl.single_controller.base import Worker
from verl.single_controller.ray.base import RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.base.decorator import Dispatch, register
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from codetiming import Timer
from verl.utils.device import (
    get_device_name,
    get_nccl_backend,
)
from torch.distributed.device_mesh import init_device_mesh
from verl.utils import hf_tokenizer, hf_processor
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

def create_device_mesh(world_size, fsdp_size):
    device_name = get_device_name()
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


@dataclass
class TrainingConfig:
    """Configuration for SFT training."""
    model_path: str = ./Qwen3-32B"
    output_dir: str = "./checkpoints/qwen3-32b-sft"
    num_samples: int = 50
    max_length: int = 8192
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-6
    num_epochs: int = 3
    save_steps: int = 100
    logging_steps: int = 10
    warmup_steps: int = 100
    fsdp_size: int = 8
    # Profiler configuration
    enable_profiler: bool = True
    profiler_steps: int = 10  # Number of steps to profile
    profiler_activities: List[str] = None  # Will be set to ["cpu", "cuda"] if None

def load_standalone_config(config_path: str = "ppo_trainer_standalone.yaml") -> DictConfig:
    """Load the standalone PPO trainer configuration."""
    if not os.path.exists(config_path):
        # Try relative path from current working directory
        config_path = os.path.join(os.getcwd(), config_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    print(f"Loading configuration from: {config_path}")
    config = OmegaConf.load(config_path)
    return config


def create_merged_config(training_config: TrainingConfig, standalone_config: DictConfig) -> DictConfig:
    """Merge training config with standalone config, allowing overrides."""
    # Start with the standalone config
    merged_config = OmegaConf.create(standalone_config)
    
    # Override with training config values
    with open_dict(merged_config):
        # Update model path
        merged_config.actor_rollout_ref.model.path = training_config.model_path
        
        # Update actor-specific settings
        merged_config.actor_rollout_ref.actor.ppo_mini_batch_size = training_config.batch_size
        merged_config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu = training_config.max_length
        merged_config.actor_rollout_ref.actor.optim.lr = training_config.learning_rate
        merged_config.actor_rollout_ref.actor.optim.lr_warmup_steps = training_config.warmup_steps
        
        # Update FSDP settings
        merged_config.actor_rollout_ref.actor.fsdp_config.fsdp_size = training_config.fsdp_size
        
        # Update data settings
        merged_config.data.max_prompt_length = 2048
        merged_config.data.max_response_length = training_config.max_length
        merged_config.data.train_batch_size = training_config.batch_size * training_config.gradient_accumulation_steps
        merged_config.data.num_samples = training_config.num_samples
        
        # Update trainer settings
        merged_config.trainer.total_epochs = training_config.num_epochs
        merged_config.trainer.save_freq = training_config.save_steps
        merged_config.trainer.experiment_name = "qwen3-32b-sft"
        merged_config.trainer.project_name = "verl_examples"
        merged_config.trainer.default_local_dir = training_config.output_dir
        
        # Update rollout settings to match data settings
        merged_config.actor_rollout_ref.rollout.response_length = training_config.max_length
        
        # Add profiler settings
        merged_config.enable_profiler = training_config.enable_profiler
        merged_config.profiler_steps = training_config.profiler_steps
    
    return merged_config


def generate_synthetic_data(num_samples: int = 1000, max_length: int = 4096, tokenizer=None) -> List[Dict[str, str]]:
    """
    Generate synthetic training data for SFT.
    Creates instruction-response pairs with varying lengths up to max_length tokens.
    """
    synthetic_data = []
    
    # Sample instructions and responses
    instructions = [
        "Explain the concept of machine learning in simple terms.",
        "Write a Python function to calculate the factorial of a number.",
        "Describe the process of photosynthesis in plants.",
        "What are the main differences between supervised and unsupervised learning?",
        "Explain the concept of recursion in programming.",
        "Describe the water cycle and its importance to Earth's climate.",
        "Write a brief explanation of quantum computing principles.",
        "What are the key components of a neural network?",
        "Explain the concept of time complexity in algorithms.",
        "Describe the structure and function of DNA in living organisms."
    ]
    
    responses = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task. It works by finding patterns in data and using those patterns to make predictions or classifications on new, unseen data.",
        "Here's a Python function to calculate factorial:\n\n```python\ndef factorial(n):\n    if n < 0:\n        return None\n    elif n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)\n```\n\nThis function uses recursion to calculate the factorial of a number.",
        "Photosynthesis is the process by which plants convert light energy into chemical energy. It occurs in the chloroplasts of plant cells and involves two main stages: the light-dependent reactions and the Calvin cycle. During this process, plants absorb carbon dioxide from the atmosphere and water from the soil, using sunlight to convert these into glucose and oxygen.",
        "Supervised learning uses labeled training data to learn a mapping from inputs to outputs, while unsupervised learning finds hidden patterns in data without labeled examples. Supervised learning is used for prediction tasks, while unsupervised learning is used for clustering, dimensionality reduction, and pattern discovery.",
        "Recursion is a programming technique where a function calls itself to solve a problem. It consists of two main parts: a base case (the stopping condition) and a recursive case (where the function calls itself with a modified input). Recursion is particularly useful for problems that can be broken down into smaller, similar subproblems.",
        "The water cycle is the continuous movement of water through Earth's atmosphere, land, and oceans. It involves evaporation, condensation, precipitation, and collection. This cycle is crucial for distributing heat around the planet, maintaining ecosystems, and providing fresh water for all living organisms.",
        "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical bits that are either 0 or 1, quantum bits (qubits) can exist in multiple states simultaneously, potentially allowing quantum computers to solve certain problems exponentially faster than classical computers.",
        "A neural network consists of interconnected nodes (neurons) organized in layers: input layer (receives data), hidden layers (process information), and output layer (produces results). Each connection has a weight that determines the strength of the signal, and neurons apply activation functions to determine their output.",
        "Time complexity describes how the runtime of an algorithm increases with the size of the input. It's expressed using Big O notation (e.g., O(n), O(log n), O(n²)). This helps programmers choose the most efficient algorithm for their specific use case and predict performance on larger datasets.",
        "DNA (Deoxyribonucleic acid) is the genetic material that carries hereditary information in all living organisms. It consists of two strands twisted into a double helix, made up of nucleotides containing four bases: adenine, thymine, guanine, and cytosine. DNA stores genetic instructions for development, functioning, and reproduction."
    ]
    
    # Generate synthetic conversations
    for i in range(num_samples):
        # Randomly select instruction and response
        instruction = instructions[i % len(instructions)]
        response = responses[i % len(responses)]
        response = (response + ' ') *  500

        # Create conversation format
        conversation = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]
        
        # Format as training data
        text = format_conversation(conversation)
        
        # Ensure the text doesn't exceed max_length tokens
        if tokenizer is not None:
            # Count tokens in the current text
            tokens = tokenizer.encode(text, add_special_tokens=False)
            current_length = len(tokens)
            
            if current_length > max_length:
                # Calculate how many tokens we need to remove from the response
                instruction_tokens = tokenizer.encode(instruction, add_special_tokens=False)
                # Leave some buffer for special tokens and formatting
                available_tokens = max_length - len(instruction_tokens) - 50
                
                if available_tokens > 0:
                    # Truncate response to fit within token limit
                    response_tokens = tokenizer.encode(response, add_special_tokens=False)
                    truncated_response_tokens = response_tokens[:available_tokens]
                    truncated_response = tokenizer.decode(truncated_response_tokens, skip_special_tokens=True)
                    
                    conversation = [
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": truncated_response}
                    ]
                    text = format_conversation(conversation)
                else:
                    # If instruction is too long, skip this sample
                    continue
        
        synthetic_data.append({
            "text": text,
            "instruction": instruction,
            "response": response
        })
    
    return synthetic_data


def format_conversation(conversation: List[Dict[str, str]]) -> str:
    """Format conversation into training text."""
    formatted_text = ""
    for message in conversation:
        if message["role"] == "user":
            formatted_text += f"<|im_start|>user\n{message['content']}<|im_end|>\n"
        elif message["role"] == "assistant":
            formatted_text += f"<|im_start|>assistant\n{message['content']}<|im_end|>\n"
    return formatted_text


@ray.remote
class SFTWorker(Worker):
    """Ray worker for SFT training with FSDP."""
    
    def __init__(self, merged_config: DictConfig):
        super().__init__()
        self.merged_config = merged_config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset = None
        
        # Profiler setup
        self.profiler = None
        self.profiler_enabled = getattr(merged_config, 'enable_profiler', True)
        self.profiler_steps = getattr(merged_config, 'profiler_steps', 10)
        self.profiler_warmup_steps = 1  # Fixed constant for warmup steps
        
        # Set up logging to file based on rank
        self.log_file = None
        self.setup_logging()

        if not torch.distributed.is_initialized():
            self.worker_print(f"Initializing distributed with rank={self._rank}, world_size={self._world_size}")
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=self._rank,
                world_size=self._world_size,
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            self.worker_print(f"Distributed initialization completed")

        # Create device mesh for FSDP using actual distributed world size
        world_size = torch.distributed.get_world_size()
        fsdp_size = self.merged_config.actor_rollout_ref.actor.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size, fsdp_size)

        # Setup profiler
        self.setup_profiler(self.merged_config.trainer.default_local_dir)    

    def setup_logging(self):
        """Set up logging to individual files for each worker."""
        output_dir = self.merged_config.trainer.default_local_dir

        # Create logs directory
        logs_dir = os.path.join(output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Create log file for this worker
        log_filename = f"worker_{self._rank}.log"
        self.log_file_path = os.path.join(logs_dir, log_filename)
        self.log_file = open(self.log_file_path, 'w')
    
    def worker_print(self, *args, **kwargs):
        """Custom print function that writes to worker-specific log file."""
        #import sys
        from datetime import datetime
        
        # Format the message with timestamp and rank
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]  # Include milliseconds
        
        # Create the formatted message
        message = f"[{timestamp}] Worker {self._rank}: " + " ".join(str(arg) for arg in args)
        
        # Write to log file
        if self.log_file:
            self.log_file.write(message + "\n")
            self.log_file.flush()  # Ensure immediate write

    def cleanup(self):
        """Clean up resources when worker is destroyed."""
        if self.log_file:
            self.log_file.close()
    
    def setup_profiler(self, output_dir: str):
        """Set up PyTorch profiler for detailed timing analysis."""
        if not self.profiler_enabled:
            return
            
        # Create profiler output directory
        profiler_dir = os.path.join(output_dir, "profiler_logs")
        os.makedirs(profiler_dir, exist_ok=True)
        
        self.profiler_dir = profiler_dir
        # Set up profiler activities
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        
        # Configure profiler schedule - start profiling immediately
        profiler_schedule = schedule(
            wait=0,  # Start immediately
            warmup=1,  # Short warmup
            active=self.profiler_steps,
            repeat=1
        )
        
        # Set up profiler with tensorboard trace handler
        trace_handler = tensorboard_trace_handler(
            dir_name=profiler_dir,
            worker_name=f"worker_{self._rank}"
        )
        
        self.profiler = profile(
            activities=activities,
            schedule=profiler_schedule,
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True
        )

    def analyze_profiler_results(self):
        """Analyze and print profiler results focusing on backward pass timing."""
        if not self.profiler or not self.profiler_enabled:
            return
            
        self.worker_print("Analyzing profiler results...")
        
        # Get profiler events
        events = self.profiler.events()
        
        # Filter for backward pass events
        backward_events = []
        for event in events:
            if "backward" in event.name.lower() or "backward" in str(event.key):
                backward_events.append(event)
        
        if backward_events:
            self.worker_print(f"Found {len(backward_events)} backward-related events")
            
            # Sort by CPU time
            backward_events.sort(key=lambda x: x.cpu_time_total, reverse=True)
            
            self.worker_print("\n=== TIMING ANALYSIS ===")
            for i, event in enumerate(backward_events[:10]):  # Top 10
                self.worker_print(f"{i+1}. {event.name}")
                self.worker_print(f"   CPU Time: {event.cpu_time_total:.2f}μs")
                if hasattr(event, 'device_time_total') and event.device_time_total > 0:
                    self.worker_print(f"   CUDA Time: {event.device_time_total:.2f}μs")
                self.worker_print(f"   Count: {event.count}")
                self.worker_print("")
        else:
            self.worker_print("No backward pass events found in profiler results")
        
        # Print overall summary
        self.worker_print("\n=== OVERALL PROFILER SUMMARY ===")
        summary = self.profiler.key_averages()
        
        # Filter for backward-related operations
        backward_summary = [item for item in summary if "backward" in item.key.lower()]
        
        if backward_summary:
            self.worker_print("Backward pass operations (sorted by CPU time):")
            for item in backward_summary[:5]:  # Top 5
                self.worker_print(f"  {item.key}: {item.cpu_time_total:.2f}μs (count: {item.count})")
        else:
            self.worker_print("No backward pass operations found in summary")
        
        # Print top operations overall
        self.worker_print("\nTop 10 operations by CPU time:")
        for item in summary[:10]:
            self.worker_print(f"  {item.key}: {item.cpu_time_total:.2f}μs (count: {item.count})")
    
    def export_profiler_trace(self, step: int):
        """Export profiler trace for detailed analysis."""
        if not self.profiler or not self.profiler_enabled:
            return
            
        events = self.profiler.events()
        trace_file = os.path.join(self.profiler_dir, f"trace_step_{step}_rank_{self._rank}.json")
        
        # Export trace as JSON for programmatic analysis
        trace_data = {
            "step": step,
            "rank": self._rank,
            "events": []
        }
        
        for event in events:
            event_data = {
                "name": event.name,
                "cpu_time_total": event.cpu_time_total,
                "cuda_time_total": getattr(event, 'cuda_time_total', 0),
                "count": event.count,
                "key": str(event.key)
            }
            trace_data["events"].append(event_data)
        
        with open(trace_file, 'w') as f:
            json.dump(trace_data, f, indent=2)
        
        self.worker_print(f"Profiler trace exported to: {trace_file} with {len(events)} events")
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer using VeRL's FSDP pattern."""
        self.worker_print(f"Loading model and tokenizer...")
        
        # Import VeRL utilities
        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.fsdp_utils import get_fsdp_wrap_policy, get_init_weight_context_manager, init_fn
        from transformers import AutoConfig
        import warnings
        
        # Get device info - let VeRL framework handle device assignment
        available_gpus = torch.cuda.device_count()
        if available_gpus == 0:
            raise RuntimeError(f"No CUDA GPUs available on this node. Workers must be placed on nodes with GPUs.")
        
        # Load tokenizer and processor
        model_path = self.merged_config.actor_rollout_ref.model.path
        trust_remote_code = self.merged_config.actor_rollout_ref.model.trust_remote_code
        self.tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(model_path, trust_remote_code=trust_remote_code)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model config
        model_config = AutoConfig.from_pretrained(
            model_path, 
            trust_remote_code=trust_remote_code, 
            attn_implementation="flash_attention_2"
        )
        
        # Override config with tokenizer info
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        update_model_config(model_config, override_config_kwargs=override_config_kwargs)
        
        # Get generation config
        self.generation_config = get_generation_config(model_path, trust_remote_code=trust_remote_code)
        
        # Set torch dtype
        torch_dtype = torch.bfloat16
        
        # Initialize model with proper context
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, 
            mesh=self.device_mesh
        )
        
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_path,
                torch_dtype=torch_dtype,
                config=model_config,
                trust_remote_code=trust_remote_code,
            )
            
            # Flash Attention 2.0 requires the model to be on GPU
            self.model.to(torch_dtype)
            self.worker_print(f'Model device: {self.model.device}')
            
            # Enable gradient checkpointing
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        torch.distributed.barrier()

        # Print model size
        if self._rank == 0:
            print_model_size(self.model)
        
        # Configure FSDP with optimized mixed precision
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32
        )
        
        # Get FSDP wrap policy from merged config
        fsdp_config = self.merged_config.actor_rollout_ref.actor.fsdp_config
        wrap_policy_config = {
            "wrap_policy": "transformer_auto_wrap_policy",
            "min_num_params": fsdp_config.wrap_policy.min_num_params
        }
        auto_wrap_policy = get_fsdp_wrap_policy(
            module=self.model,
            config=wrap_policy_config,
            is_lora=False
        )
        
        # Get sharding strategy
        sharding_strategy = get_sharding_strategy(self.device_mesh)
        
        # Configure CPU offload based on merged config
        cpu_offload = None
        if fsdp_config.param_offload:
            from torch.distributed.fsdp import CPUOffload
            cpu_offload = CPUOffload(offload_params=True)
        
        # Apply FSDP with optimized settings
        self.model = FSDP(
            self.model,
            cpu_offload=cpu_offload,
            param_init_fn=init_fn,
            auto_wrap_policy=auto_wrap_policy,
            device_id=get_device_id(),
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            use_orig_params=fsdp_config.get("use_orig_params", False),
            forward_prefetch=fsdp_config.get("forward_prefetch", False),
        )

        # Note: The old PPO actor setup has been removed as this is now SFT training
        # The model is ready for SFT training with FSDP
    
        self.worker_print(f"Model and tokenizer loaded successfully with FSDP")
    
    def create_dataset(self, synthetic_data: List[Dict[str, str]]):
        """
        Create dataset from synthetic data using standard dataset with DistributedSampler.
        
        This approach ensures:
        - All workers see the same complete dataset
        - Proper data distribution across workers via DistributedSampler
        - Consistent training behavior across different worker configurations
        - Better fault tolerance and reproducibility
        """
        def tokenize_function(example):
            # Tokenize individual text
            max_length = self.merged_config.data.max_prompt_length
            tokenized = self.tokenizer(
                example["text"],
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors="pt"
            )
            # Ensure we return tensors, not lists
            input_ids = tokenized["input_ids"].squeeze()
            attention_mask = tokenized["attention_mask"].squeeze()
            
            # Convert to lists for dataset storage (datasets library prefers lists)
            return {
                "input_ids": input_ids.tolist(),
                "attention_mask": attention_mask.tolist()
            }
        
        # Create full dataset (not sharded)
        full_dataset = Dataset.from_list(synthetic_data)
        
        # Tokenize the full dataset
        self.dataset = full_dataset.map(
            tokenize_function,
            batched=False,  # Process one example at a time
            remove_columns=full_dataset.column_names
        )
        
        self.worker_print(f"Created full dataset with {len(self.dataset)} samples for distributed training")
            

    @register(Dispatch.ONE_TO_ALL)
    def train(self, synthetic_data: List[Dict[str, str]]):
        """Main training function using VeRL pattern."""        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Create dataset
        self.create_dataset(synthetic_data)

        # Create optimizer using merged config
        from torch import optim
        actor_optim = self.merged_config.actor_rollout_ref.actor.optim
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=actor_optim.lr,
            betas=(0.9, 0.999),
            weight_decay=actor_optim.weight_decay,
        )
        
        # Create learning rate scheduler
        from verl.utils.torch_functional import get_constant_schedule_with_warmup
        batch_size = self.merged_config.actor_rollout_ref.actor.ppo_mini_batch_size
        num_epochs = self.merged_config.trainer.total_epochs
        gradient_accumulation_steps = self.merged_config.data.train_batch_size // batch_size
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, 
            num_warmup_steps=actor_optim.lr_warmup_steps
        )
        
        # Simple training loop
        self.model.train()
        
        # Pre-allocate device memory for better performance
        device_id = get_device_id()
        torch.cuda.empty_cache()  # Clear cache before training

        step = 0

        self.worker_print(f"Starting profiler")
        self.profiler.start()
        for epoch in range(num_epochs):
            self.worker_print(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            # Create data loader for each epoch with DistributedSampler
            from torch.utils.data import DataLoader, DistributedSampler
            
            # Create DistributedSampler for proper data distribution across workers
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=True,
                drop_last=True
            )
            
            # Set epoch for proper shuffling across epochs
            sampler.set_epoch(epoch)
            
            dataloader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                sampler=sampler,  # Use DistributedSampler instead of shuffle
                collate_fn=self._collate_fn,
                pin_memory=True,
                num_workers=4,  # Further reduced to avoid overhead
                persistent_workers=True,  # Enable persistent workers
                prefetch_factor=2,  # Reduced prefetch factor
                drop_last=True,  # Drop incomplete batches to ensure consistent batch sizes
            )
            
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device using VeRL's device management
                device_id = get_device_id()
                input_ids = batch["input_ids"].to(device_id, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device_id, non_blocking=True)
                
                # Create labels for language modeling (shifted by 1)
                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]  # Shift targets to the left
                labels[:, -1] = self.tokenizer.eos_token_id  # Ignore the last token (no next token to predict)

                if batch_idx == 0:  # Only print for first batch to avoid spam
                    self.worker_print(f'Batch size: {input_ids.shape[0]}, Shape: {input_ids.shape}')
                    self.worker_print(f'First token IDs: {input_ids[0][:16].tolist()}')
                    self.worker_print(f'Sample text preview: {self.tokenizer.decode(input_ids[0][:50], skip_special_tokens=True)[:100]}...')

                # Forward pass
                with Timer(name="forward_pass", logger=None) as timer:  
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                delta_time = timer.last
                self.worker_print(f"Forward pass time: {delta_time:.2f} seconds")
                
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps
                
                # Backward pass with detailed profiling
                with Timer(name="backward_pass", logger=None) as timer:
                    loss.backward()

                delta_time = timer.last
                self.worker_print(f"Backward pass time: {delta_time:.2f} seconds")

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    step += 1
                    
                    logging_steps = self.merged_config.trainer.save_freq if self.merged_config.trainer.save_freq > 0 else 10
                    if step % logging_steps == 0:
                        self.worker_print(f"Step {step}, Loss: {loss.item():.4f}, LR: {lr_scheduler.get_last_lr()[0]:.2e}")

                    save_freq = self.merged_config.trainer.save_freq
                    if save_freq > 0 and step % save_freq == 0 and self._rank == 0:
                        self._save_checkpoint(step)

                    # Memory cleanup after optimizer step
                    del loss, outputs
                    torch.cuda.empty_cache()

                self.profiler.step()

        self.profiler.stop()
        self.worker_print(f"Stopping profiler")

        # Analyze profiler results
        if self.profiler_enabled and self.profiler:
            self.analyze_profiler_results()
            # Export final profiler trace
            self.export_profiler_trace("final")


        torch.distributed.barrier()
        # Save final model (only on rank 0)
        # if self._rank == 0:
        #     self.worker_print("Saving final model...")
        #     self._save_checkpoint("final")
        
        self.worker_print("Training completed successfully!")
        
        # Close log file
        if self.log_file:
            self.worker_print(f"Log file closed: {self.log_file_path}")
            self.log_file.close()
        
        return {"status": "success", "rank": self._rank}
    
    def _collate_fn(self, batch):
        """Optimized collate function for batching."""
        # Convert lists to tensors and pad sequences to the same length within the batch
        input_ids_list = []
        attention_mask_list = []
        
        for item in batch:
            input_id = item["input_ids"]
            attention_mask = item["attention_mask"]
            
            # Convert lists to tensors
            if isinstance(input_id, list):
                input_id = torch.tensor(input_id, dtype=torch.long)
            if isinstance(attention_mask, list):
                attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            
            input_ids_list.append(input_id)
            attention_mask_list.append(attention_mask)
        
        # Find max length in this batch
        max_length = max(len(input_id) for input_id in input_ids_list)
        
        # Pre-allocate tensors for better performance
        batch_size = len(input_ids_list)
        padded_input_ids = torch.full((batch_size, max_length), self.tokenizer.pad_token_id, dtype=torch.long)
        padded_attention_masks = torch.zeros((batch_size, max_length), dtype=torch.long)
        
        # Fill in the actual data
        for i, (input_id, attention_mask) in enumerate(zip(input_ids_list, attention_mask_list)):
            seq_len = len(input_id)
            padded_input_ids[i, :seq_len] = input_id
            padded_attention_masks[i, :seq_len] = attention_mask
        
        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_masks
        }
    
    def _save_checkpoint(self, step):
        """Save model checkpoint."""
        if self._rank == 0:
            output_dir = self.merged_config.trainer.default_local_dir
            checkpoint_dir = f"{output_dir}/checkpoint-{step}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model state dict using FSDP's proper state_dict handling
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True)
            ):
                model_state = self.model.state_dict()
                torch.save(model_state, f"{checkpoint_dir}/pytorch_model.bin")
            
            # Save tokenizer
            self.tokenizer.save_pretrained(checkpoint_dir)
            
            self.worker_print(f"Saved checkpoint to {checkpoint_dir}")
        
        # Ensure all processes wait for checkpoint saving to complete
        torch.distributed.barrier()


@ray.remote(num_gpus=0, num_cpus=1)
def run(merged_config):
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [8, 8],
        # global_pool_id: [8],
    }
    mapping = {Role.ActorRollout: global_pool_id}
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping)

    resource_pool_manager.create_resource_pool()

    # Create worker group
    print("Creating Ray worker group...")
    worker_cls = RayClassWithInitArgs(cls=SFTWorker, merged_config=merged_config)
    
    worker_group = RayWorkerGroup(resource_pool_manager.get_resource_pool(Role.ActorRollout), worker_cls)

    print(f"Created worker group with {worker_group.world_size} workers")

    print("Loading tokenizer for data generation...")

    model_path = merged_config.actor_rollout_ref.model.path
    trust_remote_code = merged_config.actor_rollout_ref.model.trust_remote_code
    tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Generate synthetic data using merged config
    max_length = merged_config.data.max_prompt_length
    synthetic_data = generate_synthetic_data(merged_config.data.num_samples, max_length, tokenizer)

    # Start training
    print("Starting Ray-based SFT training...")
    results = worker_group.train(synthetic_data=synthetic_data)
    
    # Check results
    success_count = sum(1 for result in results if result.get("status") == "success")
    error_count = sum(1 for result in results if result.get("status") == "error")
    
    print(f"Training completed: {success_count} successful, {error_count} failed")


def main():
    parser = argparse.ArgumentParser(description="Ray-based SFT training for Qwen3-32B")
    parser.add_argument("--model_path", type=str, 
                       default="./Qwen3-32B",
                       help="Path to the Qwen3-32B model")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/qwen3-32b-sft",
                       help="Output directory for checkpoints")
    parser.add_argument("--num_samples", type=int, default=500,
                       help="Number of synthetic samples to generate")
    parser.add_argument("--max_length", type=int, default=4096,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Micro batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--save_steps", type=int, default=100,
                       help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Log every N steps")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Number of warmup steps")
    parser.add_argument("--fsdp_size", type=int, default=8,
                       help="FSDP group size")
    parser.add_argument("--config_path", type=str, 
                       default="ppo_trainer_standalone.yaml",
                       help="Path to the standalone PPO trainer configuration")
    parser.add_argument("--profiler_steps", type=int, default=10,
                       help="Number of steps to profile")
    
    args = parser.parse_args()

    # Load standalone configuration
    standalone_config = load_standalone_config(args.config_path)
    
    # Create training config for merging
    training_config = TrainingConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        warmup_steps=args.warmup_steps,
        fsdp_size=args.fsdp_size,
        enable_profiler=True,
        profiler_steps=args.profiler_steps,
    )
    
    # Merge configurations
    merged_config = create_merged_config(training_config, standalone_config)
    
    # Save merged configuration for reference
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.output_dir + "/merged_config.yaml", "w") as f:
        OmegaConf.save(merged_config, f)
    
    print("Configuration loaded and merged successfully!")
    print(f"Model path: {merged_config.actor_rollout_ref.model.path}")
    print(f"FSDP size: {merged_config.actor_rollout_ref.actor.fsdp_config.fsdp_size}")
    print(f"Learning rate: {merged_config.actor_rollout_ref.actor.optim.lr}")
    print(f"Batch size: {merged_config.actor_rollout_ref.actor.ppo_mini_batch_size}")
    print(f"Max length: {merged_config.data.max_prompt_length}")

    # Check if Ray is not initialized
    if not ray.is_initialized():
        print("Initializing Ray cluster...")
        PPO_RAY_RUNTIME_ENV['env_vars']['RAY_DEBUG'] = "legacy"
        ray.init(
            runtime_env=PPO_RAY_RUNTIME_ENV,
            dashboard_host="0.0.0.0",
        )
    print("Ray cluster initialized successfully")

    ray.get(run.remote(merged_config))


if __name__ == "__main__":
    main()
