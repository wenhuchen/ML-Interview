from transformers import AutoTokenizer, AutoConfig
from modeling_llama import LlamaForCausalLM
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    config = AutoConfig.from_pretrained(model_id)
    print(config)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    messages = [
        {
            "role": "system",
            "content": "You are a pirate chatbot who always responds "
                       "in pirate speak!"
        },
        {"role": "user", "content": "Who are you in the world?"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model.forward(
            input_ids,
            output_attentions=True,
            output_hidden_states=True,
            past_key_values=[],
        )

    for layer in range(10):
        attn_logits = (
            outputs.attentions[layer][0].mean(0).cpu().float().numpy()
        )
        plt.figure(figsize=(8, 6))
        # 'viridis' is a colormap
        plt.imshow(attn_logits, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Value')  # Add a colorbar
        plt.title("Matrix Heatmap")
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        plt.savefig(f'layer_{layer}_attention.png')

        hidden_states_norm = torch.norm(
            outputs.hidden_states[layer][0], dim=-1, p=2
        )
        hidden_states_norm = hidden_states_norm.cpu().float().numpy()

        plt.figure(figsize=(8, 6))  # Optional: Adjust figure size
        plt.plot(range(0, hidden_states_norm.shape[0]),
                 hidden_states_norm,
                 linewidth=2)  # Plot with a label and line width
        plt.grid(True)
        plt.savefig(f'hidden_states_{layer}_norm.png')

        key, value = outputs.past_key_values[layer]
        key_norm = (
            torch.norm(key[0].mean(0), dim=-1, p=2).cpu().float().numpy()
        )
        value_norm = (
            torch.norm(value[0].mean(0), dim=-1, p=2).cpu().float().numpy()
        )

        plt.figure(figsize=(8, 6))  # Optional: Adjust figure size
        # Plot with a label and line width
        plt.plot(
            range(0, key_norm.shape[0]), key_norm, linewidth=2,
            label='key'
        )
        plt.plot(
            range(0, value_norm.shape[0]), value_norm, linewidth=2,
            label='value'
        )
        plt.grid(True)
        plt.legend()
        plt.savefig(f'kv_{layer}_norm.png')
