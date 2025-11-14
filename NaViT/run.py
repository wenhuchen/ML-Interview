from transformers import AutoTokenizer, AutoConfig, AutoProcessor
from idefics2 import Idefics2ForConditionalGeneration
import torch
from PIL import Image


if __name__ == '__main__':
    model_id = "HuggingFaceM4/idefics2-8b"

    config = AutoConfig.from_pretrained(model_id)
    print(config)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Define mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # Reshape to (C, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    processor = AutoProcessor.from_pretrained(model_id)

    image_1 = Image.open('dog.jpg')

    inputs = processor(images=[image_1], text='Add a caption for the image <image>', return_tensors="pt")

    for i, image in enumerate(inputs['pixel_values'][0]):
        tensor = image.cpu()

        original_tensor = (tensor * std) + mean

        original_tensor = torch.clip(original_tensor * 255.0, 0, 255).byte()
        original_image = original_tensor.permute(1, 2, 0).numpy()

        image = Image.fromarray(original_image)
        image.save(f'image_{i}.jpg')

    model = Idefics2ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0", 
    )
    model.eval()

    inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=20)

    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))