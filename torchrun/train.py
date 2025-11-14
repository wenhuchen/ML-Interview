import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import torchvision.transforms as transforms

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc_block = nn.Sequential(nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10), nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, image, labels=None):
        x = self.relu(self.conv1(image))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc_block(x)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}

        # Otherwise, return only the logits
        return {"logits": logits}

# Preprocessing function
def preprocess_image(examples):
    # Define the transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    examples["image"] = [transform(image.convert('L')) for image in examples["image"]]
    return examples

def collate_fn(batch):
    pixel_values = [item["image"] for item in batch]
    labels = [item["label"] for item in batch]
    return {
        "image": torch.stack(pixel_values),
        "labels": torch.tensor(labels)
    }

def main():
    print('Starting to load MNIST dataset')
    # Load MNIST dataset
    dataset = load_dataset("ylecun/mnist")

    # Apply preprocessing
    dataset = dataset.with_transform(preprocess_image)

    training_args = TrainingArguments(
        output_dir="./cnn_mnist",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=5,
        learning_rate=1e-3,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        save_strategy="epoch",
    )

    print('Starting to load model')
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=None,  # Not needed for CNNs
        data_collator=collate_fn,
        optimizers=(optimizer, None),
    )

    trainer.train()

    # Evaluate the model
    metrics = trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()