import torch
from torch.nn import Transformer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset



# Define dataset class
class TextDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]["text"], int(self.dataset[idx]["label"])

# Load IMDb dataset from Hugging Face
dataset = load_dataset("imdb")

# Preprocess the dataset
text_dataset = TextDataset(dataset["train"])

# Define DataLoader
batch_size = 8
dataloader = DataLoader(text_dataset, batch_size=batch_size, shuffle=True)

# Define the model
class SimpleTransformer(torch.nn.Module):
    def __init__(self):
        super(SimpleTransformer, self).__init__()
        self.transformer = Transformer(
            d_model=256,  # Decrease hidden dimension
            nhead=4,      # Decrease number of attention heads
            num_encoder_layers=2,  # Decrease number of layers
            num_decoder_layers=2,
        )

    def forward(self, x):
        return self.transformer(x)

model = SimpleTransformer()

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.view(-1, outputs.size(-1)), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss}")

# Save the model
torch.save(model.state_dict(), "imdb_model.pt")
print("Model saved.")
