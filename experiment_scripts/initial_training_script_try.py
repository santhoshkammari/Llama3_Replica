import torch
from torch.utils.data import DataLoader, Dataset

# Define your dataset and DataLoader
class YourDataset(Dataset):
    def __init__(self, ...):
        # Initialize your dataset here

    def __len__(self):
        # Return the total number of samples in your dataset

    def __getitem__(self, idx):
        # Return a sample from your dataset

# Initialize your dataset and DataLoader
dataset = YourDataset(...)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize your model
model_args = ModelArgs()  # Initialize with your desired parameters
model = Transformer(model_args)

# Define your optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 10  # Define your desired number of epochs
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        inputs, targets = batch  # Assuming your dataset returns inputs and targets
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.view(-1, model.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss}")

    # Save checkpoint after each epoch
    checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
