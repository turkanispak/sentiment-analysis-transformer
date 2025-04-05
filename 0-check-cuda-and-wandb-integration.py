# Optional sanity check to see if essentials are working properly

import torch
print("CUDA Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
print("CUDA Version:", torch.version.cuda)
print("cuDNN Version:", torch.backends.cudnn.version())

import wandb

# Replace entity with your actual username
wandb.init(project="transformer-sentiment-analysis", entity="turkanispak-middle-east-technical-university")

# Log something simple
wandb.config = {
    "learning_rate": 0.001,
    "epochs": 5
}

for i in range(5):
    wandb.log({"epoch": i, "dummy_loss": 5 / (i+1)})
