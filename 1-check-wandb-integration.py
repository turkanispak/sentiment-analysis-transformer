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
