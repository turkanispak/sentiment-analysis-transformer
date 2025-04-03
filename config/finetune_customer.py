import time

out_dir = 'out-gpt2-finetune'
eval_interval = 5
eval_iters = 40
wandb_log = True # don't mind if I do
wandb_project = 'transformer-sentiment-analysis'
wandb_run_name = 'ft-gpt2-customer-' + str(time.time())

dataset = 'data_gpt2'  # uses train.txt and val.txt inside this folder
init_from = 'gpt2' # this is the GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 200

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False

compile = False  # disables Triton + TorchDynamo to avoid Windows issues
