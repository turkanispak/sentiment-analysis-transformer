# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-customer'  # Directory to save checkpoints/logs — changed from 'out-shakespeare-char'

eval_interval = 250  # keep frequent because we'll overfit
eval_iters = 200
log_interval = 50  # don't print too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True  # Enable WandB logging  # override via command line if you like
wandb_project = 'transformer-sentiment-analysis'  # Using my wandb project
wandb_run_name = 'mini-gpt-block-size-fix' # Name for WandB run

dataset = 'customer_service'  # Tells train.py to use data/customer_service/train.bin and val.bin
gradient_accumulation_steps = 1
batch_size = 16
block_size = 1024  # context of up to 1024 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 2000
lr_decay_iters = 2000  # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
compile = False  # disables Triton, avoids crash
