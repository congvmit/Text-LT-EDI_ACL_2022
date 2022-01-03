import torch
import torch.nn as nn

dummy_model = nn.Linear(2, 1)

optimizer = torch.optim.AdamW(params=dummy_model.parameters(), lr=0.001)

warmup_steps = 20
total_training_steps = 100

scheduler = torch.optim.lr_scheduler.Co(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps)

learning_rate_history = []

for step in range(total_training_steps):
    optimizer.step()
    scheduler.step()
    learning_rate_history.append(optimizer.param_groups[0]['lr'])
