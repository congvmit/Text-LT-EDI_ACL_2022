import torch

ckpt = torch.load(
    '/mnt/hdd4T/Challenges/Text-LT-EDI_ACL_2022/lightning_logs/BERT/version_0/checkpoints/model-epoch=02-val_loss=0.46.ckpt'
)

state_dict = ckpt['state_dict']
state_dict_new = {}
for k, v in state_dict.items():
    state_dict_new[k.replace('model', 'backbone')] = v

ckpt['state_dict'] = state_dict_new
torch.save(
    ckpt,
    '/mnt/hdd4T/Challenges/Text-LT-EDI_ACL_2022/lightning_logs/BERT/version_0/checkpoints/model-epoch=02-val_loss=0.46_cvt.ckpt'
)
