# Deep Learning
# https://github.com/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from utils import contractions_dict
from tqdm import tqdm
import pandas as pd
from transformers import BertModel, BertForSequenceClassification, BertForTokenClassification

from transformers import BertTokenizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset

import torchmetrics

# Random Seed Initialize
RANDOM_SEED = 2020


def seed_everything(seed=RANDOM_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything()

# Device Optimization
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')

params = {
    'device': device,
    'debug': False,
    'checkpoint': 'bert-base-uncased',  #'allenai/longformer-base-4096',
    'tokenizer_ckpt': 'bert-base-uncased',  #'allenai/longformer-base-4096',
    'output_logits': 768,
    'max_len': 512,
    'batch_size': 64,
    'dropout': 0.3,
    'num_workers': 6,
    'lr': 2e-5,
}

# tokenizer = BertTokenizer.from_pretrained(params['checkpoint'])


class TextEDIDataset(Dataset):
    def __init__(self,
                 text_df: pd.DataFrame,
                 max_len: int = params['max_len'],
                 checkpoint: str = params['tokenizer_ckpt'],
                 is_training=True):
        super().__init__()
        self.is_training = is_training
        self.text_df = text_df
        self.max_len = max_len
        self.checkpoint = checkpoint
        self.tokenizer = BertTokenizer.from_pretrained(
            checkpoint,
            max_position_embeddings=params['max_len'],
            do_lower_case=True)

    def __len__(self):
        return len(self.text_df)

    def __getitem__(self, idx):
        text = str(self.text_df.iloc[idx].Text_data)

        tokenized_text = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        input_ids = tokenized_text['input_ids']
        attention_mask = tokenized_text['attention_mask']
        token_type_ids = tokenized_text['token_type_ids']

        if self.is_training:
            label = self.text_df.iloc[idx].Label
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask,
                                               dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids,
                                               dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask,
                                               dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids,
                                               dtype=torch.long)
            }


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)

train_df = pd.read_csv('dataset/train_prepr.tsv', sep='\t')
dev_df = pd.read_csv('dataset/dev_prepr.tsv', sep='\t')

train_dataset = TextEDIDataset(train_df)
dev_dataset = TextEDIDataset(dev_df)
train_loader = DataLoader(train_dataset,
                          batch_size=params['batch_size'],
                          num_workers=params['num_workers'],
                          worker_init_fn=seed_worker,
                          generator=g)
dev_loader = DataLoader(dev_dataset,
                        batch_size=params['batch_size'],
                        num_workers=params['num_workers'],
                        worker_init_fn=seed_worker,
                        generator=g)

# class DepressionModelV2(nn.Module):
#     def __init__(self, checkpoint=params['checkpoint'], params=params):
#         super(DepressionModelV2, self).__init__()
#         self.checkpoint = checkpoint
#         self.model = BertForTokenClassification.from_pretrained(
#             self.checkpoint,
#             problem_type="multi_label_classification",
#             num_labels=3,
#             gradient_checkpointing=True)

#     def forward(self, input_ids, token_type_ids, attention_mask, labels):
#         return self.model(input_ids, token_type_ids, attention_mask, labels)


class DepressionModel(nn.Module):
    def __init__(self, checkpoint=params['checkpoint'], params=params):
        super(DepressionModel, self).__init__()
        self.checkpoint = checkpoint
        # self.model = BertForSequenceClassification.from_pretrained(
        #     'bert-base-uncased', problem_type="multi_label_classification")
        self.model = BertModel.from_pretrained(checkpoint,
                                               local_files_only=True)

        self.layer_norm = nn.LayerNorm(params['output_logits'])
        self.dropout = nn.Dropout(params['dropout'])
        self.dense = nn.Sequential(nn.Linear(params['output_logits'], 256),
                                   nn.ReLU(), nn.Dropout(params['dropout']),
                                   nn.Linear(256, 3))

    def backbone_forward(self, input_ids, token_type_ids, attention_mask):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
            return output

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.backbone_forward(input_ids, token_type_ids,
                                       attention_mask)
        cls_token = output['last_hidden_state'][:, 0, ...]

        pooled_output = self.layer_norm(cls_token)
        pooled_output = self.dropout(pooled_output)

        preds = self.dense(pooled_output)
        return preds


class ModelLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DepressionModel()
        self.criterion = CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(num_classes=3, top_k=1)
        self.dev_accuracy = torchmetrics.Accuracy(num_classes=3, top_k=1)

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        # no_decay = ["bias", "gamma", "beta"]
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [
        #             p for n, p in param_optimizer
        #             if not any(nd in n for nd in no_decay)
        #         ],
        #         "weight_decay_rate":
        #         0.01
        #     },
        #     {
        #         "params": [
        #             p for n, p in param_optimizer
        #             if any(nd in n for nd in no_decay)
        #         ],
        #         "weight_decay_rate":
        #         0.0
        #     },
        # ]
        optimizer = AdamW(
            self.model.parameters(),
            lr=params['lr'],
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        logits = self.model(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        loss = self.criterion(logits, labels)

        prob = torch.log_softmax(logits, dim=1)

        self.accuracy(prob, labels)

        self.log('acc',
                 self.accuracy,
                 logger=True,
                 prog_bar=True,
                 on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        logits = self.model(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        loss = self.criterion(logits, labels)

        prob = torch.log_softmax(logits, dim=1)

        self.dev_accuracy(prob, labels)

        self.log('val_loss',
                 loss.detach(),
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)
        self.log('val_acc',
                 self.dev_accuracy,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)
        return {
            "val_loss": loss,
        }

    def validation_end(self, outputs):
        self.log('val_acc', self.dev_accuracy, prog_bar=True, on_epoch=True)
        val_loss = sum([out["loss"] for out in outputs]) / len(outputs)

        return {
            "val_loss": val_loss.detach(),
            "val_acc": self.dev_accuracy,
        }

    def test_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        logits = self.model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        loss = self.criterion(logits, labels)

        batch_size = len(labels)

        prob = torch.softmax(logits, dim=1)
        labels_hat = torch.argmax(prob, dim=1)
        correct_count = torch.sum(labels == labels_hat)

        acc = correct_count / batch_size

        self.log('test_acc', acc, prog_bar=True, on_step=True, on_epoch=True)
        return {"loss": loss}

    def test_end(self, outputs):
        test_acc = sum([out["correct_count"]
                        for out in outputs]).float() / sum(out["batch_size"]
                                                           for out in outputs)
        test_loss = sum([out["loss"] for out in outputs]) / len(outputs)
        return {
            "test_loss": test_loss.detach(),
            "test_acc": test_acc,
        }


# 3. Init ModelCheckpoint callback, monitoring 'val_loss'
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    filename="model-{epoch:02d}-{val_loss:.2f}",
    save_top_k=2,
    mode="min",
)

early_stop_callback = EarlyStopping(monitor="val_loss",
                                    min_delta=0.0,
                                    patience=3,
                                    verbose=True,
                                    mode="min")

trainer = pl.Trainer(gpus=1,
                     callbacks=[early_stop_callback, checkpoint_callback],
                     log_every_n_steps=50,
                     gradient_clip_val=1)

model_lightning = ModelLightning()

trainer.fit(model_lightning,
            train_dataloaders=train_loader,
            val_dataloaders=dev_loader)
