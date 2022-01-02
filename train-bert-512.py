# Deep Learning
# https://github.com/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb

from sklearn.metrics import recall_score, precision_score, f1_score
from transformers import AutoModel
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from utils import get_args
from tqdm import tqdm
import pandas as pd
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset

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


class TextEDIDataset(Dataset):
    def __init__(self,
                 text_df: pd.DataFrame,
                 max_len,
                 tokenizer_name,
                 is_training=True):
        super().__init__()
        self.is_training = is_training
        self.text_df = text_df
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(
            tokenizer_name,
            max_position_embeddings=max_len,
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

        if self.is_training:
            label = self.text_df.iloc[idx].Label
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask,
                                               dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask,
                                               dtype=torch.long),
            }


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


class DepressionModel(pl.LightningModule):
    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.n_classes = n_classes
        self.n_warmup_steps = n_warmup_steps
        self.n_training_steps = n_training_steps
        self.model = AutoModel.from_pretrained(LM_MODEL_NAME,
                                               #    local_files_only=True,
                                               return_dict=True)
        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)
        self.criterion = CrossEntropyLoss()

    def backbone_forward(self, input_ids, token_type_ids, attention_mask):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
            return output

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask)
        # cls_token = output['last_hidden_state'][:, 0, ...]
        output = self.classifier(output.pooler_output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        loss, outputs = self(input_ids,
                             attention_mask=attention_mask,
                             labels=labels)
        prob = torch.log_softmax(outputs, dim=1)
        self.log("train_loss", loss, prog_bar=True,
                 on_epoch=True, on_step=False, logger=True)
        return {"loss": loss, "predictions": prob.detach(), "labels": labels.detach()}

    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int().detach().numpy()
        predictions = torch.argmax(torch.stack(
            predictions), dim=1).detach().numpy()
        macro_recall = recall_score(y_true=labels, y_pred=predictions, labels=range(
            self.n_classes), average='macro', zero_division=0)
        macro_prec = precision_score(y_true=labels, y_pred=predictions, labels=range(
            self.n_classes), average='macro', zero_division=0)
        macro_f1 = f1_score(y_true=labels, y_pred=predictions,
                            labels=range(self.n_classes), average='macro', zero_division=0)
        self.log('train_macro_recall',
                 macro_recall,
                 logger=True,
                 prog_bar=True,
                 on_epoch=True)
        self.log('train_macro_prec',
                 macro_prec,
                 logger=True,
                 prog_bar=True,
                 on_epoch=True)
        self.log('train_macro_f1',
                 macro_f1,
                 logger=True,
                 prog_bar=True,
                 on_epoch=True)

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        loss, outputs = self(input_ids,
                             attention_mask=attention_mask,
                             labels=labels)

        prob = torch.log_softmax(outputs, dim=1)
        self.log('val_loss',
                 loss.detach(),
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)
        return {"loss": loss, "predictions": prob.detach(), "labels": labels.detach()}

    def validation_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int().detach().numpy()
        predictions = torch.argmax(torch.stack(
            predictions), dim=1).detach().numpy()
        macro_recall = recall_score(y_true=labels, y_pred=predictions, labels=range(
            self.n_classes), average='macro', zero_division=0)
        macro_prec = precision_score(y_true=labels, y_pred=predictions, labels=range(
            self.n_classes), average='macro', zero_division=0)
        macro_f1 = f1_score(y_true=labels, y_pred=predictions,
                            labels=range(self.n_classes), average='macro', zero_division=0)
        self.log('dev_macro_recall',
                 macro_recall,
                 logger=True,
                 prog_bar=True,
                 on_epoch=True)
        self.log('dev_macro_prec',
                 macro_prec,
                 logger=True,
                 prog_bar=True,
                 on_epoch=True)
        self.log('dev_macro_f1',
                 macro_f1,
                 logger=True,
                 prog_bar=True,
                 on_epoch=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )


if __name__ == '__main__':
    args = get_args()

    N_WORKERS = 4
    N_CLASSES = 3
    MAX_LEN = 512
    LM_MODEL_NAME = 'bert-base-cased'
    TOKENIZER_NAME = 'bert-base-cased'

    train_df = pd.read_csv('dataset/train_80_prepr.tsv', sep='\t')
    dev_df = pd.read_csv('dataset/dev_20_prepr.tsv', sep='\t')

    train_dataset = TextEDIDataset(
        train_df, max_len=MAX_LEN, tokenizer_name=TOKENIZER_NAME)
    dev_dataset = TextEDIDataset(
        dev_df, max_len=MAX_LEN, tokenizer_name=TOKENIZER_NAME)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=N_WORKERS,
                              shuffle=True,
                              worker_init_fn=seed_worker,
                              generator=g)
    dev_loader = DataLoader(dev_dataset,
                            batch_size=args.batch_size,
                            num_workers=N_WORKERS,
                            worker_init_fn=seed_worker,
                            generator=g)

    if not args.is_test:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            filename="model-{epoch:02d}-{val_loss:.2f}",
            save_top_k=2,
            mode="min",
        )

        logger = TensorBoardLogger("lightning_logs", name="bert")
        # early_stop_callback = EarlyStopping(monitor="val_loss",
        #                                     min_delta=0.0,
        #                                     patience=3,
        #                                     verbose=True,
        #                                     mode="min")

        trainer = pl.Trainer(gpus=1,
                             logger=logger,
                             callbacks=[checkpoint_callback],
                             log_every_n_steps=10,
                             gradient_clip_val=2,
                             max_epochs=args.epochs)

        steps_per_epoch = len(train_df) // args.batch_size
        total_training_steps = steps_per_epoch * args.epochs
        warmup_steps = total_training_steps // 5  # Use first fifth steps for warmup

        model = DepressionModel(n_classes=N_CLASSES,
                                n_training_steps=total_training_steps,
                                n_warmup_steps=warmup_steps)

        trainer.fit(model,
                    train_dataloaders=train_loader,
                    val_dataloaders=dev_loader)

    else:
        test_df = pd.read_csv('dataset/dev_with_labels_prepr.tsv', sep='\t')
        test_dataset = TextEDIDataset(
            test_df, max_len=MAX_LEN, tokenizer_name=TOKENIZER_NAME)

        # TODO: Reload CKPT
        trainer = pl.Trainer(gpus=1)

        model = DepressionModel(n_classes=N_CLASSES)

        trainer.test(model,
                     test_dataloaders=test_df)
