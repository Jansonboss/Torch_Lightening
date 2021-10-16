import torch
import argparse
import torch.nn as nn
import pytorch_lightning as pl


from torch.utils import data
from torch.optim import Adam
from torchvision import transforms
from argparse import ArgumentParser
from torchvision.datasets import MNIST
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import random_split, Dataset, DataLoader

from torchmetrics.functional.classification.accuracy import accuracy



class ImageClassifier(nn.Module):

	def __init__(self):
		self.resnet = ResNet()


class ResNet(pl.LightningModule):

	def __init__(self):
		super().__init__()
		self.l1 = nn.Linear(28*28, 64)
		self.l2 = nn.Linear(64, 64)
		self.l3 = nn.Linear(64, 10)
		self.dropout = nn.Dropout(0.1)
		self.loss = nn.CrossEntropyLoss()
		self.accuracy = accuracy
	
	def forward(self, x):
		h1 = nn.functional.relu(self.l1(x))
		h2 = nn.functional.relu(self.l2(h1))
		d = self.dropout(h1 + h2)
		logits = self.l3(d)
		return logits

	def configure_optimizers(self):
		optimizer = Adam(self.parameters())
		return optimizer

	def train_dataloader(self):
		data = MNIST("data", train=True, download=True, transform=transforms.ToTensor())
		self.train_data, self.val_data = random_split(data, [55000, 5000])
		train_loader = DataLoader(dataset=self.train_data, batch_size=self.args.batch_size, shuffle=True)
		return train_loader
	 
	def val_dataloader(self):
		val_loader = DataLoader(dataset=self.val_data, batch_size=self.args.batch_size)
		return val_loader
	
	def training_step(self, batch, batch_idx):
		X, y = batch
		batch_size = X.size(0)
		X = X.view(batch_size, -1)
		logits = self(X)

		loss, acc = self.loss(logits, y), self.accuracy(logits, y)
		progress_bar_info = {"acc": acc}

		self.configure_loggers(loss, acc, type = "train")
		return {"loss": loss, "progress_bar_info": progress_bar_info}

	def train_epoch_end(self, train_step_outputs):
		avg_train_loss = torch.tensor([x['loss'] for x in train_step_outputs]).mean()
		avg_train_acc = torch.tensor([x['progress_bar_info']["acc"] for x in train_step_outputs]).mean()
		progress_bar_info = {"avg_train_acc": avg_train_acc}

		self.configure_loggers(loss=avg_train_loss, acc=avg_train_acc, type = "avg_train_per_epoch")

		self.log_dict(
			 logs = {'train_loss': avg_train_loss, 
			 'avg_train_acc': avg_train_acc,
			}
		)


		return {"train_loss": avg_train_loss, "progress_bar_info": progress_bar_info}

	def validation_step(self, batch, batch_idx):
		
		results = self.training_step(batch=batch, batch_idx=batch_idx)
		loss, acc = results["loss"], results["progress_bar_info"]["acc"]
		self.configure_loggers(loss, acc, type = "val")
		return results

	def validation_epoch_end(self, validation_step_outputs):
		avg_val_loss = torch.tensor([x['loss'] for x in validation_step_outputs]).mean()
		avg_val_acc = torch.tensor([x['progress_bar_info']["acc"] for x in validation_step_outputs]).mean()
		progress_bar_info = {"avg_val_acc": avg_val_acc}

		self.configure_loggers(loss=avg_val_loss, acc=avg_val_acc, type = "avg_val_per_epoch")

		return {"val_loss": avg_val_loss, "progress_bar_info": progress_bar_info}
	
	def configure_loggers(self, loss, acc, type):
		# for loggers' information
		self.log(name=f"{type}_loss", value=loss)
		self.log(name=f"{type}_acc", value=acc)
	
	def add_model_specific_args(self):

		parser = argparse.ArgumentParser(description='VAE MNIST Example')
		parser.add_argument('--batch-size', type=int, default=128, metavar='N',
							help='input batch size for training (default: 128)')

		parser.add_argument('--epochs', type=int, default=10, metavar='N',
							help='number of epochs to train (default: 10)')

		parser.add_argument('--quick-test', type=bool, default=False,
							help='tried compiling your stupid code (default False)')
		self.args = parser.parse_args()


def main():

	# init model
	model = ResNet()
	# parse arg from cmd
	model.add_model_specific_args()
	# set up training and val data
	train_dataloader = model.train_dataloader()
	val_dataloader = model.val_dataloader()
	# instrument experiment with W&B
	wandb_logger = WandbLogger(project="MNIST", log_model="all")

	trainer = pl.Trainer(
		max_epochs=model.args.epochs, 
		fast_dev_run=model.args.quick_test,
		log_every_n_steps=2,
		logger=wandb_logger
	)

	trainer.fit(
		model=model, 
		train_dataloader=train_dataloader,
		val_dataloaders=val_dataloader
	)


if __name__ == "__main__":
	main()