import argparse
from typing import Optional, Collection, Dict, Tuple, Union

import pytorch_lightning as pl
from datasets import load_dataset
import torch
from transformers import AutoTokenizer


class DataModule(pl.LightningDataModule):
	"""[summary]

	In pytorch we use dataLoader; while in pl we use DataModule. And it is a more
	structured definition and allows for additional optimizations such as distriibution
	worload between CPU & GPU

	DataModule is defined by an interaces:
	
		1. prepare_data[Optional]: called only once on 1 GPU; Typcial is downloading data
		2. setup: called on each GPU seperately and accetps stage to define if we 
		are at fit or test setp
		3. train_dataloader, val_dataloader, and test_dataloader to load each dataset
	

	DataModule encapsulates the five steps involved in data
	processing in pytorch;

		1. Download / tokenize / process
		2. clean, preprocess data and (mybe) save into disk
		3. Load inside Dataset
		4. Apply vector transform (rotate image, decolorize, padding..)
		5. Warp inside a DataLoader
	"""

	def __init__(self, 
				model_name="google/bert_uncased_L-2_H-128_A-2", 
				batch_size: int = 32, args: argparse.Namespace = None):
		"""
		initialize necessary components and toolkits
		eg. tokenizer, batch_size and other parameter
		"""
		super().__init__()

		# argparse -> parse the parameters
		self.args = vars(args) if args is not None else {}
		self.batch_size = self.args.get("batch_size", __BATCH_SIZE__)
		self.num_workers = self.args.get("num_workers", __NUM_WORKERS__)

		# loading toolkits
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
	
	def prepare_data(self) -> None:
		"""
		[Optional] which is called only once and on 1 GPU
		Typically speaking, something like donwload data;  (so don't set state `self.x = y`).

		Dataset({
  				features: ['sentence', 'label', 'idx'],
    			num_rows: 8551
			})
		"""

		cola_dataset = load_dataset("glue", "cola")
		self.train_data = cola_dataset["train"]
		self.val_data = cola_dataset["validation"]

	def tokenize_data(self, data):
		"""self defined methods to preprocess data"""
		return data

	def setup(self, stage: Optional[str] = None) -> None:
		"""[summary]

		Args:
			stage (Optional[str], optional): [description]. Defaults to None.
		
		Split into train, val and test dataset; and then set dimentions
		should assign "torch DataSet" objects to self.data_train, self.data_val,
		and optionally self.data_test
		"""
		
		if stage == "fit" or stage is None:
			self.train_data = self.train_data.map(self.tokenize_data, batched=True)
			self.val_data = self.val_data.map(self.tokenize_data, batched=True)

			# toeknize data when fitting
			self.train_data.set_format(
					type="torch", columns= ["input_ids", "attention_mask", "label"]
			)

			self.train_data.set_format(
					type="torch", columns= ["input_ids", "attention_mask", "label"]
			)

	
	def train_dataloader(self):
		return torch.utils.data.DataLoader(
			self.train_data, batch_size = self.batch_size, shuffle = True
		)
	
	def val_dataloader(self):
		return torch.utils.data.DataLoader(
			self.train_data, batch_size = self.batch_size, shuffle = False
		)