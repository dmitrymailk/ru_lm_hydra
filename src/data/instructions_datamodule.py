from typing import Any, Dict, Optional, Tuple, List
import os

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import transformers
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

import datasets
from datasets import load_dataset, load_from_disk


class InstructionsDataModule(LightningDataModule):
    """
    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_paths: List[str] = [
            "/home/kosenko/ru_chatGPT/ru_instruct/ru_instruct/sandbox/datasets_processed/IlyaGusev_ru_turbo_alpaca_formatted.json"
        ],
        train_val_split: List[float] = [0.9, 0.1],
        train_batch_size: int = 64,
        valid_batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        tokenizer_max_len: int = 1024,
        tokenizer: AutoTokenizer = None,
        save_data_path: str = "/home/kosenko/deepspeed/ru_lm/data/ru_instruct_v1",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

        self.tokenizer = tokenizer
        self.dataset: datasets.Dataset = None

        self.collate_fn = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        )

    def configure_tokenizer(self):
        if "llama" in self.tokenizer.name_or_path.lower():
            # unk. we want this to be different from the eos token
            self.tokenizer.pad_token_id = 0
            # Allow batched inference
            self.tokenizer.padding_side = "left"

    def prepare_data(self):
        """
        Prepare datasets
        """
        self.configure_tokenizer()
        if not os.path.isdir(self.hparams.save_data_path):
            data_files = {"train": list(self.hparams.data_paths)}
            data = load_dataset(
                "json",
                data_files=data_files,
            )

            dataset = (
                data["train"]
                .shuffle()
                .map(
                    self.generate_and_tokenize_prompt,
                    num_proc=32,
                    # batched=True,
                )
            )
            dataset = dataset.remove_columns("prompt")
            dataset = dataset.train_test_split(
                train_size=self.hparams.train_val_split[0],
                test_size=self.hparams.train_val_split[1],
            )
            dataset.save_to_disk(self.hparams.save_data_path)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        self.configure_tokenizer()
        if not self.data_train and not self.data_val:
            self.dataset = load_from_disk(self.hparams.save_data_path)
            self.data_train = self.dataset["train"]
            self.data_val = self.dataset["test"]

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.valid_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        pass

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    def generate_and_tokenize_prompt(self, prompt):
        # see in notebooks/llama_train_prep.ipynb
        prompt = prompt["prompt"]

        tokenized_prompt = self.tokenizer(
            prompt,
            max_length=self.hparams.tokenizer_max_len,
            truncation=True,
        )
        user_prompt = prompt[: prompt.index("### Assistant:")]
        tokenized_user_prompt = self.tokenizer(
            user_prompt,
            max_length=self.hparams.tokenizer_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_prompt["labels"] = [-100] * (prompt_len) + tokenized_prompt[
            "input_ids"
        ][prompt_len:]
        return tokenized_prompt


if __name__ == "__main__":
    base_model = "decapoda-research/llama-7b-hf"
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path=base_model
    )
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    data = InstructionsDataModule(
        tokenizer=tokenizer,
    )
    data.prepare_data()
    data.setup()
    print(next(iter(data.train_dataloader())))
