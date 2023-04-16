from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.data.instructions_datamodule import InstructionsDataModule

import transformers


import functools


class InstructLitModule(LightningModule):
    """
    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        # lm_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        n_gpus: int = 2,
        n_nodes: int = 1,
        max_epochs: int = 2,
        accumulate_grad_batches: int = 4,
        datamodule: Any = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            # ignore="datamodule",
            ignore=["lm_model"],
        )
        self.datamodule = datamodule

        self.lm_model = transformers.LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path="decapoda-research/llama-7b-hf",
            torch_dtype=torch.float16,
            device_map="auto",
        ).cpu()

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # for tracking best so far validation accuracy

    def forward(self, example: dict):
        return self.lm_model(**example)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.train_loss.reset()

    def model_step(self, batch: Any):
        output = self.forward(batch)
        loss = output.loss
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            prog_bar=True,
        )

        # return loss or backpropagation will fail
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log(
            "val/loss",
            self.val_loss,
            on_step=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        total_devices = self.hparams.n_gpus * self.hparams.n_nodes
        train_batches = len(self.datamodule.train_dataloader()) // total_devices
        train_steps = (
            self.hparams.max_epochs * train_batches
        ) // self.hparams.accumulate_grad_batches

        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        # if self.hparams.scheduler is not None:
        #     scheduler = self.hparams.scheduler(optimizer=optimizer)
        #     print(scheduler, scheduler is torch.optim.lr_scheduler.LinearLR)
        #     return {
        #         "optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler": scheduler,
        #             "monitor": "train/loss",
        #             "interval": "step",
        #             "frequency": 1,
        #         },
        #     }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    base_model = "decapoda-research/llama-7b-hf"
    tokenizer = transformers.LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    data = InstructionsDataModule(
        tokenizer=tokenizer,
        train_batch_size=1,
    )
    data.prepare_data()
    data.setup()
    example = next(iter(data.train_dataloader()))
    print(example)

    base_model = "decapoda-research/llama-7b-hf"
    device_map = "auto"

    model = transformers.LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=base_model,
        # load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    optimizer = torch.optim.Adam

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau

    instruct_model = InstructLitModule(
        lm_model=model, optimizer=optimizer, scheduler=scheduler
    )
    output = instruct_model(example)
    print(output)
