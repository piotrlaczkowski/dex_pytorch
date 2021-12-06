import torch
import torch.nn.functional as F
from torch import nn
from pytorch_tabular.models import BaseModel
from pytorch_tabular.config import ModelConfig
from dataclasses import dataclass
from omegaconf import DictConfig
# import config
from typing import Dict
from loguru import logger

logger.add("logs/model_test.log")


@dataclass
class MyAwesomeModelConfig(ModelConfig):
    use_batch_norm: bool = True


class MyAwesomeRegressionModel(BaseModel):
    def __init__(
        self,
        config: DictConfig,
        **kwargs
    ):
        # Save any attribute that you need in _build_network before calling super()
        # The embedding_dims will be available in the config object and after the super() call, it will be available in self.hparams
        self.embedding_cat_dim = sum([y for x, y in config.embedding_dims])
        logger.info(f"setting up embedding_cat_dim: {self.embedding_cat_dim }")
        super().__init__(config, **kwargs)
        logger.info("INIT -OK!")

    def _build_network(self):
        logger.info("Building net...")
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in self.hparams.embedding_dims]
        )
        # Continuous and Categorical Dimensions are precalculated and stored in the config
        inp_dim = self.embedding_cat_dim + self.hparams.continuous_dim
        self.linear_layer_1 = nn.Linear(inp_dim, 200)
        self.linear_layer_2 = nn.Linear(inp_dim+200, 70)
        self.linear_layer_3 = nn.Linear(inp_dim+70, 1)
        self.input_batch_norm = nn.BatchNorm1d(self.hparams.continuous_dim)
        if self.hparams.use_batch_norm:
            self.batch_norm_2 = nn.BatchNorm1d(inp_dim+200)
            self.batch_norm_3 = nn.BatchNorm1d(inp_dim+70)
        self.embedding_drop = nn.Dropout(0.6)
        self.dropout = nn.Dropout(0.3)
        logger.info("All Build - OK")

    def forward(self, x: Dict):
        continuous_data, categorical_data = x["continuous"], x["categorical"]
        x = [
            embedding_layer(categorical_data[:, i])
            for i, embedding_layer in enumerate(self.embedding_layers)
        ]
        x = torch.cat(x, 1)
        x = self.embedding_drop(x)
        continuous_data = self.input_batch_norm(continuous_data)

        inp = torch.cat([x, continuous_data], 1)
        x = F.relu(self.linear_layer_1(inp))
        x = self.dropout(x)
        x = torch.cat([x, inp], 1)
        if self.hparams.use_batch_norm:
            x = self.batch_norm_1(x)
        x = F.relu(self.linear_layer_2(x))
        x = self.dropout(x)
        x = torch.cat([x, inp], 1)
        if self.hparams.use_batch_norm:
            x = self.batch_norm_3(x)
        x = self.linear_layer_3(x)
        # target_range is a parameter defined in the ModelConfig and will be available in the config
        if (
            (self.hparams.task == "regression")
            and (self.hparams.target_range is not None)
        ):
            for i in range(self.hparams.output_dim):
                y_min, y_max = self.hparams.target_range[i]
                x[:, i] = y_min + nn.Sigmoid()(x[:, i]) * (y_max - y_min)
        return x

    def training_step(self, batch, batch_idx):
        y = batch["target"].squeeze()
        y_hat = self(batch)
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html#torch.nn.functional.nll_loss
        loss = F.poisson_nll_loss(y_hat, y)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["target"].squeeze()
        y_hat = self(batch)
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html#torch.nn.functional.nll_loss
        loss = F.poisson_nll_loss(y_hat, y)
        self.log("valid_loss", loss)
