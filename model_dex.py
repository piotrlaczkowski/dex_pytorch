from dataclasses import dataclass, field
from typing import Dict, List, Optional

import probflow as pf
import probflow.utils.ops as O
import pytorch_lightning as pl
import torch
import torch.nn as nn

# import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig
from pytorch_tabular.config import ModelConfig, _validate_choices
from pytorch_tabular.models import BaseModel
from pytorch_tabular.utils import _initialize_layers, _linear_dropout_bn

logger.add("logs/model_dex.log")


@dataclass
class DexModelConfig(ModelConfig):
    """
    DexModel configuration

    Args:
        task (str): Specify whether the problem is regression of classification.Choices are: regression classification

        learning_rate (float): The learning rate of the model

        layers (str): Hyphen-separated number of layers and units in the classification head. eg. 32-64-32.

        batch_norm_continuous_input (bool): If True, we will normalize the continuous layer by passing it through a BatchNorm layer

        activation (str): The activation type in the classification head.
            The default activation in PyTorch like ReLU, TanH, LeakyReLU, etc.
            https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity

        embedding_dims (Union[List[int], NoneType]): The dimensions of the embedding for each categorical column
            as a list of tuples (cardinality, embedding_dim). If left empty, will infer using the cardinality of the categorical column
            using the rule min(50, (x + 1) // 2)

        embedding_dropout (float): probability of an embedding element to be zeroed.
        dropout (float): probability of an classification element to be zeroed.
        use_batch_norm (bool): Flag to include a BatchNorm layer after each Linear Layer+DropOut
        initialization (str): Initialization scheme for the linear layers. Choices are: `kaiming` `xavier` `random`

    Raises:
        NotImplementedError: Raises an error if task is not in ['regression','classification']

    Notes:
        Additional input params are the following:

         loss (Union[str, NoneType]): The loss function to be applied.
            By Default it is MSELoss for regression and CrossEntropyLoss for classification.
            Unless you are sure what you are doing, leave it at MSELoss or L1Loss for regression and CrossEntropyLoss for classification
            (comes form the BaseModel init param: custom_loss)

        metrics (Union[List[str], NoneType]): the list of metrics you need to track during training.
            The metrics should be one of the metrics implemented in PyTorch Lightning.
            By default, it is Accuracy if classification and MeanSquaredLogError for regression
            (comes form the BaseModel init param: custom_metrics)

        target_range (Union[List, NoneType]): The range in which we should limit the output variable. Currently ignored for multi-target regression
            Typically used for Regression problems. If left empty, will not apply any restrictions
    """

    layers: str = field(
        default="128-64-32",
        metadata={"help": "Hyphen-separated number of layers and units in the classification head. eg. 32-64-32."},
    )
    batch_norm_continuous_input: bool = field(
        default=True,
        metadata={"help": "If True, we will normalize the contiinuous layer by passing it through a BatchNorm layer"},
    )
    activation: str = field(
        default="ReLU",
        metadata={
            "help": "The activation type in the classification head. The default activaion in PyTorch like ReLU, TanH, LeakyReLU, etc. https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity"
        },
    )
    embedding_dropout: float = field(
        default=0.4,
        metadata={"help": "probability of an embedding element to be zeroed."},
    )
    dropout: float = field(
        default=0.3,
        metadata={"help": "probability of an classification element to be zeroed."},
    )
    use_batch_norm: bool = field(
        default=False,
        metadata={"help": "Flag to include a BatchNorm layer after each Linear Layer+DropOut"},
    )
    initialization: str = field(
        default="kaiming",
        metadata={
            "help": "Initialization scheme for the linear layers",
            "choices": ["kaiming", "xavier", "random"],
        },
    )
    _module_src: str = field(default="dex")
    _model_name: str = field(default="DexEmbeddingModel")
    _config_name: str = field(default="DexModelConfig")


class FeedForwardBackbone(pl.LightningModule):
    """
    Module helper initializing all the layers dynamically
    """

    def __init__(self, config: DictConfig, **kwargs):
        logger.info(f"provided embedding model dims: {config.embedding_dims}")
        self.embedding_cat_dim = sum([y for x, y in config.embedding_dims])
        logger.info(f"embedding categorical dims: {self.embedding_cat_dim}")
        super().__init__()
        self.save_hyperparameters(config)
        self._build_network()

    def _build_network(self):
        logger.info("building linear layers")
        # Linear Layers
        layers = []
        _curr_units = self.embedding_cat_dim + self.hparams.continuous_dim

        if self.hparams.embedding_dropout != 0 and self.embedding_cat_dim != 0:
            layers.append(nn.Dropout(self.hparams.embedding_dropout))
            logger.info(f"adding dropout to embedding layer: {self.hparams.embedding_dropout}")

        for units in self.hparams.layers.split("-"):
            logger.info(
                f"adding layer with dim: {units}"
                f", act: { self.hparams.activation}"
                f", init: {self.hparams.initialization}"
                f", batch norm: {self.hparams.use_batch_norm}"
                f", _curr_units: {_curr_units}"
                f", dropout: {self.hparams.dropout,}"
            )
            layers.extend(
                _linear_dropout_bn(
                    self.hparams.activation,
                    self.hparams.initialization,
                    self.hparams.use_batch_norm,
                    _curr_units,
                    int(units),
                    self.hparams.dropout,
                )
            )
            _curr_units = int(units)
        self.linear_layers = nn.Sequential(*layers)
        self.output_dim = _curr_units

    def forward(self, x):
        x = self.linear_layers(x)
        return x


class DexModel(BaseModel):
    """
    Dynamic Regression Dex model.
    """

    def __init__(self, config: DictConfig, **kwargs):
        # The concatenated output dim of the embedding layer
        self.embedding_cat_dim = sum([y for x, y in config.embedding_dims])
        logger.info(f"embedding categorical dims: {self.embedding_cat_dim}")
        super().__init__(config, **kwargs)

    def _build_network(self):
        # Embedding layers
        logger.info("building embedding layers")
        self.embedding_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in self.hparams.embedding_dims])
        # Continuous Layers
        logger.info("building continuous layers")
        if self.hparams.batch_norm_continuous_input:
            self.normalizing_batch_norm = nn.BatchNorm1d(self.hparams.continuous_dim)
        # Backbone
        logger.info("building backbone")
        self.backbone = FeedForwardBackbone(self.hparams)

        # Adding the last layer
        logger.info(f"adding last layer, dims: {self.hparams.output_dim}")
        self.output_layer = nn.Linear(
            self.backbone.output_dim, self.hparams.output_dim
        )  # output_dim auto-calculated from other config
        _initialize_layers(self.hparams.activation, self.hparams.initialization, self.output_layer)

        # adding bayesian mean and std nets
        self.mean = pf.DenseNetwork([self.hparams.output_dim, 64, 32, 1])
        self.std = pf.DenseNetwork([self.hparams.output_dim, 64, 32, 1])

    def unpack_input(self, x: Dict):
        continuous_data, categorical_data = x["continuous"], x["categorical"]
        if self.embedding_cat_dim != 0:
            x = []
            # for i, embedding_layer in enumerate(self.embedding_layers):
            #     x.append(embedding_layer(categorical_data[:, i]))
            x = [embedding_layer(categorical_data[:, i]) for i, embedding_layer in enumerate(self.embedding_layers)]
            x = torch.cat(x, 1)

        if self.hparams.continuous_dim != 0:
            if self.hparams.batch_norm_continuous_input:
                continuous_data = self.normalizing_batch_norm(continuous_data)

            if self.embedding_cat_dim != 0:
                x = torch.cat([x, continuous_data], 1)
            else:
                x = continuous_data
        return x

    def forward(self, x: Dict):
        # taking inspiration from the example: https://github.com/brendanhasz/probflow
        x = self.unpack_input(x)
        backbone_net = self.backbone(x)
        backbone_net = torch.tensor(backbone_net)
        z = torch.nn.ReLU(backbone_net)
        return {"logits": pf.Normal(self.mean(z), torch.exp(self.std(z))), "backbone_features": backbone_net}

        # probabilistic output
        # x = torch.tensor(x)
        # y_hat_dist = pf.Normal(self.backbone(x), self.s())
        # return {"logits": y_hat_dist, "backbone_features": x}

    # def training_step(self, batch, batch_idx):
    #     y = batch["target"].squeeze()
    #     y_hat = self(batch)["logits"]
    #     # https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html#torch.nn.functional.nll_loss
    #     loss = F.poisson_nll_loss(y_hat, y)
    #     # logs metrics for each training_step,
    #     # and the average across the epoch, to the progress bar and logger
    #     self.log("train_loss", loss, on_step=True,
    #              on_epoch=True, prog_bar=True, logger=True)
    #     return loss

    # def test_step(self, batch, batch_idx):
    #     y = batch["target"].squeeze()
    #     y_hat = self(batch)["logits"]
    #     loss = F.poisson_nll_loss(y_hat, y)
    #     self.log("test_loss", loss)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     y = batch["target"].squeeze()
    #     y_hat = self(batch)["logits"]
    #     # https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html#torch.nn.functional.nll_loss
    #     loss = F.poisson_nll_loss(y_hat, y)
    #     self.log("valid_loss", loss)


class NeuralLinear(pf.ContinuousModel):
    def __init__(self, dims):
        self.net = pf.DenseNetwork(dims, probabilistic=False)
        self.loc = pf.Dense(dims[-1], 1)
        self.std = pf.Dense(dims[-1], 1)

    def __call__(self, x):
        loc = O.relu(self.net(x.float()))
        return pf.Normal(self.loc(loc), O.softplus(self.std(loc)))


class DenseRegression(pf.Model):
    """
    Playing with: https://probflow.readthedocs.io/en/latest/examples/fully_connected.html
    Args:
        pf ([type]): [description]
    """

    def __init__(self, d_in):
        self.net = pf.Sequential(
            [
                pf.Dense(d_in, 32),
                torch.nn.ReLU(),
                # pf.Dense(512, 256),
                # torch.nn.ReLU(),
                # pf.Dense(256, 128),
                # torch.nn.ReLU(),
                # pf.Dense(128, 64),
                # torch.nn.ReLU(),
                # pf.Dense(64, 32),
                # torch.nn.ReLU(),
                pf.Dense(32, 16),
                torch.nn.ReLU(),
                pf.Dense(16, 1),
                torch.nn.ReLU(),
            ]
        )
        self.s = pf.ScaleParameter()

    def __call__(self, x):
        x = torch.tensor(x)
        loc = O.relu(self.net(x.float()))
        # return pf.Normal(loc, self.s())
        dist = torch.distributions.log_normal.LogNormal(loc, self.s(), validate_args=None)
        return dist
