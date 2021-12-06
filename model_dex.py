import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from pytorch_tabular.models import BaseModel
from pytorch_tabular.config import ModelConfig
from dataclasses import dataclass, field
from omegaconf import DictConfig
# import config
from typing import Dict, List, Tuple


@dataclass
class MyAwesomeModelConfig(ModelConfig):
    use_batch_norm: bool = True
    embedding_sizes: List[Tuple[int, int]] = field(
        default_factory=lambda: [(10, 10)])
    dropouts: float = 0.3
    layers_size: Tuple[int] = (256, 128,  64,  1)


class MyAwesomeRegressionModel(BaseModel):
    def __init__(
        self,
        config: DictConfig,
        **kwargs
    ):
        logger.info("Initializing MyAwesomeRegressionModel")
        # Save any attribute that you need in _build_network before calling super()
        # The embedding_dims will be available in the config object and after the super() call, it will be available in self.hparams
        # self.embedding_cat_dim = sum([y for x, y in config.embedding_dims])
        super().__init__(config, **kwargs)
        logger.info("Initialized - OK!")

    def _build_network(self):
        logger.info(
            f"setting up embeddings for #{len(self.hparams.embedding_sizes)} columns with: {self.hparams.embedding_sizes}")
        # ====================== CAETGORICAL ======================
        self.embeddings = nn.ModuleList(
            [nn.Embedding(categories + 1, size)
             for categories, size in self.hparams.embedding_sizes]
        )
        # length of all embeddings combined
        self.n_emb = sum(e.embedding_dim for e in self.embeddings)
        logger.info(f"embeddings input total length: {self.n_emb}")

        self.total_input_len = self.n_emb + self.hparams.continuous_dim
        logger.info(f"TOTAL INPUT length: {self.total_input_len}")

        self.emb_drop = nn.Dropout(self.hparams.dropouts)

        # ====================== NUMERICAL ======================
        self.bn_num = nn.BatchNorm1d(self.hparams.continuous_dim)
        # ====================== JOINED CORE ======================
        # first layer
        self.lin1 = nn.Linear(self.total_input_len,
                              self.hparams.layers_size[0])
        self.drop1 = nn.Dropout(self.hparams.dropouts)
        self.bn1 = nn.BatchNorm1d(self.hparams.layers_size[0])
        # second layer
        self.lin2 = nn.Linear(
            self.hparams.layers_size[0],  self.hparams.layers_size[1])
        self.drop2 = nn.Dropout(self.hparams.dropouts)
        self.bn2 = nn.BatchNorm1d(self.hparams.layers_size[1])
        # output
        self.lin3 = nn.Linear(
            self.hparams.layers_size[2],  self.hparams.layers_size[3])

    def forward(self, x: Dict):
        # the data is passed to the model as a dictionary
        continuous_data, categorical_data = x["continuous"], x["categorical"]

        emb_input = []
        for idx, emb_layer in enumerate(self.embeddings):
            logger.info(f"emb: {emb_layer}")
            logger.info(f"x_cat shape: {categorical_data.shape}")
            try:
                data = categorical_data[:, idx]
                emb_input.append(emb_layer(data))
                logger.info(f"Embeddings idx: {idx} OK !!!")
            except Exception as err:
                logger.error(f"ERROR idx: {idx}: {err}")

        logger.info("Embeddings generation OK !!!")
        x = torch.cat(emb_input, 1)
        x = self.emb_drop(x)  # setting up embeddings dropout
        # numerical
        x2 = self.bn_num(continuous_data)  # batch normalization
        # concatenation of categorical_data and numerical data
        x = torch.cat([x, x2], 1)

        # first layer
        x = self.lin1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.bn1(x)
        # second layer
        x = F.relu(self.lin2(x))
        x = self.drop2(x)
        x = self.bn2(x)
        # output
        x = self.lin3(x)
        return x

        # # target_range is a parameter defined in the ModelConfig and will be available in the config
        # if (
        #     (self.hparams.task == "regression")
        #     and (self.hparams.target_range is not None)
        # ):
        #     for i in range(self.hparams.output_dim):
        #         y_min, y_max = self.hparams.target_range[i]
        #         x[:, i] = y_min + nn.Sigmoid()(x[:, i]) * (y_max - y_min)
        # return x

    def training_step(self, batch, batch_idx):
        x_cat, x_num, y = batch
        y_hat = self(x_cat, x_num)
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html#torch.nn.functional.nll_loss
        loss = F.nll_loss(y_hat, y)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_cat, x_num, y = batch
        y_hat = self(x_cat, x_num)
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html#torch.nn.functional.nll_loss
        loss = F.nll_loss(y_hat, y)
        self.log("val_loss", loss)

# class DexLightning(pl.LightningModule):
#     """
#     Training Procedure:
#         autoencoder = Autoencoder()
#         trainer = pl.Trainer(gpus=1)
#         trainer.fit(autoencoder, train_dataloader, val_dataloader)

#     Args:
#         embedding_sizes (List[Tuples]) -> data in the format (class unique nr, emb_size)
#         n_cont (int): number of continuse variables (numerical columns)
#     """

#     def __init__(self):
#         super().__init__()

#         # setting up general model architecture
#         self.model_config = {
#             "layers_size": (256, 128, 64, 1),
#             "dropouts": 0.3,
#         }
#         logger.info(f"MODEL CONFIG: {self.model_config}")

#         # Initialize the modules we need to build the network
#         self.embedding_sizes = config.EMBEDDING_SIZES
#         self.n_cont = len(config.NUM_COLS)

#         logger.info(f"setting up embeddings for #{len(self.embedding_sizes)} columns with: {self.embedding_sizes}")

#         # ====================== CAETGORICAL ======================
#         self.embeddings = nn.ModuleList(
#             [nn.Embedding(categories + 1, size) for categories, size in self.embedding_sizes]
#         )
#         self.n_emb = sum(e.embedding_dim for e in self.embeddings)  # length of all embeddings combined
#         logger.info(f"embeddings input total length: {self.n_emb}")
#         self.total_input_len = self.n_emb + self.n_cont
#         logger.info(f"TOTAL INPUT length: {self.total_input_len}")

#         self.emb_drop = nn.Dropout(self.model_config["dropouts"])

#         # ====================== NUMERICAL ======================
#         self.bn_num = nn.BatchNorm1d(self.n_cont)
#         # ====================== JOINED CORE ======================
#         # first layer
#         self.lin1 = nn.Linear(self.total_input_len, self.model_config["layers_size"][0])
#         self.drop1 = nn.Dropout(self.model_config["dropouts"])
#         self.bn1 = nn.BatchNorm1d(self.model_config["layers_size"][0])
#         # second layer
#         self.lin2 = nn.Linear(self.model_config["layers_size"][0], self.model_config["layers_size"][1])
#         self.drop2 = nn.Dropout(self.model_config["dropouts"])
#         self.bn2 = nn.BatchNorm1d(self.model_config["layers_size"][1])
#         # output
#         self.lin3 = nn.Linear(self.model_config["layers_size"][2], self.model_config["layers_size"][3])

#     def build_categorical_embeddings(self):
#         pass

#     def forward(self, x_cat, x_cont):
#         """
#         The forward function is where the computation of the module is taken place, and is executed when you call the module (nn = MyModule(); nn(x)).
#         In the init function, we usually create the parameters of the module, using nn.Parameter, or defining other modules that are used in the forward function.
#         The backward calculation is done automatically, but could be overwritten as well if wanted.
#         """
#         # inputs
#         # logger.info(f"self.embeddings: {self.embeddings}")
#         ## categorical
#         emb_input = []
#         for idx, emb_layer in enumerate(self.embeddings):
#             logger.info(f"emb: {emb_layer}")
#             logger.info(f"x_cat shape: {x_cat.shape}")
#             try:
#                 data = x_cat[:, idx]
#                 emb_input.append(emb_layer(data))
#                 logger.info(f"Embeddings idx: {idx} OK !!!")
#             except Exception as err:
#                 logger.error(f"ERROR idx: {idx}: {err}")

#         # x = [e(x_cat[:,i]) for i, e in enumerate(self.embeddings)]
#         # x = torch.cat(x, 1)
#         logger.info("Embeddings generation OK !!!")
#         x = torch.cat(emb_input, 1)
#         logger.info(f"x 79 shape: {x.shape}")
#         x = self.emb_drop(x)
#         logger.info(f"x 81 shape: {x.shape}")
#         ## numerical
#         x2 = self.bn_num(x_cont)
#         logger.info(f"x 84 shape: {x.shape}")
#         x = torch.cat([x, x2], 1)
#         logger.info(f"x 100 shape: {x.shape}")

#         # first layer
#         x = self.lin1(x)
#         x = F.relu(x)
#         x = self.drop1(x)
#         x = self.bn1(x)
#         # second layer
#         x = F.relu(self.lin2(x))
#         x = self.drop2(x)
#         x = self.bn2(x)
#         # output
#         x = self.lin3(x)

#         return x

#     def training_step(self, batch, batch_idx):
#         x_cat, x_num, y = batch
#         y_hat = self(x_cat, x_num)
#         # https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html#torch.nn.functional.nll_loss
#         loss = F.nll_loss(y_hat, y)

#         # logs metrics for each training_step,
#         # and the average across the epoch, to the progress bar and logger
#         self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x_cat, x_num, y = batch
#         y_hat = self(x_cat, x_num)
#         # https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html#torch.nn.functional.nll_loss
#         loss = F.nll_loss(y_hat, y)
#         self.log("val_loss", loss)

#     def configure_optimizers(self):
#         """
#         Note:
#             L2 weights is added using , weight_decay=1e-5 function in the optimizer.
#         """
#         gen_opt = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=1e-5)
#         # dis_opt = torch.optim.Adam(self.parameters(), lr=0.02, weight_decay=1e-5)
#         # sequential optimizers
#         # return (
#         #     {'optimizer': dis_opt, 'frequency': 5},
#         #     {'optimizer': gen_opt, 'frequency': 1}
#         # )
#         return gen_opt

#     def configure_callbacks(self):
#         early_stop = EarlyStopping(monitor="val_acc", mode="max")
#         # 3. Init ModelCheckpoint callback, monitoring 'val_loss'
#         checkpoint = ModelCheckpoint(monitor="val_loss")

#         return [early_stop, checkpoint]


# SOME TOOLS
# Loading the training dataset. We need to split it into a training and validation part
# train_dataset = FashionMNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
# train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])

# # Loading the test set
# test_set = FashionMNIST(root=DATASET_PATH, train=False, transform=transform, download=True)
