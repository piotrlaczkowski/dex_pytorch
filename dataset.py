import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config


class DexDataset(Dataset):
    """

    DEX dataset.

    Example:

            # Load dataset
            dataset = DexDataset(file_path)

            # Split into training and test
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            trainset, testset = random_split(dataset, [train_size, test_size])

            # Dataloaders
            trainloader = DataLoader(trainset, batch_size=200, shuffle=True)
            testloader = DataLoader(testset, batch_size=200, shuffle=False)
    """

    def __init__(self, file_path: str):
        """Initializes instance of class StudentsPerformanceDataset.
        Args:
            csv_file (str): Path to the csv file with the students data.
        """
        self.df = pd.read_csv(file_path)
        self.df.columns = self.df.columns.str.lower()

        # Selecting columns
        self.cat = list(config.CAT_COLS.keys())
        self.num = list(config.NUM_COLS.keys())
        self.target = list(config.TARGET_COL.keys())[0]
        self.col_names = self.cat + self.num + [self.target]

        # filtering out other columns and fixing order
        self.df = self.df[self.col_names]

        # Save target and predictors
        self.X_cat = self.df[self.cat].copy().astype(config.CAT_COLS).values
        self.X_num = self.df[self.num].copy().astype(config.NUM_COLS).values
        self.y = self.df[self.target].astype(config.TARGET_COL).values

    def emb_sz_rule(self, n_cat: int) -> int:
        """
        Embeddings size calculation from Fast.AI
        initial value 600 from Adriens script: `products embeddings.ipynb`
        """
        return min(500, (n_cat + 1) // 2, round(1.6 * n_cat ** 0.56))

    def get_embedding_sizes(self):

        # getting all for the categoricals
        df_cat = self.df[self.cat]
        embedding_sizes = []
        for col_name in df_cat.columns:
            nr_unique_items = len(df_cat[col_name].unique())
            emb_size = self.emb_sz_rule(n_cat=nr_unique_items)
            embedding_sizes.append((nr_unique_items, emb_size))
        # getting all for the numericals
        nr_num = len(self.num)
        return embedding_sizes, nr_num

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        # return self.X_cat.iloc[idx].values, self.X_num.iloc[idx].values, self.y[idx]
        return self.X_cat[idx], self.X_num[idx], self.y[idx]
        # return self.transform(self.X_cat[idx]), self.transform(self.X_num[idx]), self.transform(self.y[idx])


class ShelterOutcomeDataset(Dataset):
    """
    Splitting categorial and numerical columns

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, X, Y, embedded_col_names):
        X = X.copy()
        self.X1 = X.loc[:, embedded_col_names].copy().values.astype(np.int64)  # categorical columns
        self.X2 = X.drop(columns=embedded_col_names).copy().values.astype(np.float32)  # numerical columns
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]
