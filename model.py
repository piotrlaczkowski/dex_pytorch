# based on: https://jovian.ai/aakanksha-ns/shelter-outcome
import probflow as pf
import torch


class ShelterOutcomeModel(nn.Module):
    """
    Core code can be found here:
    https://jovian.ai/aakanksha-ns/shelter-outcome
    """

    def __init__(self, embedding_sizes, n_cont):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings)  # length of all embeddings combined
        self.n_emb, self.n_cont = n_emb, n_cont
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 200)
        self.lin2 = nn.Linear(200, 70)
        self.lin3 = nn.Linear(70, 5)
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(200)
        self.bn3 = nn.BatchNorm1d(70)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)

    def emb_size(self, X):
        # categorical embedding for columns having more than two values
        emb_c = {n: len(col.cat.categories) for n, col in X.items() if len(col.cat.categories) > 2}
        emb_cols = emb_c.keys()  # names of columns chosen for embedding
        emb_szs = [(c, min(50, (c + 1) // 2)) for _, c in emb_c.items()]  # embedding sizes for the chosen columns
        return emb_szs

    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn1(x_cont)
        x = torch.cat([x, x2], 1)
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        x = self.bn3(x)
        x = self.lin3(x)
        return x


def train_model(model, optim, train_dl):
    model.train()
    total = 0
    sum_loss = 0
    for x1, x2, y in train_dl:
        batch = y.shape[0]
        output = model(x1, x2)
        loss = F.cross_entropy(output, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += batch
        sum_loss += batch * (loss.item())
    return sum_loss / total


def val_loss(model, valid_dl):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    for x1, x2, y in valid_dl:
        current_batch_size = y.shape[0]
        out = model(x1, x2)
        loss = F.cross_entropy(out, y)
        sum_loss += current_batch_size * (loss.item())
        total += current_batch_size
        pred = torch.max(out, 1)[1]
        correct += (pred == y).float().sum().item()
    print("valid loss %.3f and accuracy %.3f" % (sum_loss / total, correct / total))
    return sum_loss / total, correct / total


# PROBABILISTIC OUTPUT


class DensityNetwork(pf.ContinuousModel):
    def __init__(self, units, head_units):
        self.core = pf.DenseNetwork(units)
        self.mean = pf.DenseNetwork(head_units)
        self.std = pf.DenseNetwork(head_units)

    def __call__(self, x):
        x = torch.tensor(x)
        z = torch.nn.ReLU()(self.core(x))
        return pf.Normal(self.mean(z), torch.exp(self.std(z)))


# DEX v3. class


class DEX(nn.Module):
    def __init__(self):
        pass

    @staticmethod
    def emb_sz_rule(n_cat: int) -> int:
        """
        Embeddings size calculation from Fast.AI
        initial value 600 from Adriens script: `products embeddings.ipynb`
        """
        return min(500, (n_cat + 1) // 2, round(1.6 * n_cat ** 0.56))
