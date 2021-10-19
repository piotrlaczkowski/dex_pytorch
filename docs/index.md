## Welcome to DEX 2.0 (Deep Learning Expert) documentations

#### Base model architecture for this branch:

![alt text](../CORE/model.png "Dex Architecture")

### Branches

For the development of different ideas we make use of different branches.

- `master` -> base functional model at the actual state of development.

#### CI/CD

We use GitHub actions to generate CI pipelines. Actually we have the following pipelines:

```
- pushing master branch changes to all branches automatically
```

### Inspecting the model

```bash
saved_model_cli show --dir /tmp/saved_model_dir --all
```

## Data Preparation

For the data cleaning part it is important to have:

- values >= 0 (not negative, as sometimes id_model can be)
- no NaNs
- no Infs
- dtypes corresponding to the ones in the TF

```python
df.columns = df.columns.str.lower()
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df.drop_duplicates([config.INDEX_COL_NAME])
df = df.loc[df["id_model"] >= 0,: ]
df = df.loc[df["price"] > 15,: ]
df = df.astype(config.NP_DTYPES).astype({"price": np.int32})
```
