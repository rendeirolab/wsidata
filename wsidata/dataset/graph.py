from torch.utils.data import Dataset

from .._model import WSIData


def graph_data(wsi: WSIData, key: str = "graph", target_key: str = None):
    tables = wsi.tables[key]
    X = tables.X
    tables = tables.obs
    targets = None
    if target_key in tables:
        targets = tables[target_key].to_numpy()
    return X, targets
