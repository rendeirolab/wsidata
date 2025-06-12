from .._model import WSIData


def graph_data(
    wsi: WSIData,
    feature_key: str,
    tile_key: str = "tiles",
    graph_key: str = None,
    target_key: str = None,
):
    """
    Create a PyTorch Geometric Data object from the graph data in WSIData.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object containing the graph data.
    feature_key : str
        The key for the tile features.
    tile_key : str, default: "tiles"
        The key for the tiles.
    graph_key : str, optional
        The key for tile graph, defaults to "{tile_key}_graph".
    target_key : str, optional
        The key for the target data in the observation table.

    Returns
    -------
    :class:`torch_geometric.data.Data`
        A PyTorch Geometric Data object containing:
        - x: Node features (image features)
        - edge_index: Graph connectivity
        - edge_attr: Edge attributes (distances)
        - y: Target values (if target_key is provided)

    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError(
            "Please install torch_geometric to use this function. "
            "You can install it via 'pip install torch_geometric'."
        )
    import numpy as np
    import scipy.sparse as sp
    import torch

    # Get the AnnData object containing the graph data
    if graph_key is None:
        graph_key = f"{tile_key}_graph"

    feature_key = wsi._check_feature_key(feature_key, tile_key)
    graph_tables = wsi.tables[graph_key]
    features = wsi.tables[feature_key]

    # Extract node features (image features)
    x = torch.tensor(features.X, dtype=torch.float)

    # Extract graph structure from obsp
    if (
        "spatial_connectivities" in graph_tables.obsp
        and "spatial_distances" in graph_tables.obsp
    ):
        # Get connectivity matrix (adjacency matrix)
        conn_matrix = graph_tables.obsp["spatial_connectivities"]

        # Get distance matrix
        dist_matrix = graph_tables.obsp["spatial_distances"]

        # Convert sparse matrices to edge_index and edge_attr
        if sp.issparse(conn_matrix):
            # Get indices of non-zero elements (edges)
            edges = sp.find(conn_matrix)
            edge_index = torch.tensor(np.vstack((edges[0], edges[1])), dtype=torch.long)

            # Get corresponding distances as edge attributes
            if sp.issparse(dist_matrix):
                # Extract distances for the same edges
                edge_attr = torch.tensor(
                    dist_matrix[edges[0], edges[1]].A1, dtype=torch.float
                ).view(-1, 1)
            else:
                # If dist_matrix is dense
                edge_attr = torch.tensor(
                    dist_matrix[edges[0], edges[1]], dtype=torch.float
                ).view(-1, 1)
        else:
            # If conn_matrix is dense
            edge_index = torch.tensor(
                np.array(np.where(conn_matrix > 0)), dtype=torch.long
            )
            edge_attr = torch.tensor(
                dist_matrix[edge_index[0], edge_index[1]], dtype=torch.float
            ).view(-1, 1)
    else:
        # If graph structure is not available, create an empty graph
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)

    # Extract targets if available
    y = None
    if target_key is not None:
        tiles = wsi.shapes[tile_key]
        if target_key in tiles:
            y = torch.tensor(tiles[target_key])

    # Create and return PyG Data object
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
