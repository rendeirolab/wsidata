API Reference
=============


IO and Utilities Functions
--------------------------

.. currentmodule:: wsidata
.. autosummary::
    :toctree: _autogen
    :nosignatures:

    open_wsi
    agg_wsi
    get_reader

WSIData
-------

.. currentmodule:: wsidata
.. autosummary::
    :toctree: _autogen
    :template: wsidata
    :nosignatures:

    WSIData
    TileSpec

Accessors
---------

.. currentmodule:: wsidata
.. autosummary::
    :toctree: _autogen
    :template: autosummary
    :nosignatures:

    register_wsidata_accessor
    FetchAccessor
    IterAccessor
    DatasetAccessor

Datasets
--------

.. currentmodule:: wsidata.dataset
.. autosummary::
    :toctree: _autogen
    :template: autosummary
    :nosignatures:

    TileImagesDataset

Readers
-------

.. currentmodule:: wsidata.reader
.. autosummary::
    :toctree: _autogen
    :template: autosummary
    :nosignatures:

    ReaderBase
    OpenSlideReader
    TiffSlideReader
    BioFormatsReader
    SlideProperties
