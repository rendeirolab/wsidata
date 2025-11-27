WSI Readers
-----------

All readers are registered under the :code:`READERS` space.

.. code-block:: python

    from wsidata import READERS

To open a slide with any available reader:

.. code-block:: python

    READERS.try_open("sample.svs") # Select a good one
    READERS.try_open("sample.svs", reader="bioformats")  # Only try bioformats

To retrieve a reader class

.. code-block:: python

    ReaderClass = READERS['openslide']
    reader = ReaderClass('sample.svs')


.. currentmodule:: wsidata.reader
.. autosummary::
    :toctree: _autogen
    :template: autosummary
    :nosignatures:

    ReaderBase
    OpenSlideReader
    TiffSlideReader
    BioFormatsReader
    SpatialDataImage2DReader
