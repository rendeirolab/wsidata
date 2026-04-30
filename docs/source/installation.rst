Installation
============

You can install :code:`wsidata` with whatever package manager you prefer.
The default installation only ship with :code:`tiffslide`, please refer to next section on how to install
different slide readers to support variety of file formats.

.. tab-set::

    .. tab-item:: PyPI

        The default installation.

        .. code-block:: bash

            pip install wsidata

    .. tab-item:: uv

        .. code-block:: bash

            uv add wsidata

    .. tab-item:: Conda

        .. code-block:: bash

            conda install conda-forge::wsidata

    .. tab-item:: Development

        If you want to install the latest version from the GitHub repository, you can use the following command:

        .. code-block:: bash

            pip install git+https://github.com/rendeirolab/wsidata.git


Installation for slide readers
------------------------------


.. tab-set::

    .. tab-item:: TiffSlide

        `TiffSlide <https://github.com/Bayer-Group/tiffslide>`_ is a cloud native openslide-python replacement
        based on tifffile.

        TiffSlide is installed by default. You don't need to install it manually.

        .. code-block:: bash

            pip install tiffslide

    .. tab-item:: OpenSlide

        `OpenSlide <https://openslide.org/>`_ is a C library that provides a simple interface to read whole-slide images.

        OpenSlide is installed by default from v0.3.0. You don't need to install it manually.

        You can easily install from PyPI

        .. code-block:: bash

            pip install openslide-python openslide-bin

        In case that PyPI doesn't work for you:

        For Linux and OSX users, it's suggested that you install :code:`openslide` with conda or mamba:

        .. code-block:: bash

            conda install -c conda-forge openslide-python
            # or
            mamba install openslide-python


        For Windows users, you need to download compiled :code:`openslide` from
        `GitHub Release <https://github.com/openslide/openslide-bin/releases>`_.
        If you open the folder, you should find a :code:`bin` folder.

        Make sure you point the :code:`bin` folder for python to locate the :code:`openslide` binary.
        You need to run following code to import the :code:`openslide`,
        it's suggested to run this code before everything:

        .. code-block:: python

            import os
            with os.add_dll_directory("path/to/openslide/bin")):
                import openslide

    .. tab-item:: BioFormats

        `BioFormats <https://www.openmicroscopy.org/bio-formats/>`_ is a standalone Java library
        for reading and writing life sciences image file formats.

        `scyjava <https://github.com/scijava/scyjava>`_ is used to interact with the BioFormats library.

        .. code-block:: bash

            pip install scyjava

    .. tab-item:: CuCIM

        `CuCIM <https://github.com/rapidsai/cucim>`_ is a GPU-accelerated image I/O library.

        Please refer to the `CuCIM GitHub <https://github.com/rapidsai/cucim>`_.

    .. tab-item:: pyisyntax

        `pyisyntax <https://github.com/anibali/pyisyntax>`_ a Python library for working with
        pathology images in the iSyntax file format, powered by `libisyntax <https://github.com/amspath/libisyntax>`_.

        .. code-block:: bash

            pip install pyisyntax

    .. tab-item:: pylibCZIrw

        `pylibCZIrw <https://github.com/ZEISS/pylibczirw>`_ is the official
        Python binding to `libCZI <https://github.com/ZEISS/libczi>`_,
        maintained by Zeiss, for reading and writing Zeiss CZI whole-slide
        images. Unlike BioFormats, it decodes JPEG-XR natively on every
        platform it ships wheels for - including ``arm64`` macOS, where
        BioFormats cannot load the ``ome:jxrlib`` native library. It is
        therefore the recommended backend for ``.czi`` files on Apple
        Silicon.

        .. code-block:: bash

            pip install pylibCZIrw

        Wheels are published for Linux (``x86_64`` and ``aarch64``),
        macOS (``arm64``), and Windows (``x86_64``) on CPython 3.9-3.13.
        There is no macOS ``x86_64`` wheel, so Intel Macs will need to
        build from source or use BioFormats.

