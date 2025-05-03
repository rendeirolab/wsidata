from __future__ import annotations

from enum import Enum
from os import PathLike
from typing import Dict

from . import ReaderBase
from .bioformats import BioFormatsReader
from .cucim import CuCIMReader
from .openslide import OpenSlideReader
from .tiffslide import TiffSlideReader


class Reader(Enum):
    OPENSLIDE = "openslide"
    TIFFSLIDE = "tiffslide"
    BIOFORMATS = "bioformats"
    CUCIM = "cucim"


InstallationTips = {
    Reader.OPENSLIDE: "`pip install openslide-python openslide-bin`",
    Reader.CUCIM: "Please see `https://github.com/rapidsai/cucim` for installation instructions",
    Reader.TIFFSLIDE: "`pip install tiffslide`",
    Reader.BIOFORMATS: "`pip install scyjava`",
}


OPENSLIDE_FORMATS = {
    "svs",
    "dcm",
    "ndpi",
    "vms",
    "vmu",
    "scn",
    "mrxs",
    "svslide",
    "bif",
    "tif",
    "tiff",
}

TIFFSLIDE_FORMATS = {
    "svs",
    "ndpi",
    "vms",
    "vmu",
    "scn",
    "tif",
    "tiff",
}

CUCIM_FORMATS = {
    "svs",
    "tiff",
}

# Bio-Formats is the last resort for all formats


def _try_readers() -> (Dict[str, ReaderBase], Dict[str, Exception]):
    """
    Attempt to get current available readers.

    Returns
    -------
    readers : dict
        Dictionary of available readers.
    error_stack : dict
        Dictionary of errors encountered while trying to import readers.

    """
    readers = {r: None for r in Reader}
    error_stack = {r: None for r in Reader}
    catch_error = (ModuleNotFoundError, OSError, ImportError)

    try:
        import openslide

        readers[Reader.OPENSLIDE] = OpenSlideReader
    except catch_error as e:
        error_stack[Reader.OPENSLIDE] = e

    try:
        import tiffslide

        readers[Reader.TIFFSLIDE] = TiffSlideReader
    except catch_error as e:
        error_stack[Reader.TIFFSLIDE] = e

    try:
        import scyjava

        readers[Reader.BIOFORMATS] = BioFormatsReader
    except catch_error as e:
        error_stack[Reader.BIOFORMATS] = e

    try:
        import cucim

        readers[Reader.CUCIM] = CuCIMReader
    except catch_error as e:
        error_stack[Reader.CUCIM] = e
    return readers, error_stack


def get_available_readers() -> Dict[str, ReaderBase]:
    """
    Get a list of available readers.

    Returns
    -------
    readers : list
        List of available readers.

    """
    readers, error_stack = _try_readers()
    return {k: v for k, v in readers.items() if v is not None}


def try_reader(img_path: PathLike, reader=None) -> ReaderBase | None:
    """
    Try to get the reader based on the input image path.

    Parameters
    ----------
    img_path : path or str
        Path to the image file.
    reader : optional
        The reader to use. If None, will try to find a suitable reader.

    Returns
    -------
    reader : ReaderBase
        Reader object to read the image file.

    """
    readers, error_stack = _try_readers()
    if reader is not None:
        reader = Reader(reader)
        if reader not in readers:
            raise ValueError(
                f"Requested reader not available, must be one of {readers.keys()}"
            )
        ReaderObj = readers.get(reader)
        if ReaderObj is None:
            raise ValueError(
                f"Requested reader not available: {reader}, if you don't have it installed, "
                f"{InstallationTips[reader]}\n"
                f"Following error occurred when loading {reader}: {error_stack[reader]}"
            )
        return ReaderObj(img_path)

    available_readers = {k: v for k, v in readers.items() if v is not None}

    if len(available_readers) == 0:
        msg = "None of the readers are available:"
        for name, err_msg in error_stack.items():
            if err_msg is not None:
                msg += f"\n{name}: {err_msg}"
        raise ValueError(msg)

    for name, ReaderObj in available_readers.items():
        try:
            return ReaderObj(img_path)
        except Exception as _:
            pass
    raise ValueError(
        f"None of the readers were able to read the image file: {img_path}"
    )


def get_reader(reader: str = None, format: str = None) -> Reader:
    """
    Get the reader based on the input format and reader name.

    Parameters
    ----------
    reader : str
        Name of the reader to use. If None, will try to find a suitable reader.
    format : str
        Format of the image file. If None, will try to find a suitable reader.
    img_path : path or str
        Path to the image file. If None, will try to find a suitable reader.

    Returns
    -------
    reader : ReaderBase
        Reader object to read the image file.

    """
    readers, error_stack = _try_readers()

    reader_candidates = []
    if format is not None and reader is None:
        format = format.strip().lstrip(".").lower()
        if format in OPENSLIDE_FORMATS:
            reader_candidates.append(Reader.OPENSLIDE)
        if format in TIFFSLIDE_FORMATS:
            reader_candidates.append(Reader.TIFFSLIDE)
        reader_candidates.append(Reader.BIOFORMATS)
        if format in CUCIM_FORMATS:
            reader_candidates.append(Reader.CUCIM)
    else:
        reader_candidates = list(Reader)

    if reader is None:
        for i in reader_candidates:
            reader = readers.get(i)
            if reader is not None:
                return reader
        msg = "None of the readers are available:"
        for i in reader_candidates:
            msg += f"\n{i}: {error_stack[i]}"
        raise ValueError(msg)
    else:
        reader = Reader(reader)
        if reader not in reader_candidates:
            raise ValueError(
                f"Requested reader not available, must be one of {reader_candidates}"
            )
        else:
            used_reader = readers.get(reader)
            if used_reader is None:
                raise ValueError(
                    f"Requested reader not available: {reader}, if you don't have it installed, "
                    f"{InstallationTips[reader]}\n"
                    f"Following error occurred: {error_stack[reader]}"
                )
            return used_reader
