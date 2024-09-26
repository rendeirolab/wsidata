from enum import Enum

from .bioformats import BioFormatsReader
from .openslide import OpenSlideReader
from .cucim import CuCIMReader
from .tiffslide import TiffSlideReader


class Reader(Enum):
    OPENSLIDE = "openslide"
    CUCIM = "cucim"
    TIFFSLIDE = "tiffslide"
    BIOFORMATS = "bioformats"


InstallationTips = {
    Reader.OPENSLIDE: "Please install OpenSlide with conda/mamba, `conda install openslide-python`",
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


def get_reader(reader: str = None, format: str = None):
    """
    Get the reader based on the input format and reader name.

    Parameters
    ----------
    reader : str
        Name of the reader to use. If None, will try to find a suitable reader.
    format : str
        Format of the image file. If None, will try to find a suitable reader.

    Returns
    -------
    reader : ReaderBase
        Reader object to read the image file.

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
                f"Requested reader not available, "
                f"must be one of {reader_candidates}"
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
