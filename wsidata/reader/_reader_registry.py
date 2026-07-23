import warnings
from collections.abc import MutableMapping
from html import escape
from pathlib import Path
from typing import Dict, Iterator, List, Type

from .._utils import find_stack_level
from .base import ReaderBase, SceneSelectionError


class ReaderRegistry(MutableMapping):
    # ``pylibczi`` is deliberately placed ahead of ``bioformats`` because
    # it is CZI-only: pyczi.open_czi raises on any non-CZI input, so
    # ``try_open`` falls straight through to BioFormats for other formats.
    # Placing it after BioFormats would mean a ``.czi`` file auto-detects
    # to BioFormats on arm64 macOS, where the required JPEG-XR native lib
    # is unavailable, and the user would only see the failure at first
    # read.
    priority = [
        "openslide",
        "tiffslide",
        "fastslide",
        "pylibczi",
        "bioformats",
        "cucim",
        "isyntax",
    ]

    def __init__(self):
        self._readers: Dict[str, type[ReaderBase]] = {}
        self._ext_index: Dict[str, List[str]] | None = None

    def __setitem__(self, key: str, value: type[ReaderBase]) -> None:
        assert issubclass(value, ReaderBase)
        self._readers[key] = value
        self._ext_index = None  # invalidate extension index

    def __getitem__(self, key: str) -> type[ReaderBase]:
        reader = self._readers.get(key)
        if reader is None:
            raise KeyError(f"Cannot find reader '{key}' in registry.")
        return reader

    def __delitem__(self, key: str) -> None:
        del self._readers[key]
        self._ext_index = None  # invalidate extension index

    def __iter__(self) -> Iterator[type[ReaderBase]]:
        return iter(self._readers.values())

    def __len__(self) -> int:
        return len(self._readers)

    def __contains__(self, key: str) -> bool:
        return key in self._readers

    def _repr_reader_order(self):
        repr_order = []
        for name, reader_cls in self._readers.items():
            repr_order.append((name, reader_cls.is_available()))
        # Sort by availability
        repr_order.sort(key=lambda x: x[1], reverse=True)
        return repr_order

    def __repr__(self) -> str:
        lines = []
        for name, available in self._repr_reader_order():
            lines.append(
                f"{name}  ({'✓ Available' if available else '✗ Not Installed'})"
            )

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        # HTML representation suitable for Jupyter rich display
        rows = [
            "<table>\n<thead>\n<tr><th style='text-align:left'>Name</th>"
            "<th style='text-align:left'>Availability</th></tr>\n</thead>",
            "<tbody>",
        ]
        for name, available in self._repr_reader_order():
            rows.append(
                f"<tr><td style='text-align:left'>{escape(str(name))}</td>"
                f"<td style='text-align:left'>{'✓ Available' if available else '✗ Not Installed'}</td></tr>"
            )
        rows.append("</tbody>\n</table>")
        return "\n".join(rows)

    def _build_ext_index(self):
        """Build extension → reader-names index, sorted by priority."""
        index: Dict[str, List[str]] = {}
        for name, reader_cls in self._readers.items():
            if not reader_cls.extensions:  # skip None and ()
                continue
            for ext in reader_cls.extensions:
                ext = ext.lower()
                if ext not in index:
                    index[ext] = []
                index[ext].append(name)
        # Sort each list by priority position
        priority_rank = {name: i for i, name in enumerate(self.priority)}
        for ext in index:
            index[ext].sort(key=lambda n: priority_rank.get(n, len(self.priority)))
        self._ext_index = index

    @staticmethod
    def _get_extension(img_path) -> str:
        """Extract lowercase file extension."""
        p = Path(str(img_path))
        suffixes = p.suffixes
        if not suffixes:
            return ""
        if len(suffixes) > 1:
            if suffixes[-2].lower() == ".ome":
                return f".ome{suffixes[-1]}".lower()
        return suffixes[-1].lower()

    @staticmethod
    def _open_reader(reader_cls, img_path, scene):
        if scene is not None and not reader_cls.supports_scenes:
            if scene != 0:
                raise SceneSelectionError(
                    f"Reader '{reader_cls.name}' does not support scene selection."
                )
            return reader_cls(img_path)
        kwargs = {"scene": scene} if reader_cls.supports_scenes else {}
        return reader_cls(img_path, **kwargs)

    def try_open(
        self, img_path: str, reader: str = None, scene: int | None = None
    ) -> ReaderBase:
        if reader is not None:
            reader_cls = self[reader]
            reader_cls.is_available(raise_error=True)
            return self._open_reader(reader_cls, img_path, scene)

        # Prefer extension matches, then fall back to the global priority.
        candidates = []
        ext = self._get_extension(img_path)
        if ext:
            if self._ext_index is None:
                self._build_ext_index()
            candidates.extend(self._ext_index.get(ext, []))
        for name in self.priority:
            if name in self._readers and name not in candidates:
                candidates.append(name)

        # An explicit scene asks for scene semantics, so capable readers get
        # the first opportunity even if a single-image reader has higher priority.
        if scene is not None:
            candidates.sort(key=lambda name: not self[name].supports_scenes)

        scene_error = None
        for name in candidates:
            reader_cls = self[name]
            if not reader_cls.is_available():
                continue
            if scene not in (None, 0) and not reader_cls.supports_scenes:
                continue
            try:
                return self._open_reader(reader_cls, img_path, scene)
            except SceneSelectionError as error:
                scene_error = error
            except Exception:  # noqa: BLE001
                continue

        if scene_error is not None:
            raise scene_error

        raise ValueError(f"Cannot open image '{img_path}' using any of the readers.")


# Global instance
READERS = ReaderRegistry()


def register(
    name: str,
):
    def decorator(reader_cls: Type[ReaderBase]):
        if name in READERS:
            warnings.warn(
                f"{name} is already registered.", stacklevel=find_stack_level()
            )
        READERS[name] = reader_cls
        return reader_cls

    return decorator
