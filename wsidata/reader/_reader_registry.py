import warnings
from collections.abc import MutableMapping
from html import escape
from typing import Dict, Iterator, Type

from .._utils import find_stack_level
from .base import ReaderBase


class ReaderRegistry(MutableMapping):
    priority = ["openslide", "tiffslide", "fastslide", "bioformats", "cucim"]

    def __init__(self):
        self._readers: Dict[str, type[ReaderBase]] = {}

    def __setitem__(self, key: str, value: type[ReaderBase]) -> None:
        assert issubclass(value, ReaderBase)
        self._readers[key] = value

    def __getitem__(self, key: str) -> type[ReaderBase]:
        reader = self._readers.get(key)
        if reader is None:
            raise KeyError(f"Cannot find reader '{key}' in registry.")
        return reader

    def __delitem__(self, key: str) -> None:
        del self._readers[key]

    def __iter__(self) -> Iterator[type[ReaderBase]]:
        return iter(self._readers)

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
            lines.append(f"{name}  ({'✓ Available' if available else '✗ Not Install'})")

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
                f"<td style='text-align:left'>{'✓ Available' if available else '✗ Not Install'}</td></tr>"
            )
        rows.append("</tbody>\n</table>")
        return "\n".join(rows)

    def try_open(self, img_path: str, reader: str = None) -> ReaderBase:
        if reader is not None:
            reader = self[reader]
            reader.is_available(raise_error=True)
            return reader(img_path)
        for reader in self.priority:
            reader = self[reader]
            if reader.is_available():
                try:
                    return reader(img_path)
                except Exception:  # noqa
                    continue
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
