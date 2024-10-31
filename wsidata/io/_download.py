from pathlib import Path

import fsspec


def _create_progress(pbar=True, known_length=True):
    from rich.progress import (
        Progress,
        BarColumn,
        TextColumn,
        DownloadColumn,
        TransferSpeedColumn,
        TimeRemainingColumn,
        SpinnerColumn,
    )

    if known_length:
        return Progress(
            TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
            BarColumn(bar_width=20),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
            disable=not pbar,
        )
    else:
        return Progress(
            TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
            SpinnerColumn(),
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            disable=not pbar,
        )


class CacheDownloader:
    """A downloader with simple cache mechanism.

    This class can download from arbitrary URLs and cache the downloaded file.

    The hash of the downloaded file is stored in a hidden file
    in the same directory of the downloaded file.

    Parameters
    ----------
    url : str
        URL to download.
    name : str, optional
        Name of the file, by default None.
    cache_dir : str, optional
        Cache directory, by default None.

    """

    hash_method = "sha256"

    def __init__(self, url, name=None, cache_dir=None):
        self.url = url
        with fsspec.open(self.url, "rb") as fsrc:
            self.total_size = fsrc.size  # Retrieve the total file size
            if name is None:
                if hasattr(fsrc, "name"):
                    name = fsrc.name
        if name is None:
            name = Path(url).name
        if cache_dir is None:
            cache_dir = "."  # Default to current directory
        cache_dir = Path(cache_dir)
        self.name = name
        self.dest = cache_dir / name
        self.hash_path = cache_dir / f".{name}.{self.hash_method}"

    def is_cache(self):
        if self.dest.exists() & self.hash_path.exists():
            with open(self.hash_path, "r") as f:
                last_file_hash = f.read()
            with open(self.dest, "rb") as f:
                current_file_hash = self._hash_file(f)
            return last_file_hash == current_file_hash
        return False

    def _hash_file(self, fileobj):
        import hashlib

        if hasattr(hashlib, "file_digest"):
            digest = hashlib.file_digest(fileobj, self.hash_method)
            hex_digest = digest.hexdigest()
        else:
            h = getattr(hashlib, self.hash_method)()
            b = bytearray(128 * 1024)
            mv = memoryview(b)
            while n := fileobj.readinto(mv):
                h.update(mv[:n])
            hex_digest = h.hexdigest()

        return hex_digest

    def download(self, pbar=True):
        """Download a single file with progress tracking."""
        if self.is_cache():
            return self.dest
        else:
            known_length = self.total_size is not None
            progress = _create_progress(pbar, known_length)
            with progress:
                with fsspec.open(self.url, "rb") as fsrc:
                    task_id = progress.add_task(
                        "Download", filename=self.dest, total=self.total_size
                    )
                    with fsspec.open(f"{self.dest}", "wb") as fdst:
                        progress.start_task(task_id)
                        chunk_size = 1024 * 1024  # 1 MB
                        while chunk := fsrc.read(chunk_size):
                            fdst.write(chunk)
                            progress.advance(task_id, len(chunk))
                progress.refresh()
                # Create a hash file
                with open(self.hash_path, "w") as f:
                    with open(self.dest, "rb") as fdst:
                        f.write(self._hash_file(fdst))
            return self.dest
