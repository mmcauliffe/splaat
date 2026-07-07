from __future__ import annotations

import logging
import os
import pathlib
import typing

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

console = Console(
    theme=Theme(
        {
            "logging.level.debug": "cyan",
            "logging.level.info": "green",
            "logging.level.warning": "yellow",
            "logging.level.error": "red",
        }
    ),
    stderr=True,
)


def get_splaat_version() -> str:
    """
    Get the current splaat version

    Returns
    -------
    str
        splaat version
    """
    try:
        from ._version import version as __version__  # noqa
    except ImportError:
        __version__ = "0.1.0"
    return __version__


def get_temporary_directory() -> pathlib.Path:
    temp_directory = pathlib.Path(os.environ.get("SPLAAT_ROOT", os.path.expanduser("~/.splaat")))
    try:
        temp_directory.mkdir(exist_ok=True, parents=True)
    except Exception:
        pass
    return temp_directory


def get_next_primary_key(session, database_table):
    import sqlalchemy

    pk = session.query(sqlalchemy.func.max(database_table.id)).scalar()
    if not pk:
        pk = 0
    return pk + 1


def configure_logger(
    identifier: str,
    log_file: typing.Optional[pathlib.Path] = None,
    verbose: bool = False,
    quiet: bool = False,
) -> None:
    """
    Configure logging for the given identifier

    Parameters
    ----------
    identifier: str
        Logger identifier
    log_file: str
        Path to file to write all messages to
    verbose: bool, optional
        Whether to print debug level messages to the console, defaults to False
    quiet: bool, optional
        Whether to suppress console logging, defaults to False
    """
    logger = logging.getLogger(identifier)
    logger.setLevel(logging.DEBUG)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf8")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if not quiet:
        for handler in logger.handlers:
            if isinstance(handler, RichHandler):
                return
        handler = RichHandler(
            rich_tracebacks=True, log_time_format="", console=console, show_path=False
        )
        if verbose:
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
