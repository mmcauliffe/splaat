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
