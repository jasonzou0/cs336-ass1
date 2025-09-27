try:
    import importlib.metadata
    __version__ = importlib.metadata.version("cs336_basics")
except (importlib.metadata.PackageNotFoundError, ImportError):
    __version__ = "1.0.5"  # fallback version
