"""Smoke tests: the package imports and exposes a version."""


def test_import_version() -> None:
    import cfd10

    assert cfd10.__version__
