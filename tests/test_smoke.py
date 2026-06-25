import gradlint


def test_package_version_available() -> None:
    assert gradlint.__version__


def test_core_version_matches_package() -> None:
    assert gradlint.core_version() == gradlint.__version__
