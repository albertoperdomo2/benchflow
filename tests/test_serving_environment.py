import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "serving_check",
    Path(__file__).resolve().parents[1] / "build/check_serving_environment.py",
)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def distribution(name, version, requires=()):
    return SimpleNamespace(metadata={"Name": name}, version=version, requires=requires)


def test_explicit_upstream_nccl_override_is_honored():
    packages = [
        distribution("torch", "2.13.0", ["nvidia-nccl-cu13==2.29.7"]),
        distribution("nvidia-nccl-cu13", "2.30.7"),
    ]
    assert not checker.dependency_errors(packages, ["nvidia-nccl-cu13==2.30.7"])
    assert checker.dependency_errors(packages, [])


def test_nccl_override_cannot_hide_telemetry_conflict():
    packages = [
        distribution("opentelemetry-sdk", "1.40.0"),
        distribution(
            "opentelemetry-exporter-otlp-common",
            "0.66b1",
            ["opentelemetry-sdk~=1.45.1"],
        ),
    ]
    assert checker.dependency_errors(packages, ["nvidia-nccl-cu13==2.30.7"])


def test_wrong_installed_nccl_still_fails():
    packages = [
        distribution("torch", "2.13.0", ["nvidia-nccl-cu13==2.29.7"]),
        distribution("nvidia-nccl-cu13", "2.29.7"),
    ]
    assert checker.dependency_errors(packages, ["nvidia-nccl-cu13==2.30.7"])


def test_missing_dependency_still_fails():
    assert checker.dependency_errors([distribution("test", "1.0", ["missing>=1"])], [])


def test_unrelated_override_is_rejected():
    with pytest.raises(ValueError, match="Unexpected"):
        checker.dependency_errors([], ["opentelemetry-sdk==1.40.0"])


@pytest.mark.parametrize(
    "versions,valid", [(("2.10.1", "2.7.0"), True), (("2.7.0", "2.10.1"), False)]
)
def test_system_package_duplicates_follow_python_search_order(
    tmp_path, versions, valid
):
    from importlib import metadata

    paths = []
    for index, version in enumerate(versions):
        root = tmp_path / str(index)
        info = root / f"PyJWT-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: PyJWT\nVersion: {version}\n"
        )
        paths.append(str(root))
    packages = [
        distribution("mcp", "2.3.0", ["pyjwt[crypto]>=2.10.1"]),
        *metadata.distributions(path=paths),
    ]
    assert bool(checker.dependency_errors(packages, [])) is not valid
