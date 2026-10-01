"""Step 0a smoke tests: environment + dependency verification (PRD §10)."""
import importlib


def test_core_imports():
    for module in [
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "matplotlib",
        "dateutil",
        "requests",
        "yaml",
        "pytest",
        "dotenv",
        "streamlit",
        "acnportal",
        "torch",
    ]:
        importlib.import_module(module)


def test_torch_cpu_works():
    import torch

    assert torch.tensor([1.0, 2.0]).sum().item() == 3.0


def test_acnportal_dataclient_signature():
    from acnportal.acndata import DataClient
    import inspect

    sig = inspect.signature(DataClient.__init__)
    assert "api_token" in sig.parameters, "DataClient must accept api_token"


def test_acnportal_timeseries_flag_exists():
    # AT-1a groundwork: the documented timeseries parameter must exist on the
    # generator methods (PRD §3.2).
    from acnportal.acndata import DataClient
    import inspect

    for method in ("get_sessions", "get_sessions_by_time"):
        sig = inspect.signature(getattr(DataClient, method))
        assert "timeseries" in sig.parameters, f"{method} lacks timeseries param"
