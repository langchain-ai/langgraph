import io
import json
import pathlib
import urllib.error
import urllib.request

import click
import pytest

from langgraph_cli.api_version import (
    _fetch_matching_version,
    resolve_langgraph_api_version,
)


@pytest.fixture()
def config_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path


def _write_config(
    config_dir: pathlib.Path, api_version: str | None = None
) -> pathlib.Path:
    cfg: dict = {"dependencies": ["."], "graphs": {"agent": "agent.py:graph"}}
    if api_version is not None:
        cfg["api_version"] = api_version
    path = config_dir / "langgraph.json"
    path.write_text(json.dumps(cfg))
    return path


class TestResolveLanggraphApiVersion:
    def test_exact_patch_from_cli(self, config_dir: pathlib.Path) -> None:
        path = _write_config(config_dir)
        assert resolve_langgraph_api_version(path, "0.7.67") == "0.7.67"

    def test_exact_patch_from_json(self, config_dir: pathlib.Path) -> None:
        path = _write_config(config_dir, api_version="0.7.67")
        assert resolve_langgraph_api_version(path, None) == "0.7.67"

    def test_neither_source_fetches_latest(
        self, config_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _write_config(config_dir)

        fake_body = json.dumps(
            {"results": [{"name": "latest"}, {"name": "0.9.2"}]}
        ).encode()

        def mock_urlopen(url, *, timeout=None):
            assert "name=" not in url
            return io.BytesIO(fake_body)

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
        assert resolve_langgraph_api_version(path, None) == "0.9.2"

    def test_both_sources_raises(self, config_dir: pathlib.Path) -> None:
        path = _write_config(config_dir, api_version="0.7.67")
        with pytest.raises(click.ClickException, match="both"):
            resolve_langgraph_api_version(path, "0.8.0")

    def test_partial_version_resolves_from_dockerhub(
        self, config_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _write_config(config_dir, api_version="0.7")

        fake_body = json.dumps(
            {"results": [{"name": "latest"}, {"name": "0.7.67"}]}
        ).encode()

        def mock_urlopen(url, *, timeout=None):
            assert "name=0.7" in url
            return io.BytesIO(fake_body)

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
        assert resolve_langgraph_api_version(path, None) == "0.7.67"

    def test_partial_cli_version_resolves_from_dockerhub(
        self, config_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _write_config(config_dir)

        fake_body = json.dumps(
            {"results": [{"name": "0.8.1"}, {"name": "0.8.0"}]}
        ).encode()

        def mock_urlopen(url, *, timeout=None):
            assert "name=0.8" in url
            return io.BytesIO(fake_body)

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
        assert resolve_langgraph_api_version(path, "0.8") == "0.8.1"

    def test_missing_config_file_fetches_latest(
        self, config_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = config_dir / "nonexistent.json"

        fake_body = json.dumps({"results": [{"name": "0.9.2"}]}).encode()

        def mock_urlopen(url, *, timeout=None):
            return io.BytesIO(fake_body)

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
        assert resolve_langgraph_api_version(path, None) == "0.9.2"

    def test_missing_config_file_with_cli_version(
        self, config_dir: pathlib.Path
    ) -> None:
        path = config_dir / "nonexistent.json"
        assert resolve_langgraph_api_version(path, "0.7.67") == "0.7.67"


class TestFetchMatchingVersion:
    def test_empty_prefix_returns_latest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_body = json.dumps(
            {"results": [{"name": "latest"}, {"name": "0.9.2"}, {"name": "abc123"}]}
        ).encode()

        def mock_urlopen(url, *, timeout=None):
            assert "name=" not in url
            return io.BytesIO(fake_body)

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
        assert _fetch_matching_version() == "0.9.2"

    def test_returns_first_semver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_body = json.dumps(
            {"results": [{"name": "latest"}, {"name": "abc1234"}, {"name": "0.7.67"}]}
        ).encode()

        def mock_urlopen(url, *, timeout=None):
            return io.BytesIO(fake_body)

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
        assert _fetch_matching_version("0.7") == "0.7.67"

    def test_no_semver_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_body = json.dumps(
            {"results": [{"name": "latest"}, {"name": "abc1234"}]}
        ).encode()

        def mock_urlopen(url, *, timeout=None):
            return io.BytesIO(fake_body)

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
        with pytest.raises(click.ClickException, match="Could not find a version"):
            _fetch_matching_version("0.7")

    def test_network_error_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def mock_urlopen(url, *, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
        with pytest.raises(click.ClickException, match="Failed to fetch"):
            _fetch_matching_version("0.7")
