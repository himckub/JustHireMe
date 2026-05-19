from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def test_vector_store_uses_jhm_app_data_dir(monkeypatch, tmp_path):
    calls: list[str] = []

    fake_lancedb = types.SimpleNamespace(
        connect=lambda path: calls.append(path) or types.SimpleNamespace()
    )
    monkeypatch.setitem(sys.modules, "lancedb", fake_lancedb)
    monkeypatch.setenv("JHM_APP_DATA_DIR", str(tmp_path / "roaming-app-data"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local-app-data"))

    from data.vector import connection

    module = importlib.reload(connection)

    expected_base = tmp_path / "roaming-app-data" / "JustHireMe"
    expected_vector = expected_base / "vector"
    assert Path(module.BASE_DIR) == expected_base
    assert Path(module.VECTOR_DIR) == expected_vector
    assert Path(calls[-1]) == expected_vector


def test_vector_store_falls_back_to_local_app_data(monkeypatch, tmp_path):
    calls: list[str] = []

    fake_lancedb = types.SimpleNamespace(
        connect=lambda path: calls.append(path) or types.SimpleNamespace()
    )
    monkeypatch.setitem(sys.modules, "lancedb", fake_lancedb)
    monkeypatch.delenv("JHM_APP_DATA_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local-app-data"))

    from data.vector import connection

    module = importlib.reload(connection)

    expected_vector = tmp_path / "local-app-data" / "JustHireMe" / "vector"
    assert Path(module.VECTOR_DIR) == expected_vector
    assert Path(calls[-1]) == expected_vector


def test_null_vector_store_reports_disabled_reason():
    from data.vector.connection import NullVectorStore

    vec = NullVectorStore("LanceDB not bundled")

    assert vec.available is False
    assert vec.reason == "LanceDB not bundled"
    assert vec.list_tables() == []


def test_vector_status_reports_disabled_for_null_store(monkeypatch):
    from data.vector import connection

    monkeypatch.setattr(connection, "vec", connection.NullVectorStore("LanceDB not bundled"))
    monkeypatch.setattr(connection, "vector_runtime_ready", lambda: False)

    status = connection.vector_status()

    assert status["status"] == "disabled"
    assert status["error"] == "LanceDB not bundled"


def test_graph_vector_sync_skips_when_store_is_disabled(monkeypatch):
    from data.graph import profile as graph_profile
    from data.vector import connection as vector_connection
    from data.vector.connection import NullVectorStore

    monkeypatch.setattr(vector_connection, "vec", NullVectorStore("LanceDB not bundled"))
    monkeypatch.setattr(
        graph_profile,
        "execute_query",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("graph should not be queried")),
    )

    status = graph_profile.sync_vectors_from_graph()

    assert status == {"status": "disabled", "synced": 0, "error": "LanceDB not bundled"}


def test_vector_runtime_asset_name_is_platform_specific(monkeypatch):
    from data.vector import runtime

    monkeypatch.setattr(runtime, "sys_platform", lambda: "windows")
    assert runtime.vector_runtime_asset_name() == "JustHireMe-vector-runtime-windows.zip"
    assert runtime.runtime_pack_asset_name() == "JustHireMe-runtime-pack-windows.zip"

    monkeypatch.setattr(runtime, "sys_platform", lambda: "darwin")
    assert runtime.vector_runtime_asset_name() == "JustHireMe-vector-runtime-macos.zip"
    assert runtime.runtime_pack_asset_name() == "JustHireMe-runtime-pack-macos.zip"

    monkeypatch.setattr(runtime, "sys_platform", lambda: "linux")
    assert runtime.vector_runtime_asset_name() == "JustHireMe-vector-runtime-linux.zip"
    assert runtime.runtime_pack_asset_name() == "JustHireMe-runtime-pack-linux.zip"


def test_vector_runtime_roots_include_common_site_package_layouts(tmp_path, monkeypatch):
    from data.vector import runtime

    monkeypatch.setenv("JHM_VECTOR_RUNTIME_DIR", str(tmp_path / "vector-runtime"))

    roots = runtime.vector_runtime_roots()

    assert tmp_path / "vector-runtime" in roots
    assert tmp_path / "vector-runtime" / "site-packages" in roots
    assert tmp_path / "vector-runtime" / "Lib" / "site-packages" in roots


def test_runtime_status_requires_single_pack_components(monkeypatch, tmp_path):
    from data.vector import runtime

    monkeypatch.setenv("JHM_VECTOR_RUNTIME_DIR", str(tmp_path / "vector-runtime"))
    monkeypatch.setenv("JHM_BROWSER_RUNTIME_DIR", str(tmp_path / "browser-runtime" / "ms-playwright"))
    monkeypatch.setattr(runtime, "sys_platform", lambda: "windows")
    monkeypatch.setattr(runtime, "vector_runtime_ready", lambda _path=None: True)
    monkeypatch.setattr(runtime, "browser_runtime_ready", lambda _path=None: False)

    status = runtime.vector_runtime_status()

    assert status["ready"] is False
    assert status["asset"] == "JustHireMe-runtime-pack-windows.zip"
    assert status["vector"]["ready"] is True
    assert status["browser"]["ready"] is False


def test_runtime_pack_payload_detection_finds_vector_and_browser(tmp_path):
    from data.vector import runtime

    root = tmp_path / "extract"
    vector = root / "vector-runtime"
    browser = root / "browser-runtime" / "ms-playwright"
    (vector / "lancedb").mkdir(parents=True)
    (vector / "pyarrow").mkdir()
    (browser / "chromium-1200").mkdir(parents=True)

    vector_payload, browser_payload = runtime._runtime_pack_payloads(root)

    assert vector_payload == vector
    assert browser_payload == browser


def test_hash_embedding_fallback_reports_ok(monkeypatch):
    from data.vector import embeddings

    monkeypatch.setattr(embeddings, "_st", "hashing")
    monkeypatch.setattr(embeddings, "_st_error", "No module named 'sentence_transformers'")

    status = embeddings.embedding_status()

    assert status["status"] == "ok"
    assert status["mode"] == "hashing"
    assert "error" not in status


def test_embedding_loader_does_not_log_suppressed_exception_noise():
    text = Path(__file__).resolve().parents[1].joinpath("data", "vector", "embeddings.py").read_text(encoding="utf-8")

    assert "suppressed exception" not in text
