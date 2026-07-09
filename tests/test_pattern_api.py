"""型紙生成API のテスト (SPEC.md §3.12 / FR-13, Phase 4a)。

`server/pattern/` 自体は純粋モジュール単体テスト(test_pattern_segment.py)で
検証済み。ここではmainのアダプタ(ジョブディレクトリ接続・バリデーション・
ステータスコード)をmockジェネレータ+TestClientで検証する。
"""
from __future__ import annotations

import time

import pytest
import trimesh
from fastapi.testclient import TestClient

from tests.conftest import make_test_png_bytes


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from server import config

    data_dir = tmp_path / "data"
    jobs_dir = data_dir / "jobs"
    jobs_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "JOBS_DIR", jobs_dir)

    from server import main as main_module

    main_module.job_manager.jobs = {}

    with TestClient(main_module.app) as c:
        yield c


def _wait_for_completion(client, job_id, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        res = client.get(f"/api/jobs/{job_id}")
        assert res.status_code == 200
        job = res.json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.1)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def _create_completed_job(client) -> str:
    png_bytes = make_test_png_bytes()
    res = client.post(
        "/api/jobs",
        files={"image": ("test.png", png_bytes, "image/png")},
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    job = _wait_for_completion(client, job_id)
    assert job["status"] == "completed"
    return job_id


def test_pattern_generation_e2e(client):
    job_id = _create_completed_job(client)

    res = client.post(f"/api/jobs/{job_id}/pattern", json={"n_panels": 6})
    assert res.status_code == 200
    data = res.json()
    assert data["job_id"] == job_id
    assert data["n_panels_requested"] == 6
    assert 2 <= data["n_panels_actual"] <= 6
    assert len(data["panels"]) == data["n_panels_actual"]
    for panel in data["panels"]:
        assert "panel_id" in panel
        assert "n_faces" in panel
        assert "area_mm2" in panel
        assert "boundary_loops" in panel
        assert "disk_topology" in panel

    # pattern.json取得
    res_json = client.get(f"/api/jobs/{job_id}/pattern.json")
    assert res_json.status_code == 200
    assert res_json.json()["n_panels_actual"] == data["n_panels_actual"]

    # pattern_preview.glb取得。trimeshで再読込できること。
    res_glb = client.get(f"/api/jobs/{job_id}/pattern_preview.glb")
    assert res_glb.status_code == 200
    assert res_glb.headers["content-type"] == "model/gltf-binary"

    import io

    loaded = trimesh.load(io.BytesIO(res_glb.content), file_type="glb")
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        assert len(meshes) >= 1
    else:
        assert len(loaded.faces) > 0


def test_pattern_generation_uses_defaults_with_empty_body(client):
    job_id = _create_completed_job(client)
    res = client.post(f"/api/jobs/{job_id}/pattern", json={})
    assert res.status_code == 200
    data = res.json()
    assert data["n_panels_requested"] == 6
    assert data["use_colors"] is True


def test_pattern_generation_rejects_incomplete_job(client):
    png_bytes = make_test_png_bytes()
    res = client.post(
        "/api/jobs",
        files={"image": ("test.png", png_bytes, "image/png")},
    )
    job_id = res.json()["job_id"]

    # 完了前(queued直後)にリクエストするとレースになりうるため、
    # 明示的にjobオブジェクトのstatusを差し替える。
    from server import main as main_module

    job = main_module.job_manager.get_job(job_id)
    job.status = "generating"

    res_pattern = client.post(f"/api/jobs/{job_id}/pattern", json={"n_panels": 6})
    assert res_pattern.status_code == 409


def test_pattern_generation_rejects_out_of_range_n_panels(client):
    job_id = _create_completed_job(client)

    res = client.post(f"/api/jobs/{job_id}/pattern", json={"n_panels": 3})
    assert res.status_code == 400

    res2 = client.post(f"/api/jobs/{job_id}/pattern", json={"n_panels": 13})
    assert res2.status_code == 400


def test_pattern_generation_rejects_bad_smooth_iterations(client):
    job_id = _create_completed_job(client)
    res = client.post(f"/api/jobs/{job_id}/pattern", json={"smooth_iterations": -1})
    assert res.status_code == 400


def test_pattern_json_and_glb_404_before_generation(client):
    job_id = _create_completed_job(client)
    res = client.get(f"/api/jobs/{job_id}/pattern.json")
    assert res.status_code == 404
    res2 = client.get(f"/api/jobs/{job_id}/pattern_preview.glb")
    assert res2.status_code == 404


def test_pattern_generation_nonexistent_job_404(client):
    res = client.post("/api/jobs/does-not-exist/pattern", json={"n_panels": 6})
    assert res.status_code == 404
