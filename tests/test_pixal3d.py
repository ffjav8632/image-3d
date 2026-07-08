"""Pixal3Dジェネレータ統合のGPU不要テスト (SPEC.md §3.3)。

実モデルのロード・生成はGPU/23GB重みが必要なためここでは行わず、
以下のGPU不要な純ロジックのみを検証する:
  - ジェネレータのバリデーション(マルチビュー拒否、アルファ無し画像拒否)
  - _build_generator("pixal3d") の解決
  - カメラ距離計算(固定FOV、upstream inference.py と等価であること)
  - 頂点カラー検出ヘルパ(jobs._mesh_vertex_colors)
  - 頂点カラー付きraw mesh → 後処理 → 転写 → color4量子化・GLB出力のE2E
    (頂点カラーを返すスタブジェネレータ使用)
"""
import io
import math
import time

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from PIL import Image

from server.generators.base import GenerationParams
from server.generators.mock import MockGenerator
from server.generators.pixal3d import (
    Pixal3DGenerator,
    _distance_from_fov,
    _has_meaningful_alpha,
)


def make_rgba_image(size=64, with_alpha=True) -> Image.Image:
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[:, :] = [200, 50, 50, 255]
    if with_alpha:
        arr[: size // 4, :, 3] = 0  # 上1/4を透明(背景除去済みを模す)
    return Image.fromarray(arr, "RGBA")


# --- ジェネレータバリデーション ------------------------------------------------


def test_generator_name():
    gen = Pixal3DGenerator()
    assert gen.name == "pixal3d"


def test_multiview_rejected_without_loading_pipeline():
    """extra_views指定時は(モデルロード前に)明示的なエラーを送出する。"""
    gen = Pixal3DGenerator()
    image = make_rgba_image()
    with pytest.raises(ValueError, match="マルチビュー"):
        gen.generate(
            image,
            GenerationParams(),
            extra_views={"back": make_rgba_image()},
        )
    assert gen._pipeline is None  # ロードは走っていない


def test_image_without_alpha_rejected_without_loading_pipeline():
    """背景除去無し(アルファ無し)の画像は意味のあるエラーで拒否する。"""
    gen = Pixal3DGenerator()
    image = Image.new("RGB", (64, 64), (200, 50, 50))
    with pytest.raises(RuntimeError, match="アルファ"):
        gen.generate(image, GenerationParams())
    assert gen._pipeline is None


def test_has_meaningful_alpha():
    assert _has_meaningful_alpha(make_rgba_image(with_alpha=True))
    # 全画素不透明のRGBAは「背景除去済み」とみなさない
    assert not _has_meaningful_alpha(make_rgba_image(with_alpha=False))
    assert not _has_meaningful_alpha(Image.new("RGB", (8, 8)))


# --- カメラ距離計算 -------------------------------------------------------------


def test_distance_from_fov_matches_upstream():
    """upstream inference.py の distance_from_fov と等価であること(実測基準値)。

    fov=0.6 rad, image_resolution=512 のとき distance=1.61636... (実機E2Eで確認)。
    """
    d = _distance_from_fov(0.6, image_resolution=512)
    assert d == pytest.approx(1.6163638830184937, rel=1e-6)


# --- _build_generator 解決 ------------------------------------------------------


def test_build_generator_resolves_pixal3d(monkeypatch):
    from server import config
    from server.main import _build_generator

    monkeypatch.setattr(config, "GENERATOR", "pixal3d")
    gen = _build_generator()
    assert isinstance(gen, Pixal3DGenerator)
    assert gen.name == "pixal3d"


def test_build_generator_auto_does_not_resolve_pixal3d(monkeypatch):
    """autoの解決順は現状維持(hunyuan3d/mock)。pixal3dは明示指定のみ。"""
    from server import config
    from server.main import _build_generator

    monkeypatch.setattr(config, "GENERATOR", "auto")
    gen = _build_generator()
    assert gen.name in ("hunyuan3d", "mock")


# --- 頂点カラー検出ヘルパ -------------------------------------------------------


def test_mesh_vertex_colors_detects_explicit_colors():
    from server.jobs import _mesh_vertex_colors

    mesh = trimesh.creation.box(extents=[1, 1, 1])
    assert _mesh_vertex_colors(mesh) is None  # デフォルトカラーは「無し」扱い

    colors = np.tile(np.array([255, 0, 0, 255], dtype=np.uint8), (len(mesh.vertices), 1))
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
    detected = _mesh_vertex_colors(mesh)
    assert detected is not None
    assert detected.shape == (len(mesh.vertices), 4)
    np.testing.assert_array_equal(detected[0], [255, 0, 0, 255])


# --- 頂点カラー付きジェネレータのジョブE2E(スタブ使用、GPU不要) ------------------


class ColoredMockGenerator(MockGenerator):
    """raw meshにZ座標ベースの2色頂点カラーを付与して返すスタブ。

    pixal3dの「テクスチャ→頂点カラー→jobs.pyで転写」経路を実モデル無しで
    検証するためのテスト専用ジェネレータ。
    """

    name = "colored-mock"

    def generate(self, image, params, extra_views=None):
        mesh = super().generate(image, params, extra_views)
        z = mesh.vertices[:, 2]
        z_mid = (z.min() + z.max()) / 2.0
        colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
        colors[:] = [0, 0, 255, 255]  # 下半分: 青
        colors[z > z_mid] = [255, 0, 0, 255]  # 上半分: 赤
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
        return mesh


@pytest.fixture()
def colored_client(tmp_path, monkeypatch):
    from server import config
    from server import main as main_module

    data_dir = tmp_path / "data"
    jobs_dir = data_dir / "jobs"
    jobs_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "JOBS_DIR", jobs_dir)

    main_module.job_manager.jobs = {}
    monkeypatch.setattr(main_module.job_manager, "generator", ColoredMockGenerator())

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


def make_png_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_generator_vertex_colors_transferred_to_glb_and_palette(colored_client):
    """頂点カラー付きraw mesh → meshproc → 最近傍転写 → color4量子化・GLB出力。"""
    res = colored_client.post(
        "/api/jobs",
        files={"image": ("test.png", make_png_bytes(), "image/png")},
        data={"params": '{"color_mode": "color4", "n_colors": 4, "remove_bg": false, "seed": 7}'},
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    job = _wait_for_completion(colored_client, job_id)
    assert job["status"] == "completed", job.get("error")

    # パレットはスタブの2色(赤/青)近傍に量子化される
    palette = job["stats"]["palette"]
    assert 2 <= len(palette) <= 4
    hexes = {p["hex"] for p in palette}
    assert any(h.startswith("#") for h in hexes)

    # GLBに頂点カラーが載っていること
    res = colored_client.get(f"/api/jobs/{job_id}/model.glb")
    assert res.status_code == 200
    scene = trimesh.load(io.BytesIO(res.content), file_type="glb", process=False)
    meshes = (
        list(scene.geometry.values()) if isinstance(scene, trimesh.Scene) else [scene]
    )
    mesh = meshes[0]
    assert mesh.visual.kind == "vertex"
    vc = np.asarray(mesh.visual.vertex_colors)
    # 赤・青両方が転写されている(単色ではない)
    assert len(np.unique(vc[:, :3], axis=0)) >= 2

    # color4の3MF(色分割版)がオブジェクトを持つこと
    res = colored_client.get(f"/api/jobs/{job_id}/download?format=3mf")
    assert res.status_code == 200
    tmf = trimesh.load(io.BytesIO(res.content), file_type="3mf")
    assert isinstance(tmf, trimesh.Scene)
    assert 2 <= len(tmf.geometry) <= 4


def test_generator_vertex_colors_on_glb_without_color_mode(colored_client):
    """color_mode=none でも、ジェネレータ由来の頂点カラーはGLBに保存される。"""
    res = colored_client.post(
        "/api/jobs",
        files={"image": ("test.png", make_png_bytes(), "image/png")},
        data={"params": '{"remove_bg": false, "seed": 7}'},
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    job = _wait_for_completion(colored_client, job_id)
    assert job["status"] == "completed", job.get("error")

    res = colored_client.get(f"/api/jobs/{job_id}/model.glb")
    assert res.status_code == 200
    scene = trimesh.load(io.BytesIO(res.content), file_type="glb", process=False)
    meshes = (
        list(scene.geometry.values()) if isinstance(scene, trimesh.Scene) else [scene]
    )
    assert meshes[0].visual.kind == "vertex"
