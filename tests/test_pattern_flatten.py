"""server/pattern/flatten.py の単体テスト (Phase 4b / SPEC.md §3.12 FR-13)。

`server/pattern/` は純粋モジュール(server内の他モジュールをimportしない)。
既知形状(開いた円筒側面: 解析解あり、平面: 歪みほぼゼロ、半球: 非可展だが
妥当な歪み範囲)での平坦化結果を検証する。TestClient不要の純粋関数テスト。
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from server.pattern.flatten import flatten_panel


def _open_cylinder_side_mesh(
    n_theta: int = 24, n_h: int = 10, height: float = 40.0, radius: float = 8.0
) -> tuple[trimesh.Trimesh, int]:
    """開いた円筒側面を1本の縦シームで切り開いた(円盤位相の)メッシュを作る。

    UVパラメータ化が既知(theta方向=周長、z方向=高さ)なので、平坦化結果の
    高さ・周長が解析解と一致するかを検証できる。

    Returns:
        (mesh, cols): cols は1行あたりの頂点数(n_theta+1、シームで複製された列を含む)。
    """
    cols = n_theta + 1
    verts = []
    for j in range(n_h):
        for i in range(cols):
            theta = 2 * np.pi * i / n_theta
            z = height * j / (n_h - 1)
            verts.append([radius * np.cos(theta), radius * np.sin(theta), z])
    verts = np.array(verts)

    faces = []
    for j in range(n_h - 1):
        for i in range(n_theta):
            a = j * cols + i
            b = j * cols + i + 1
            c = (j + 1) * cols + i
            d = (j + 1) * cols + i + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    faces = np.array(faces)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return mesh, cols


def _planar_grid_mesh(nx: int = 8, ny: int = 8, width: float = 30.0, height: float = 20.0):
    xs, ys = np.meshgrid(np.linspace(0, width, nx), np.linspace(0, height, ny))
    verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(nx * ny)], axis=1)
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            b = j * nx + i + 1
            c = (j + 1) * nx + i
            d = (j + 1) * nx + i + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    faces = np.array(faces)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


# --------------------------------------------------------------------------
# 円筒側面: 解析解との比較
# --------------------------------------------------------------------------
def test_flatten_cylinder_matches_analytic_height_and_circumference():
    height = 40.0
    radius = 8.0
    mesh, cols = _open_cylinder_side_mesh(n_theta=24, n_h=10, height=height, radius=radius)

    result = flatten_panel(mesh, np.arange(len(mesh.faces)), n_arap_iterations=15)
    assert result["flatten_failed"] is False

    uv = result["vertices_2d"]
    n_h = len(mesh.vertices) // cols
    row0 = uv[0:cols]
    row_last = uv[(n_h - 1) * cols : (n_h - 1) * cols + cols]

    # row0・row_lastは(展開後は)直線かつ平行であるはずなので、その間の
    # 垂直距離が高さに、row0の全長が周長に一致するかを見る。
    direction = row0[-1] - row0[0]
    direction = direction / np.linalg.norm(direction)
    perp = np.array([-direction[1], direction[0]])
    measured_height = abs(np.dot(row_last[0] - row0[0], perp))
    measured_circumference = np.linalg.norm(row0[-1] - row0[0])

    expected_circumference = 2 * np.pi * radius * (24 / 24)  # 24分割中の弧(1周分)
    # シーム込みなのでcols-1=n_theta区間が1周に相当する
    expected_circumference = 2 * np.pi * radius

    assert measured_height == pytest.approx(height, rel=0.01)
    assert measured_circumference == pytest.approx(expected_circumference, rel=0.01)


def test_flatten_cylinder_low_edge_length_distortion():
    mesh, _ = _open_cylinder_side_mesh(n_theta=24, n_h=10, height=40.0, radius=8.0)
    result = flatten_panel(mesh, np.arange(len(mesh.faces)), n_arap_iterations=15)
    assert result["flatten_failed"] is False
    distortion = result["distortion"]
    assert distortion["edge_length_ratio_mean"] == pytest.approx(1.0, abs=0.02)
    assert distortion["edge_length_over_10pct_fraction"] < 0.05
    assert distortion["area_ratio_2d_to_3d"] == pytest.approx(1.0, abs=0.05)


# --------------------------------------------------------------------------
# 平面: 歪みほぼゼロ
# --------------------------------------------------------------------------
def test_flatten_planar_mesh_near_zero_distortion():
    mesh = _planar_grid_mesh()
    result = flatten_panel(mesh, np.arange(len(mesh.faces)), n_arap_iterations=8)
    assert result["flatten_failed"] is False
    distortion = result["distortion"]
    assert distortion["edge_length_ratio_max"] == pytest.approx(1.0, abs=1e-4)
    assert distortion["edge_length_ratio_min"] == pytest.approx(1.0, abs=1e-4)
    assert distortion["area_ratio_2d_to_3d"] == pytest.approx(1.0, abs=1e-4)
    assert distortion["edge_length_over_10pct_fraction"] == 0.0


# --------------------------------------------------------------------------
# 半球: 非可展だが歪みは妥当な範囲
# --------------------------------------------------------------------------
def test_flatten_hemisphere_distortion_within_reasonable_range():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=20.0)
    face_centers = sphere.triangles_center
    upper_faces = np.where(face_centers[:, 2] > 0)[0]

    result = flatten_panel(sphere, upper_faces, n_arap_iterations=12)
    assert result["flatten_failed"] is False
    distortion = result["distortion"]

    # 非可展面なので歪みは出るが、平均辺長歪みは1に近いレンジ、面積比も
    # 大きく崩れない(半球は比較的緩やかな曲率のため)ことを確認する。
    assert 0.8 <= distortion["edge_length_ratio_mean"] <= 1.2
    assert 0.5 <= distortion["area_ratio_2d_to_3d"] <= 1.5


# --------------------------------------------------------------------------
# ARAP反復による歪み改善
# --------------------------------------------------------------------------
def test_arap_iterations_improve_on_lscm_only():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=20.0)
    face_centers = sphere.triangles_center
    upper_faces = np.where(face_centers[:, 2] > 0)[0]

    lscm_only = flatten_panel(sphere, upper_faces, n_arap_iterations=0)
    with_arap = flatten_panel(sphere, upper_faces, n_arap_iterations=12)

    assert lscm_only["flatten_failed"] is False
    assert with_arap["flatten_failed"] is False

    lscm_over_10pct = lscm_only["distortion"]["edge_length_over_10pct_fraction"]
    arap_over_10pct = with_arap["distortion"]["edge_length_over_10pct_fraction"]
    assert arap_over_10pct < lscm_over_10pct

    lscm_area_err = abs(lscm_only["distortion"]["area_ratio_2d_to_3d"] - 1.0)
    arap_area_err = abs(with_arap["distortion"]["area_ratio_2d_to_3d"] - 1.0)
    assert arap_area_err < lscm_area_err


# --------------------------------------------------------------------------
# 円盤位相でないパネル: 例外にせず flatten_failed=True
# --------------------------------------------------------------------------
def test_flatten_non_disk_topology_reports_failure_without_raising():
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=10.0)  # 閉曲面(境界なし)
    result = flatten_panel(sphere, np.arange(len(sphere.faces)))
    assert result["flatten_failed"] is True
    assert "reason" in result
    assert result["reason"]


def test_flatten_empty_panel_reports_failure_without_raising():
    sphere = trimesh.creation.icosphere(subdivisions=1, radius=10.0)
    result = flatten_panel(sphere, np.array([], dtype=np.int64))
    assert result["flatten_failed"] is True


def test_flatten_disconnected_faces_reports_failure_without_raising():
    # 2つの独立した平面片(非連結)をfaceとして渡す
    mesh1 = _planar_grid_mesh(nx=3, ny=3, width=5, height=5)
    mesh2 = _planar_grid_mesh(nx=3, ny=3, width=5, height=5)
    mesh2.vertices += np.array([100.0, 100.0, 100.0])
    combined = trimesh.util.concatenate([mesh1, mesh2])

    result = flatten_panel(combined, np.arange(len(combined.faces)))
    assert result["flatten_failed"] is True


# --------------------------------------------------------------------------
# 出力構造の基本性質
# --------------------------------------------------------------------------
def test_flatten_output_shapes_are_consistent():
    mesh = _planar_grid_mesh()
    result = flatten_panel(mesh, np.arange(len(mesh.faces)), n_arap_iterations=5)
    assert result["flatten_failed"] is False

    n_verts = len(result["vertices_2d"])
    assert result["vertices_2d"].shape == (n_verts, 2)
    assert result["vertices_3d"].shape == (n_verts, 3)
    assert result["faces"].max() < n_verts
    assert result["boundary_loop_2d"].ndim == 2
    assert result["boundary_loop_2d"].shape[1] == 2
    assert len(result["boundary_loop_indices"]) == len(result["boundary_loop_2d"])
    assert np.all(np.isfinite(result["vertices_2d"]))
