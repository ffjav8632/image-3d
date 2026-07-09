"""server/pattern の単体テスト (Phase 4a / SPEC.md §3.12 FR-13)。

`server/pattern/` は純粋モジュール(server内の他モジュールをimportしない)。
ここではTestClient不要の純粋関数テストを中心に、icosphere・カプセルで
パネル分割の基本性質(全面被覆・パネル数・連結性・円盤位相・面積整合)を検証する。
色境界誘導(use_colors)は上半球赤/下半球青の球で境界エッジの色差整合率を検証する。
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from server.pattern.preprocess import prepare_mesh
from server.pattern.preview import build_preview_mesh
from server.pattern.segment import panel_stats, segment_panels


def _panel_is_connected(mesh: trimesh.Trimesh, labels: np.ndarray, panel_id: int) -> bool:
    face_idx = np.where(labels == panel_id)[0]
    if len(face_idx) == 0:
        return False
    sub = mesh.submesh([face_idx], append=True, repair=False)
    components = sub.split(only_watertight=False)
    return len(components) == 1


@pytest.fixture()
def icosphere():
    return trimesh.creation.icosphere(subdivisions=3, radius=50.0)


@pytest.fixture()
def capsule():
    return trimesh.creation.capsule(height=80.0, radius=20.0, count=[32, 32])


@pytest.mark.parametrize("n_panels", [4, 6, 8, 12])
def test_segment_panels_icosphere_basic_properties(icosphere, n_panels):
    mesh = icosphere
    labels = segment_panels(mesh, n_panels=n_panels, seed=0)

    assert len(labels) == len(mesh.faces)

    unique_labels = np.unique(labels)
    n_actual = len(unique_labels)
    assert 2 <= n_actual <= n_panels
    # ラベルは0始まりの連番であること
    assert list(unique_labels) == list(range(n_actual))

    for panel_id in unique_labels:
        assert _panel_is_connected(mesh, labels, panel_id)

    stats = panel_stats(mesh, labels)
    assert len(stats) == n_actual
    for s in stats:
        assert s["boundary_loops"] == 1
        assert s["disk_topology"] is True

    total_area_stats = sum(s["area_mm2"] for s in stats)
    assert total_area_stats == pytest.approx(mesh.area, rel=1e-6)

    total_faces_stats = sum(s["n_faces"] for s in stats)
    assert total_faces_stats == len(mesh.faces)


@pytest.mark.parametrize("n_panels", [4, 6, 8])
def test_segment_panels_capsule_basic_properties(capsule, n_panels):
    mesh = capsule
    labels = segment_panels(mesh, n_panels=n_panels, seed=1)

    assert len(labels) == len(mesh.faces)
    unique_labels = np.unique(labels)
    n_actual = len(unique_labels)
    assert 2 <= n_actual <= n_panels

    for panel_id in unique_labels:
        assert _panel_is_connected(mesh, labels, panel_id)

    stats = panel_stats(mesh, labels)
    for s in stats:
        assert s["disk_topology"] is True

    total_faces_stats = sum(s["n_faces"] for s in stats)
    assert total_faces_stats == len(mesh.faces)


def test_segment_panels_all_faces_covered_exactly_once(icosphere):
    labels = segment_panels(icosphere, n_panels=6, seed=3)
    assert labels.shape == (len(icosphere.faces),)
    assert labels.dtype == np.int64
    assert np.all(labels >= 0)


def test_segment_panels_small_mesh_returns_single_panel():
    box = trimesh.creation.box(extents=[10, 10, 10])
    # n_faces(12) <= n_panels(20) の場合は単一パネルに落とす
    labels = segment_panels(box, n_panels=20, seed=0)
    assert len(np.unique(labels)) == 1


def test_segment_panels_empty_mesh_returns_empty_array():
    empty = trimesh.Trimesh()
    labels = segment_panels(empty, n_panels=6)
    assert labels.shape == (0,)


def _hemisphere_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    """z>=0を赤、z<0を青に塗った頂点カラー(RGBA)を返す。"""
    colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
    colors[:, 3] = 255
    upper = mesh.vertices[:, 2] >= 0
    colors[upper] = [255, 0, 0, 255]
    colors[~upper] = [0, 0, 255, 255]
    return colors


def test_use_colors_guides_panel_boundary_to_color_edge(icosphere):
    """色境界誘導(use_colors=True)が、境界エッジを赤/青境界に沿わせることを検証する。

    n_panels=4 では自然な(色なしの)Voronoi分割は赤道と一致しないため、
    誘導ONの場合とOFFの場合で「パネル境界のうち色境界でもある比率」を
    比較する。誘導ONの方が明確に高いことを期待する。
    """
    mesh = icosphere
    colors = _hemisphere_colors(mesh)

    face_colors = colors[:, :3][mesh.faces].mean(axis=1).astype(np.float64)
    adjacency = mesh.face_adjacency
    color_diff = np.linalg.norm(
        face_colors[adjacency[:, 0]] - face_colors[adjacency[:, 1]], axis=1
    )
    is_color_boundary_edge = color_diff > 50.0
    assert is_color_boundary_edge.sum() > 0

    aligned_fractions_with = []
    aligned_fractions_without = []
    for seed in range(5):
        labels_with = segment_panels(
            mesh, n_panels=4, vertex_colors=colors, use_colors=True, seed=seed
        )
        labels_without = segment_panels(
            mesh, n_panels=4, vertex_colors=colors, use_colors=False, seed=seed
        )

        is_panel_boundary_with = labels_with[adjacency[:, 0]] != labels_with[adjacency[:, 1]]
        is_panel_boundary_without = (
            labels_without[adjacency[:, 0]] != labels_without[adjacency[:, 1]]
        )

        if is_panel_boundary_with.sum() > 0:
            aligned_fractions_with.append(
                is_color_boundary_edge[is_panel_boundary_with].mean()
            )
        if is_panel_boundary_without.sum() > 0:
            aligned_fractions_without.append(
                is_color_boundary_edge[is_panel_boundary_without].mean()
            )

    mean_with = float(np.mean(aligned_fractions_with))
    mean_without = float(np.mean(aligned_fractions_without))

    assert mean_with > 0.5
    assert mean_with > mean_without + 0.3


def test_prepare_mesh_smooths_simplifies_and_keeps_largest_component():
    box = trimesh.creation.icosphere(subdivisions=3, radius=30.0)
    debris = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    debris.apply_translation([100, 100, 100])
    combined = trimesh.util.concatenate([box, debris])

    prepared = prepare_mesh(combined, target_faces=200, smooth_iterations=5)

    assert len(prepared.faces) <= 260  # simplify目標付近(多少の余裕を許容)
    # デブリ(離れた連結成分)が除去され、本体のみが残ること
    assert prepared.bounds[1][0] < 90  # 元のdebris位置(x=100)まで広がっていない


def test_prepare_mesh_transfers_vertex_colors():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=30.0)
    colors = _hemisphere_colors(mesh)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)

    prepared = prepare_mesh(mesh, target_faces=300, smooth_iterations=3)

    visual = prepared.visual
    assert isinstance(visual, trimesh.visual.ColorVisuals)
    assert visual.kind == "vertex"
    new_colors = np.asarray(visual.vertex_colors)
    upper = prepared.vertices[:, 2] >= 0
    # 平滑化・簡略化後も概ね上半球=赤寄り、下半球=青寄りであること
    assert new_colors[upper][:, 0].mean() > new_colors[upper][:, 2].mean()
    assert new_colors[~upper][:, 2].mean() > new_colors[~upper][:, 0].mean()


def test_build_preview_mesh_preserves_topology_and_colors_by_panel():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=50.0)
    labels = segment_panels(mesh, n_panels=6, seed=0)
    preview = build_preview_mesh(mesh, labels)

    assert len(preview.faces) == len(mesh.faces)
    assert len(preview.vertices) == len(mesh.vertices)
    assert preview.visual.kind == "face"

    face_colors = np.asarray(preview.visual.face_colors)
    # 同じパネルの面はすべて同色であること
    for panel_id in np.unique(labels):
        idx = np.where(labels == panel_id)[0]
        colors_for_panel = face_colors[idx]
        assert np.all(colors_for_panel == colors_for_panel[0])

    # 異なるパネルは異なる色であること(先頭12パネルまでは固定パレットが巡回する前提)
    unique_panel_ids = np.unique(labels)
    if len(unique_panel_ids) <= 12:
        first_colors = [tuple(face_colors[labels == pid][0]) for pid in unique_panel_ids]
        assert len(set(first_colors)) == len(unique_panel_ids)
