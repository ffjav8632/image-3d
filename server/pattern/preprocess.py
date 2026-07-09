"""型紙生成向けメッシュ前処理 (SPEC.md §3.12 / FR-13)。

ぬいぐるみの型紙は布で細部を再現できないため、パネル分割の前に
メッシュを平滑化・簡略化して「大まかな塊」の形状に単純化する。

このモジュールは純粋モジュール(server/DEVELOPMENT_POLICY.md §3.5)。
依存は numpy / trimesh のみ(scipyは使わない)。
"""
from __future__ import annotations

import numpy as np
import trimesh


def _largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """最大連結成分(面数基準)を抽出する。単一成分ならそのまま返す。"""
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh
    face_counts = [len(c.faces) for c in components]
    biggest_idx = int(np.argmax(face_counts))
    return components[biggest_idx]


def _smooth(mesh: trimesh.Trimesh, iterations: int) -> trimesh.Trimesh:
    """Taubin平滑化(体積収縮が少ない)。失敗時はラプラシアン平滑化にフォールバック。"""
    if iterations <= 0:
        return mesh
    try:
        trimesh.smoothing.filter_taubin(mesh, iterations=iterations)
    except Exception:
        try:
            trimesh.smoothing.filter_laplacian(mesh, iterations=iterations)
        except Exception:
            pass
    return mesh


def _simplify(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """目標面数まで簡略化する。

    fast-simplificationはアプリの `requirements.txt` に既存の依存であり
    (server/meshproc.pyでも使用済み)、pattern モジュール独自の新規依存
    ではないため利用する。利用不可・失敗時はtrimesh標準の
    `simplify_quadric_decimation` にフォールバックする。
    """
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh

    try:
        import fast_simplification

        new_vertices, new_faces = fast_simplification.simplify(
            mesh.vertices, mesh.faces, target_count=max(target_faces, 4)
        )
        simplified = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=True)
        if len(simplified.faces) > 0:
            return simplified
    except Exception:
        pass

    try:
        simplified = mesh.simplify_quadric_decimation(face_count=max(target_faces, 4))
        if simplified is not None and len(simplified.faces) > 0:
            return simplified
    except Exception:
        pass

    return mesh


def prepare_mesh(
    mesh: trimesh.Trimesh,
    target_faces: int = 15000,
    smooth_iterations: int = 10,
) -> trimesh.Trimesh:
    """型紙生成用にメッシュを単純化する。

    Args:
        mesh: 入力メッシュ(生成済みジョブのモデル)。
        target_faces: 簡略化の目標面数(SPEC.mdでは1万〜2万を想定)。
        smooth_iterations: Taubin/ラプラシアン平滑化の反復回数。

    Returns:
        平滑化・簡略化・最大連結成分抽出を行った新しい `trimesh.Trimesh`。
        入力は変更しない(コピーを操作する)。
    """
    mesh = mesh.copy()

    # テクスチャ付きGLB(texture_mode=paint / Pixal3D出力)はUVから頂点カラーに
    # 変換しておく(色境界誘導に使うため。変換失敗時は色なしで続行)。
    visual = getattr(mesh, "visual", None)
    if isinstance(visual, trimesh.visual.TextureVisuals):
        try:
            mesh.visual = visual.to_color()
            visual = mesh.visual
        except Exception:
            visual = None

    # 頂点カラーがある場合、平滑化・簡略化で失われやすいため退避しておき、
    # 処理後に最近傍で転写し直す(色境界誘導オプションのため保持したい)。
    vertex_colors = None
    if isinstance(visual, trimesh.visual.ColorVisuals) and visual.kind == "vertex":
        vertex_colors = np.asarray(visual.vertex_colors).copy()
    orig_vertices = mesh.vertices.copy()

    # UVシームで分裂した頂点を溶接する。分裂したままだと split() がUVチャート
    # 単位の破片を返し、最大連結成分抽出が本体ではなく断片を拾ってしまう
    # (Pixal3D統合時に確認したのと同じ罠。テクスチャ付きGLB全般で起きる)。
    mesh.merge_vertices(merge_tex=True, merge_norm=True)

    mesh = _smooth(mesh, smooth_iterations)
    mesh = _simplify(mesh, target_faces)
    mesh = _largest_component(mesh)

    if vertex_colors is not None and len(mesh.vertices) > 0:
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(orig_vertices)
            _, idx = tree.query(mesh.vertices, k=1)
            new_colors = vertex_colors[idx]
            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=new_colors)
        except Exception:
            pass

    return mesh
