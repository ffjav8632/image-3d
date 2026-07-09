"""パネル平坦化 (SPEC.md §3.12 / FR-13 の4b部分)。

円盤位相の3Dパネル(部分メッシュ)を2Dへ展開する。

アルゴリズム:
    1. **LSCM初期解**(least squares conformal map): 各三角形をその面の
       局所2D等長座標系(法線に直交する平面へ正射影)へ写し、コーシー・
       リーマン方程式の残差を最小二乗で解く(scipy.sparse.linalg使用)。
       2頂点を固定して自明解(全頂点0)を避ける。
    2. **ARAP反復**(as-rigid-as-possible): LSCM解を初期値として、
       局所ステップ(三角形ごとに3D→2Dの最適回転を2x2 SVDで求める)と
       大域ステップ(コットジェント重み付きラプラシアン系を解いて頂点位置を
       更新)を5〜15回交互に反復し、辺長歪みを低減する。

歪み指標:
    - 辺長歪み: 各辺について (2D長 / 3D長) の比を計算し、最大・平均・
      ±10%超(比が0.9未満または1.1超)の辺の割合を報告する。
    - 面積比: 2D総面積 / 3D総面積。

円盤位相でないパネル(境界ループが1本でない、または非連結)は例外にせず
`{"flatten_failed": True, "reason": ...}` を返し、呼び出し側が他パネルの
処理を継続できるようにする。縮退三角形・特異な線形システムには
擬似逆・微小正則化で防御する。

このモジュールは純粋モジュール(server/DEVELOPMENT_POLICY.md §3.5)。
依存は numpy / scipy / trimesh のみ。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import trimesh
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import lsqr, spsolve

_EPS = 1e-12


# --------------------------------------------------------------------------
# サブメッシュ抽出・境界検出
# --------------------------------------------------------------------------
def _extract_submesh(mesh: trimesh.Trimesh, face_indices: np.ndarray) -> Optional[trimesh.Trimesh]:
    face_indices = np.asarray(face_indices, dtype=np.int64)
    if len(face_indices) == 0:
        return None
    sub = mesh.submesh([face_indices], append=True, repair=False)
    if sub is None or len(sub.faces) == 0:
        return None
    return sub


def _boundary_loops(sub: trimesh.Trimesh) -> list[list[int]]:
    """境界ループを頂点インデックスの順序付きリストのリストとして返す。

    trimeshの `outline()` が使えればそれを使い、失敗時は手動でエッジを
    たどって復元する。
    """
    try:
        outline = sub.outline(process=False)
        if outline is not None and len(outline.entities) > 0:
            loops = []
            for entity in outline.entities:
                pts = list(entity.points)
                if len(pts) >= 2 and pts[0] == pts[-1]:
                    pts = pts[:-1]
                loops.append([int(p) for p in pts])
            return loops
    except Exception:
        pass
    return _boundary_loops_manual(sub)


def _boundary_loops_manual(sub: trimesh.Trimesh) -> list[list[int]]:
    edges = sub.edges_sorted
    edges_unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = edges_unique[counts == 1]
    if len(boundary_edges) == 0:
        return []

    adjacency: dict[int, list[int]] = {}
    for a, b in boundary_edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))

    visited: set[int] = set()
    loops: list[list[int]] = []
    for start in adjacency:
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        current = start
        prev = None
        while True:
            neighbors = [n for n in adjacency[current] if n != prev]
            nxt = None
            for n in neighbors:
                if n == start and len(loop) > 2:
                    nxt = start
                    break
                if n not in visited:
                    nxt = n
                    break
            if nxt is None or nxt == start:
                break
            loop.append(nxt)
            visited.add(nxt)
            prev = current
            current = nxt
        loops.append(loop)
    return loops


def _is_disk_topology(sub: trimesh.Trimesh, loops: list[list[int]]) -> tuple[bool, str]:
    components = sub.split(only_watertight=False)
    if len(components) != 1:
        return False, f"non_connected (components={len(components)})"
    if len(loops) != 1:
        return False, f"boundary_loops={len(loops)} (expected 1)"
    if len(loops[0]) < 3:
        return False, "boundary_loop_too_short"
    return True, ""


# --------------------------------------------------------------------------
# LSCM 初期解
# --------------------------------------------------------------------------
def _local_triangle_basis(tri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """三角形頂点(3,3)から局所2D座標(等長: 辺長を保つ正射影)を返す。

    Returns:
        (x, y): 各頂点の局所2D座標を表す長さ3配列2本。
        頂点0が原点、頂点1がx軸上。
    """
    p0, p1, p2 = tri
    e0 = p1 - p0
    len_e0 = np.linalg.norm(e0)
    if len_e0 < _EPS:
        return np.zeros(3), np.zeros(3)
    x_axis = e0 / len_e0
    e1 = p2 - p0
    normal = np.cross(e0, e1)
    norm_len = np.linalg.norm(normal)
    if norm_len < _EPS:
        return np.zeros(3), np.zeros(3)
    normal = normal / norm_len
    y_axis = np.cross(normal, x_axis)

    x = np.array([0.0, len_e0, np.dot(e1, x_axis)])
    y = np.array([0.0, 0.0, np.dot(e1, y_axis)])
    return x, y


def _choose_pin_vertices(vertices: np.ndarray, faces: np.ndarray) -> tuple[int, int]:
    """固定する2頂点を選ぶ。

    バウンディングボックス対角の両端(任意の2頂点)を選ぶと、その2点間の
    直線距離がメッシュ上の実際の測地距離と乖離している場合に大きな
    シェア(せん断)を生み、ARAP反復でも解消できない大域的な歪みの
    原因になる(円筒側面のような細長いパネルで顕著)。そこで、実際に
    メッシュ上で最長のエッジ(2頂点が直接隣接)を選ぶことで、固定した
    2点間の距離が常に実測の辺長と一致するようにし、シェアの種を作らない。
    """
    n_verts = len(vertices)
    if n_verts < 2:
        return 0, min(1, max(n_verts - 1, 0))

    edges = set()
    for i0, i1, i2 in faces:
        for a, b in ((int(i0), int(i1)), (int(i1), int(i2)), (int(i2), int(i0))):
            edges.add((a, b) if a < b else (b, a))

    if not edges:
        return 0, min(1, n_verts - 1)

    best_pair = None
    best_len = -1.0
    for a, b in edges:
        length = float(np.linalg.norm(vertices[a] - vertices[b]))
        if length > best_len:
            best_len = length
            best_pair = (a, b)
    return best_pair


def _lscm_solve(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """LSCM (least squares conformal map) で初期2D展開を求める。

    2頂点(メッシュ上で実際に隣接する最長エッジの両端)を固定して自明解を
    避ける(理由は `_choose_pin_vertices` docstring参照)。複素表現
    z = u + i v を使い、各三角形についてコーシー・リーマン残差
    (実部・虚部) を最小二乗の行として積む(標準的なLSCM定式化)。
    """
    n_verts = len(vertices)
    n_faces = len(faces)
    if n_verts < 3 or n_faces == 0:
        return np.zeros((n_verts, 2))

    pin_a, pin_b = _choose_pin_vertices(vertices, faces)
    pin_dist = np.linalg.norm(vertices[pin_a] - vertices[pin_b])
    pin_uv = np.array([[0.0, 0.0], [max(pin_dist, 1e-6), 0.0]])

    free_mask = np.ones(n_verts, dtype=bool)
    free_mask[[pin_a, pin_b]] = False
    free_indices = np.where(free_mask)[0]
    free_map = -np.ones(n_verts, dtype=np.int64)
    free_map[free_indices] = np.arange(len(free_indices))
    n_free = len(free_indices)

    if n_free == 0:
        result = np.zeros((n_verts, 2))
        result[pin_a] = pin_uv[0]
        result[pin_b] = pin_uv[1]
        return result

    return _lscm_assemble_and_solve(
        vertices, faces, pin_a, pin_b, pin_uv, free_map, n_free, n_verts
    )


def _lscm_assemble_and_solve(
    vertices: np.ndarray,
    faces: np.ndarray,
    pin_a: int,
    pin_b: int,
    pin_uv: np.ndarray,
    free_map: np.ndarray,
    n_free: int,
    n_verts: int,
) -> np.ndarray:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs: dict[int, float] = {}
    row_counter = 0

    pin_full = {pin_a: pin_uv[0], pin_b: pin_uv[1]}

    for i0, i1, i2 in faces:
        tri3d = vertices[[i0, i1, i2]]
        x, y = _local_triangle_basis(tri3d)
        area2 = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
        sqrt_area = np.sqrt(max(abs(area2) / 2.0, _EPS))
        if sqrt_area < 1e-9:
            continue

        w0 = complex(x[2] - x[1], y[2] - y[1])
        w1 = complex(x[0] - x[2], y[0] - y[2])
        w2 = complex(x[1] - x[0], y[1] - y[0])
        inv_norm = 1.0 / sqrt_area
        coeffs = {int(i0): w0 * inv_norm, int(i1): w1 * inv_norm, int(i2): w2 * inv_norm}

        row_r = row_counter * 2
        row_i = row_counter * 2 + 1
        const_r = 0.0
        const_i = 0.0

        for vidx, c in coeffs.items():
            if free_map[vidx] >= 0:
                fi = int(free_map[vidx])
                rows.append(row_r)
                cols.append(fi * 2)
                vals.append(c.real)
                rows.append(row_r)
                cols.append(fi * 2 + 1)
                vals.append(-c.imag)

                rows.append(row_i)
                cols.append(fi * 2)
                vals.append(c.imag)
                rows.append(row_i)
                cols.append(fi * 2 + 1)
                vals.append(c.real)
            else:
                u_p, v_p = pin_full[vidx]
                const_r += c.real * u_p - c.imag * v_p
                const_i += c.imag * u_p + c.real * v_p

        rhs[row_r] = -const_r
        rhs[row_i] = -const_i
        row_counter += 1

    n_rows = row_counter * 2
    n_cols = n_free * 2
    if n_rows == 0 or n_cols == 0:
        result = np.zeros((n_verts, 2))
        result[pin_a] = pin_uv[0]
        result[pin_b] = pin_uv[1]
        return result

    A = coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols)).tocsr()
    b = np.zeros(n_rows)
    for r, v in rhs.items():
        b[r] = v

    try:
        sol = lsqr(A, b, atol=1e-10, btol=1e-10, iter_lim=2000)[0]
    except Exception:
        sol = np.zeros(n_cols)

    result = np.zeros((n_verts, 2))
    result[pin_a] = pin_uv[0]
    result[pin_b] = pin_uv[1]
    for vidx in range(n_verts):
        fi = free_map[vidx]
        if fi >= 0:
            result[vidx, 0] = sol[fi * 2]
            result[vidx, 1] = sol[fi * 2 + 1]
    return result


# --------------------------------------------------------------------------
# ARAP 反復
# --------------------------------------------------------------------------
def _cotangent_weights(vertices: np.ndarray, faces: np.ndarray) -> dict[tuple[int, int], float]:
    """三角形ごとのコットジェント重みを辺(i,j)ペアごとに累積する。"""
    weights: dict[tuple[int, int], float] = {}
    for tri in faces:
        i0, i1, i2 = [int(v) for v in tri]
        p0, p1, p2 = vertices[i0], vertices[i1], vertices[i2]
        _accumulate_cot(weights, i0, i1, i2, p0, p1, p2)
        _accumulate_cot(weights, i1, i2, i0, p1, p2, p0)
        _accumulate_cot(weights, i2, i0, i1, p2, p0, p1)
    return weights


def _accumulate_cot(
    weights: dict[tuple[int, int], float],
    i: int,
    j: int,
    k: int,
    pi: np.ndarray,
    pj: np.ndarray,
    pk: np.ndarray,
) -> None:
    """頂点kの対角にある辺(i,j)へのコットジェント寄与を加算する。"""
    v1 = pi - pk
    v2 = pj - pk
    cross = np.linalg.norm(np.cross(v1, v2))
    if cross < _EPS:
        cot = 0.0
    else:
        cot = float(np.dot(v1, v2) / cross)
    key = (i, j) if i < j else (j, i)
    weights[key] = weights.get(key, 0.0) + 0.5 * cot


def _half_cotangent(p_opposite: np.ndarray, p_i: np.ndarray, p_j: np.ndarray) -> float:
    """辺(i,j)の対角にある頂点p_oppositeの角度のコットジェントの半分を返す
    (この三角形1つのみの寄与。局所2D座標(等長)で計算する)。
    """
    v1 = p_i - p_opposite
    v2 = p_j - p_opposite
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = np.dot(v1, v2)
    if abs(cross) < _EPS:
        return 0.0
    return 0.5 * float(dot / abs(cross))


def _local_rotations(
    vertices3d: np.ndarray, faces: np.ndarray, uv: np.ndarray
) -> list[np.ndarray]:
    """三角形ごとに、3D辺ベクトル集合を2D辺ベクトル集合へ最も近づける
    最適回転(2x2、行列式>0)を2x2 SVDで求める。
    """
    rotations = []
    for i0, i1, i2 in faces:
        tri3d = vertices3d[[i0, i1, i2]]
        x, y = _local_triangle_basis(tri3d)
        p_local = np.stack([x, y], axis=1)  # (3,2) local isometric 3D coords

        p0_2d, p1_2d, p2_2d = uv[i0], uv[i1], uv[i2]

        # 局所3D座標系での辺ベクトル (2,2行列: 各列が辺)
        e1_3 = p_local[1] - p_local[0]
        e2_3 = p_local[2] - p_local[0]
        S3 = np.stack([e1_3, e2_3], axis=1)  # (2,2)

        e1_2 = p1_2d - p0_2d
        e2_2 = p2_2d - p0_2d
        S2 = np.stack([e1_2, e2_2], axis=1)  # (2,2)

        M = S2 @ np.linalg.pinv(S3)
        try:
            U, _, Vt = np.linalg.svd(M)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt
        except np.linalg.LinAlgError:
            R = np.eye(2)
        rotations.append(R)
    return rotations


def _flatten_lscm_and_arap(
    vertices3d: np.ndarray,
    faces: np.ndarray,
    n_iterations: int = 10,
) -> np.ndarray:
    """LSCM初期解 → ARAP反復で2D頂点座標を求める。"""
    n_verts = len(vertices3d)
    uv = _lscm_solve(vertices3d, faces)

    if n_iterations <= 0 or len(faces) == 0:
        return uv

    cot_weights = _cotangent_weights(vertices3d, faces)

    # 大域ステップ用ラプラシアン行列(固定)を事前構築
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    diag = np.zeros(n_verts)
    for (i, j), w in cot_weights.items():
        w = max(w, 0.0)  # 負の重み(鈍角三角形由来)は不安定要因になりうるためクランプ
        rows.append(i)
        cols.append(j)
        vals.append(-w)
        rows.append(j)
        cols.append(i)
        vals.append(-w)
        diag[i] += w
        diag[j] += w
    for i in range(n_verts):
        rows.append(i)
        cols.append(i)
        vals.append(diag[i])

    L = coo_matrix((vals, (rows, cols)), shape=(n_verts, n_verts)).tocsr()

    # ピン頂点(LSCMと同じ選択基準: 実際に隣接する最長エッジの両端)を
    # 固定して大域ステップの特異性を避ける。
    pin_a, pin_b = _choose_pin_vertices(vertices3d, faces)

    free_mask = np.ones(n_verts, dtype=bool)
    free_mask[[pin_a, pin_b]] = False
    free_indices = np.where(free_mask)[0]

    if len(free_indices) == 0:
        return uv

    L_free = L[free_indices][:, free_indices]
    # 微小正則化(特異性防御)
    L_free = L_free + csr_matrix(
        (1e-9 * np.ones(len(free_indices)), (np.arange(len(free_indices)), np.arange(len(free_indices)))),
        shape=L_free.shape,
    )
    L_pin = L[free_indices][:, [pin_a, pin_b]]

    # 三角形ごとの辺の局所2D基底(3D等長座標)を事前計算しておく
    local_bases = []
    for i0, i1, i2 in faces:
        tri3d = vertices3d[[i0, i1, i2]]
        x, y = _local_triangle_basis(tri3d)
        local_bases.append((x, y))

    for _ in range(n_iterations):
        rotations = _local_rotations(vertices3d, faces, uv)

        rhs = np.zeros((n_verts, 2))
        for f_idx, (i0, i1, i2) in enumerate(faces):
            tri = [int(i0), int(i1), int(i2)]
            x, y = local_bases[f_idx]
            p_local = np.stack([x, y], axis=1)  # (3,2)
            R = rotations[f_idx]
            # このtriangleにおける各辺の対角コットジェント(このtriangleのみの寄与、
            # 半分の重み)。大域行列Lは両側三角形の和(0.5*(cot_a+cot_b))を持つため、
            # ここでは同じ0.5*cot_thisを使い、隣接三角形側の反復でもう半分が
            # 加算されることで整合させる(標準ARAP: RHS_i = sum_{j in N(i)} 0.5*w_ij*(R_i+R_j)(p_i-p_j)
            # の各三角形ごとの半分の寄与に相当)。
            half_cot = {
                (tri[0], tri[1]): _half_cotangent(p_local[2], p_local[0], p_local[1]),
                (tri[1], tri[2]): _half_cotangent(p_local[0], p_local[1], p_local[2]),
                (tri[2], tri[0]): _half_cotangent(p_local[1], p_local[2], p_local[0]),
            }
            for a_local, b_local in ((0, 1), (1, 2), (2, 0)):
                gi = tri[a_local]
                gj = tri[b_local]
                w = half_cot[(gi, gj)]
                if w <= 0.0:
                    continue
                edge_local = p_local[a_local] - p_local[b_local]
                target = w * (R @ edge_local)
                rhs[gi] += target
                rhs[gj] -= target

        rhs_free = rhs[free_indices]
        pin_coords = np.stack([uv[pin_a], uv[pin_b]], axis=0)
        rhs_adjusted = rhs_free - L_pin @ pin_coords

        try:
            new_free = np.column_stack(
                [
                    spsolve(L_free, rhs_adjusted[:, 0]),
                    spsolve(L_free, rhs_adjusted[:, 1]),
                ]
            )
        except Exception:
            break

        new_uv = uv.copy()
        new_uv[free_indices] = new_free
        uv = new_uv

    return uv


# --------------------------------------------------------------------------
# 歪み指標
# --------------------------------------------------------------------------
def _distortion_metrics(
    vertices3d: np.ndarray, uv: np.ndarray, faces: np.ndarray
) -> dict:
    edges = set()
    for i0, i1, i2 in faces:
        for a, b in ((i0, i1), (i1, i2), (i2, i0)):
            key = (int(a), int(b)) if a < b else (int(b), int(a))
            edges.add(key)

    ratios = []
    for i, j in edges:
        len3d = np.linalg.norm(vertices3d[i] - vertices3d[j])
        len2d = np.linalg.norm(uv[i] - uv[j])
        if len3d < _EPS:
            continue
        ratios.append(len2d / len3d)

    ratios_arr = np.array(ratios) if ratios else np.array([1.0])
    over_10pct = float(np.mean(np.abs(ratios_arr - 1.0) > 0.1)) if len(ratios) else 0.0

    area_3d = 0.0
    area_2d = 0.0
    for i0, i1, i2 in faces:
        p0, p1, p2 = vertices3d[i0], vertices3d[i1], vertices3d[i2]
        area_3d += 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        q0, q1, q2 = uv[i0], uv[i1], uv[i2]
        area_2d += 0.5 * abs((q1[0] - q0[0]) * (q2[1] - q0[1]) - (q2[0] - q0[0]) * (q1[1] - q0[1]))

    area_ratio = float(area_2d / area_3d) if area_3d > _EPS else 1.0

    return {
        "edge_length_ratio_max": float(np.max(ratios_arr)) if len(ratios) else 1.0,
        "edge_length_ratio_min": float(np.min(ratios_arr)) if len(ratios) else 1.0,
        "edge_length_ratio_mean": float(np.mean(ratios_arr)) if len(ratios) else 1.0,
        "edge_length_over_10pct_fraction": over_10pct,
        "area_ratio_2d_to_3d": area_ratio,
    }


# --------------------------------------------------------------------------
# 公開API
# --------------------------------------------------------------------------
def flatten_panel(
    mesh: trimesh.Trimesh,
    face_indices: np.ndarray,
    n_arap_iterations: int = 10,
) -> dict:
    """パネル(円盤位相の部分メッシュ)を2Dへ展開する。

    Args:
        mesh: 元メッシュ。
        face_indices: このパネルに属する面インデックス(mesh.faces基準)。
        n_arap_iterations: ARAP反復回数(5〜15を想定)。

    Returns:
        成功時:
            `{"vertices_2d": (V,2) ndarray, "vertices_3d": (V,3) ndarray
              (submesh頂点順の3D座標),
              "faces": (F,3) ndarray (submesh基準のローカルインデックス),
              "boundary_loop_2d": (B,2) ndarray (境界ループの2D座標、順序付き),
              "boundary_loop_indices": (B,) ndarray (submesh頂点インデックス),
              "distortion": {...歪み指標...}, "flatten_failed": False}`
        失敗時(円盤位相でない等): `{"flatten_failed": True, "reason": str}`
    """
    sub = _extract_submesh(mesh, face_indices)
    if sub is None:
        return {"flatten_failed": True, "reason": "empty_panel"}

    loops = _boundary_loops(sub)
    is_disk, reason = _is_disk_topology(sub, loops)
    if not is_disk:
        return {"flatten_failed": True, "reason": reason}

    vertices3d = np.asarray(sub.vertices, dtype=np.float64)
    faces = np.asarray(sub.faces, dtype=np.int64)

    if len(vertices3d) < 3 or len(faces) == 0:
        return {"flatten_failed": True, "reason": "degenerate_panel"}

    try:
        uv = _flatten_lscm_and_arap(vertices3d, faces, n_iterations=n_arap_iterations)
    except Exception as exc:  # 数値特異等への最終防御
        return {"flatten_failed": True, "reason": f"flatten_error: {exc}"}

    if not np.all(np.isfinite(uv)):
        return {"flatten_failed": True, "reason": "non_finite_uv"}

    distortion = _distortion_metrics(vertices3d, uv, faces)

    boundary_loop_indices = np.array(loops[0], dtype=np.int64)
    boundary_loop_2d = uv[boundary_loop_indices]

    return {
        "flatten_failed": False,
        "reason": "",
        "vertices_2d": uv,
        "vertices_3d": vertices3d,
        "faces": faces,
        "boundary_loop_2d": boundary_loop_2d,
        "boundary_loop_indices": boundary_loop_indices,
        "distortion": distortion,
    }
