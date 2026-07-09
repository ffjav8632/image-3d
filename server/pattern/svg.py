"""実寸SVG型紙出力 (SPEC.md §3.12 / FR-13 の4b部分)。

`flatten.flatten_panel()` が返す平坦化済みパネル(2D頂点・境界ループ・
3D頂点・歪み指標)のリストから、mm実寸のSVG型紙を組み立てる。

含まれる要素:
    - パネルごとの縫い線(実線、境界ループ)+縫い代線(破線、外側オフセット)
    - シーム(隣接パネルの共有境界)ごとの合印(ノッチ、対応する2パネルの
      両側に同一記号)
    - パネル番号ラベル・布目線(縦の両矢印)
    - 凡例(モデル名・高さ・縫い代・シーム対応表)

パネル配置は単純なシェルフ法(左上から右方向に敷き詰め、行の最大高さを
超えたら次の行へ)による矩形パッキング。

縫い代オフセットは境界ポリゴンの頂点法線(隣接エッジの外向き法線の平均)
方向へのオフセット+短エッジ間引きによる簡易クリーンアップ(自己交差の
完全排除は保証しないが、実用レベルの型紙には十分)。

このモジュールは純粋モジュール(server/DEVELOPMENT_POLICY.md §3.5)。
依存は numpy / scipy / trimesh のみ(SVGはXML文字列を直接組み立てる。
lxml等のXMLライブラリはアプリ/テスト側でのみ使用し、ここでは使わない)。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

_EPS = 1e-9

# シーム(隣接パネル境界)対応の3D距離許容誤差(mm)。平坦化前の3D座標が
# 元メッシュの共有頂点由来のため、通常は厳密一致に近いが、浮動小数点誤差や
# パネル抽出の丸めを見込んで許容幅を持たせる。
_SEAM_MATCH_TOLERANCE_MM = 0.5

# シェルフパッキングのパネル間余白(mm)。
_PACK_MARGIN_MM = 15.0

_SVG_NS = "http://www.w3.org/2000/svg"


# --------------------------------------------------------------------------
# ユーティリティ
# --------------------------------------------------------------------------
def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ensure_ccw(points: np.ndarray) -> np.ndarray:
    if _polygon_area(points) < 0:
        return points[::-1].copy()
    return points.copy()


def _offset_polygon(points: np.ndarray, distance: float, min_edge_len: float = 0.3) -> np.ndarray:
    """境界ポリゴンを外側(反時計回りを前提に法線=左向き90度回転の逆)へ
    `distance` だけオフセットした簡易オフセットポリゴンを返す。

    手法: 各頂点で隣接する2エッジの外向き法線(エッジ法線の平均、正規化)
    方向にオフセットする(マイタ近似)。短すぎるエッジは事前に間引いて
    ジグザグによる自己交差を軽減する(簡易クリーンアップ)。
    """
    pts = _ensure_ccw(points)
    pts = _dedup_short_edges(pts, min_edge_len)
    n = len(pts)
    if n < 3:
        return pts

    offset_pts = np.zeros_like(pts)
    for i in range(n):
        prev_pt = pts[(i - 1) % n]
        cur_pt = pts[i]
        next_pt = pts[(i + 1) % n]

        e1 = cur_pt - prev_pt
        e2 = next_pt - cur_pt
        n1 = _outward_normal(e1)
        n2 = _outward_normal(e2)
        normal = n1 + n2
        norm_len = np.linalg.norm(normal)
        if norm_len < _EPS:
            normal = n1
            norm_len = np.linalg.norm(n1)
        if norm_len < _EPS:
            offset_pts[i] = cur_pt
            continue
        normal = normal / norm_len

        # マイタ長の暴走を防ぐため、内積が小さい(鋭角)場合はクランプする
        cos_half = float(np.dot(normal, _outward_normal(e1)))
        miter_scale = 1.0 / max(cos_half, 0.5)
        miter_scale = min(miter_scale, 3.0)

        offset_pts[i] = cur_pt + normal * distance * miter_scale

    return offset_pts


def _outward_normal(edge: np.ndarray) -> np.ndarray:
    """CCWポリゴンの辺ベクトルから外向き法線(右手系で辺を時計回りに90度
    回転させた向き)を返す。"""
    length = np.linalg.norm(edge)
    if length < _EPS:
        return np.zeros(2)
    direction = edge / length
    return np.array([direction[1], -direction[0]])


def _dedup_short_edges(pts: np.ndarray, min_edge_len: float) -> np.ndarray:
    if len(pts) < 4:
        return pts
    kept = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - kept[-1]) >= min_edge_len:
            kept.append(p)
    if len(kept) >= 2 and np.linalg.norm(kept[0] - kept[-1]) < min_edge_len:
        kept.pop()
    if len(kept) < 3:
        return pts
    return np.array(kept)


def _polygon_bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


# --------------------------------------------------------------------------
# シェルフパッキング
# --------------------------------------------------------------------------
def _shelf_pack(
    boxes: list[tuple[float, float]], margin: float = _PACK_MARGIN_MM, max_width: float = 900.0
) -> tuple[list[tuple[float, float]], float, float]:
    """(width, height) のリストをシェルフ法でパッキングし、各アイテムの
    左下オフセット(x, y)のリストと全体のサイズ(width, height)を返す。
    """
    x_cursor = margin
    y_cursor = margin
    shelf_height = 0.0
    offsets: list[tuple[float, float]] = []
    total_width = margin
    total_height = margin

    for w, h in boxes:
        if x_cursor + w + margin > max_width and x_cursor > margin:
            # 次の行へ
            x_cursor = margin
            y_cursor += shelf_height + margin
            shelf_height = 0.0
        offsets.append((x_cursor, y_cursor))
        x_cursor += w + margin
        shelf_height = max(shelf_height, h)
        total_width = max(total_width, x_cursor)
        total_height = max(total_height, y_cursor + shelf_height + margin)

    return offsets, total_width, total_height


# --------------------------------------------------------------------------
# シーム(隣接パネル境界)対応の検出
# --------------------------------------------------------------------------
def _detect_seams(panels: list[dict]) -> list[dict]:
    """パネル間で3D座標が近い境界点同士を突き合わせ、シーム(隣接境界)を
    検出する。

    Returns:
        `{"seam_id": int, "panel_a": panel_id, "panel_b": panel_id,
          "points_a": (N,2) ndarray, "points_b": (N,2) ndarray}` のリスト。
        `points_a`/`points_b` は3D位置で対応付けられた同数の2D座標列
        (それぞれのパネルの `boundary_loop_2d` 上の座標)。
    """
    valid_panels = [p for p in panels if not p.get("flatten_failed")]
    seams = []
    seam_id = 0

    for a_idx in range(len(valid_panels)):
        for b_idx in range(a_idx + 1, len(valid_panels)):
            panel_a = valid_panels[a_idx]
            panel_b = valid_panels[b_idx]

            loop_a_idx = panel_a["boundary_loop_indices"]
            loop_b_idx = panel_b["boundary_loop_indices"]
            verts3d_a = panel_a["vertices_3d"][loop_a_idx]
            verts3d_b = panel_b["vertices_3d"][loop_b_idx]

            if len(verts3d_a) == 0 or len(verts3d_b) == 0:
                continue

            # 総当たりで近接点を対応付け(パネル境界は通常小規模なため許容)
            matches_a = []
            matches_b = []
            used_b = set()
            for ia, p3 in enumerate(verts3d_a):
                dists = np.linalg.norm(verts3d_b - p3, axis=1)
                jb = int(np.argmin(dists))
                if dists[jb] <= _SEAM_MATCH_TOLERANCE_MM and jb not in used_b:
                    matches_a.append(ia)
                    matches_b.append(jb)
                    used_b.add(jb)

            if len(matches_a) < 2:
                continue

            uv_a = panel_a["boundary_loop_2d"][matches_a]
            uv_b = panel_b["boundary_loop_2d"][matches_b]

            seams.append(
                {
                    "seam_id": seam_id,
                    "panel_a": panel_a["panel_id"],
                    "panel_b": panel_b["panel_id"],
                    "points_a": uv_a,
                    "points_b": uv_b,
                }
            )
            seam_id += 1

    return seams


def _select_notch_points(points: np.ndarray, n_notches: int) -> np.ndarray:
    """境界点列から等間隔に近いn_notches個の代表点(インデックス)を選ぶ。"""
    n = len(points)
    if n == 0:
        return np.array([], dtype=np.int64)
    if n <= n_notches:
        return np.arange(n)
    idx = np.linspace(0, n - 1, n_notches).astype(np.int64)
    return np.unique(idx)


def _notch_direction(loop_points: np.ndarray, idx: int) -> np.ndarray:
    """境界ループ上の点における外向き法線方向(合印チックの向き)を返す。"""
    n = len(loop_points)
    prev_pt = loop_points[(idx - 1) % n]
    next_pt = loop_points[(idx + 1) % n]
    edge = next_pt - prev_pt
    normal = _outward_normal(edge)
    norm_len = np.linalg.norm(normal)
    if norm_len < _EPS:
        return np.array([0.0, 1.0])
    return normal / norm_len


# --------------------------------------------------------------------------
# SVG要素組み立て
# --------------------------------------------------------------------------
def _points_to_path(points: np.ndarray, closed: bool = True) -> str:
    if len(points) == 0:
        return ""
    parts = [f"M {points[0][0]:.3f},{points[0][1]:.3f}"]
    for p in points[1:]:
        parts.append(f"L {p[0]:.3f},{p[1]:.3f}")
    if closed:
        parts.append("Z")
    return " ".join(parts)


def _grainline_svg(bbox: tuple[float, float, float, float]) -> str:
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    margin = (y1 - y0) * 0.12
    top = y0 + margin
    bottom = y1 - margin
    if bottom <= top:
        top, bottom = y0, y1
    arrow = 3.0
    return (
        f'<g class="grainline" stroke="#1565c0" stroke-width="0.6" fill="none">'
        f'<line x1="{cx:.3f}" y1="{top:.3f}" x2="{cx:.3f}" y2="{bottom:.3f}" />'
        f'<path d="M {cx - arrow:.3f},{top + arrow:.3f} L {cx:.3f},{top:.3f} L {cx + arrow:.3f},{top + arrow:.3f}" />'
        f'<path d="M {cx - arrow:.3f},{bottom - arrow:.3f} L {cx:.3f},{bottom:.3f} L {cx + arrow:.3f},{bottom - arrow:.3f}" />'
        f"</g>"
    )


def _notch_svg(point: np.ndarray, direction: np.ndarray, seam_id: int, length: float = 4.0) -> str:
    p0 = point
    p1 = point + direction * length
    mid = point + direction * (length * 0.55)
    return (
        f'<g class="notch">'
        f'<line x1="{p0[0]:.3f}" y1="{p0[1]:.3f}" x2="{p1[0]:.3f}" y2="{p1[1]:.3f}" '
        f'stroke="#c62828" stroke-width="0.5" />'
        f'<text x="{mid[0]:.3f}" y="{mid[1]:.3f}" font-size="2.6" fill="#c62828" '
        f'text-anchor="middle">{seam_id}</text>'
        f"</g>"
    )


def _panel_stats_svg(distortion: Optional[dict], x: float, y: float) -> str:
    if not distortion:
        return ""
    over_10pct = distortion.get("edge_length_over_10pct_fraction", 0.0)
    color = "#c62828" if over_10pct > 0.2 else "#37474f"
    text = f"歪み±10%超: {over_10pct * 100:.0f}%"
    return (
        f'<text x="{x:.3f}" y="{y:.3f}" font-size="3" fill="{color}">{text}</text>'
    )


# --------------------------------------------------------------------------
# 公開API
# --------------------------------------------------------------------------
def build_pattern_svg(
    panels_2d: list[dict],
    seam_allowance_mm: float = 7.0,
    label_prefix: str = "P",
    model_name: str = "",
    model_height_mm: float = 0.0,
) -> str:
    """平坦化済みパネルのリストから実寸SVG型紙を組み立てる。

    Args:
        panels_2d: `flatten_panel()` の戻り値に `panel_id` を加えた辞書の
            リスト。`flatten_failed=True` のパネルはスキップし、SVGには
            含めない(呼び出し側が別途警告表示することを想定)。
        seam_allowance_mm: 縫い代幅(mm)。
        label_prefix: パネル番号ラベルの接頭辞("P" → "P1", "P2", ...)。
        model_name: 凡例に表示するモデル名(任意)。
        model_height_mm: 凡例に表示するモデル高さ(mm、任意)。

    Returns:
        SVG文字列(viewBoxはmm単位、width/heightに `mm` 単位を明記)。
    """
    valid_panels = [p for p in panels_2d if not p.get("flatten_failed")]

    seams = _detect_seams(valid_panels)
    # パネルごとのシームID一覧(凡例テーブル用)
    panel_seam_ids: dict[int, list[int]] = {}
    for seam in seams:
        panel_seam_ids.setdefault(seam["panel_a"], []).append(seam["seam_id"])
        panel_seam_ids.setdefault(seam["panel_b"], []).append(seam["seam_id"])

    # 各パネルの境界ループ(縫い線)+縫い代線を準備し、パッキング用の
    # バウンディングボックスサイズを求める。
    prepared = []
    for panel in valid_panels:
        loop = _ensure_ccw(np.asarray(panel["boundary_loop_2d"], dtype=np.float64))
        offset_loop = _offset_polygon(loop, seam_allowance_mm)
        bbox_seam = _polygon_bbox(offset_loop)
        width = bbox_seam[2] - bbox_seam[0]
        height = bbox_seam[3] - bbox_seam[1]
        prepared.append(
            {
                "panel_id": panel["panel_id"],
                "loop": loop,
                "offset_loop": offset_loop,
                "bbox_seam": bbox_seam,
                "width": width,
                "height": height,
                "distortion": panel.get("distortion"),
            }
        )

    boxes = [(p["width"], p["height"]) for p in prepared]
    offsets, total_width, total_height = _shelf_pack(boxes)

    body_parts: list[str] = []

    for panel, (ox, oy) in zip(prepared, offsets):
        bx0, by0, _, _ = panel["bbox_seam"]
        shift = np.array([ox - bx0, oy - by0])

        loop_shifted = panel["loop"] + shift
        offset_shifted = panel["offset_loop"] + shift
        bbox_shifted = (
            bx0 + shift[0],
            by0 + shift[1],
            panel["bbox_seam"][2] + shift[0],
            panel["bbox_seam"][3] + shift[1],
        )

        panel_id = panel["panel_id"]
        label = f"{label_prefix}{panel_id + 1}"

        group_parts = [f'<g class="panel" data-panel-id="{panel_id}">']
        group_parts.append(
            f'<path d="{_points_to_path(offset_shifted)}" fill="none" '
            f'stroke="#455a64" stroke-width="0.4" stroke-dasharray="2,1.5" />'
        )
        group_parts.append(
            f'<path d="{_points_to_path(loop_shifted)}" fill="none" '
            f'stroke="#212121" stroke-width="0.5" />'
        )
        group_parts.append(_grainline_svg(_polygon_bbox(loop_shifted)))

        label_x = (bbox_shifted[0] + bbox_shifted[2]) / 2.0
        label_y = bbox_shifted[1] + 5.0
        group_parts.append(
            f'<text x="{label_x:.3f}" y="{label_y:.3f}" font-size="5" '
            f'font-weight="bold" text-anchor="middle" fill="#000000">{label}</text>'
        )
        group_parts.append(_panel_stats_svg(panel["distortion"], label_x, label_y + 4.5))

        seam_note = ",".join(f"S{sid}" for sid in sorted(set(panel_seam_ids.get(panel_id, []))))
        if seam_note:
            group_parts.append(
                f'<text x="{label_x:.3f}" y="{bbox_shifted[3] - 2.0:.3f}" font-size="2.6" '
                f'text-anchor="middle" fill="#546e7a">seam: {seam_note}</text>'
            )

        group_parts.append("</g>")
        panel["_render_shift"] = shift
        panel["_loop_shifted"] = loop_shifted
        body_parts.append("".join(group_parts))

    # 合印(ノッチ): シームごとに両パネルの対応点へ、対応する平行移動を
    # 適用した座標へ描画する。
    shift_by_panel = {p["panel_id"]: p["_render_shift"] for p in prepared}
    loop_by_panel = {p["panel_id"]: p["_loop_shifted"] for p in prepared}

    for seam in seams:
        pa = seam["panel_a"]
        pb = seam["panel_b"]
        if pa not in shift_by_panel or pb not in shift_by_panel:
            continue
        n_notches = min(4, max(2, len(seam["points_a"]) // 6 + 2))
        idx_sel = _select_notch_points(seam["points_a"], n_notches)

        loop_a_shifted = loop_by_panel[pa]
        loop_b_shifted = loop_by_panel[pb]

        for i in idx_sel:
            pt_a = seam["points_a"][i] + shift_by_panel[pa]
            pt_b = seam["points_b"][i] + shift_by_panel[pb]

            dir_a = _notch_direction_from_shifted(loop_a_shifted, pt_a)
            dir_b = _notch_direction_from_shifted(loop_b_shifted, pt_b)

            body_parts.append(_notch_svg(pt_a, dir_a, seam["seam_id"]))
            body_parts.append(_notch_svg(pt_b, dir_b, seam["seam_id"]))

    # 凡例
    legend_x = _PACK_MARGIN_MM
    legend_y = total_height + 6.0
    legend_lines = [
        f"モデル: {model_name}" if model_name else "モデル: (未指定)",
        f"高さ: {model_height_mm:.1f} mm" if model_height_mm else "高さ: -",
        f"縫い代: {seam_allowance_mm:.1f} mm",
        f"パネル数: {len(valid_panels)} / シーム数: {len(seams)}",
    ]
    if len(valid_panels) < len(panels_2d):
        n_failed = len(panels_2d) - len(valid_panels)
        legend_lines.append(f"※平坦化失敗パネル: {n_failed}(型紙に含まれません)")

    legend_svg_parts = ['<g class="legend" font-size="3" fill="#212121">']
    for i, line in enumerate(legend_lines):
        legend_svg_parts.append(
            f'<text x="{legend_x:.3f}" y="{legend_y + i * 4.0:.3f}">{_xml_escape(line)}</text>'
        )
    legend_svg_parts.append("</g>")
    body_parts.append("".join(legend_svg_parts))

    total_height_with_legend = legend_y + len(legend_lines) * 4.0 + 6.0

    svg = (
        f'<svg xmlns="{_SVG_NS}" width="{total_width:.3f}mm" height="{total_height_with_legend:.3f}mm" '
        f'viewBox="0 0 {total_width:.3f} {total_height_with_legend:.3f}">'
        f"{''.join(body_parts)}"
        f"</svg>"
    )
    return svg


def _notch_direction_from_shifted(loop_shifted: np.ndarray, point: np.ndarray) -> np.ndarray:
    """シフト後の境界ループ上で `point` に最も近い頂点の外向き法線を返す。"""
    dists = np.linalg.norm(loop_shifted - point, axis=1)
    idx = int(np.argmin(dists))
    return _notch_direction(loop_shifted, idx)


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
