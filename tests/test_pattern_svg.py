"""server/pattern/svg.py の単体テスト (Phase 4b / SPEC.md §3.12 FR-13)。

`server/pattern/` は純粋モジュール(server内の他モジュールをimportしない)。
ここではXMLパースにより構造(viewBox実寸・パネル数・縫い代パス・合印数)を
検証する。`xml.etree.ElementTree`(標準ライブラリ)はテスト側でのみ使用し、
`server/pattern/svg.py` 自体はXMLライブラリに依存しない(文字列組み立て)。
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

from server.pattern.flatten import flatten_panel
from server.pattern.segment import segment_panels
from server.pattern.svg import _detect_seams, _ensure_ccw, _offset_polygon, _polygon_area, build_pattern_svg

_SVG_NS = {"svg": "http://www.w3.org/2000/svg"}


def _flatten_all_panels(mesh: trimesh.Trimesh, n_panels: int, seed: int = 0) -> list[dict]:
    labels = segment_panels(mesh, n_panels=n_panels, seed=seed)
    panels = []
    for panel_id in np.unique(labels):
        face_idx = np.where(labels == panel_id)[0]
        result = flatten_panel(mesh, face_idx, n_arap_iterations=8)
        result["panel_id"] = int(panel_id)
        panels.append(result)
    return panels


@pytest.fixture()
def icosphere_panels():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=50.0)
    return mesh, _flatten_all_panels(mesh, n_panels=6, seed=0)


# --------------------------------------------------------------------------
# 実寸viewBox・基本構造
# --------------------------------------------------------------------------
def test_svg_viewbox_is_mm_real_scale(icosphere_panels):
    _mesh, panels = icosphere_panels
    svg = build_pattern_svg(panels, seam_allowance_mm=7.0, model_name="test", model_height_mm=100.0)

    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert root.attrib["width"].endswith("mm")
    assert root.attrib["height"].endswith("mm")

    view_box = [float(v) for v in root.attrib["viewBox"].split()]
    width_mm = float(root.attrib["width"].replace("mm", ""))
    height_mm = float(root.attrib["height"].replace("mm", ""))
    assert view_box[2] == pytest.approx(width_mm, rel=1e-3)
    assert view_box[3] == pytest.approx(height_mm, rel=1e-3)

    # 100mmモデルの複数パネル型紙が数百mm規模になるのは(パッキング余白込みで)
    # 妥当。極端に小さい(mm単位を見失っている)/大きすぎることがないか検査する。
    assert 10 < width_mm < 5000
    assert 10 < height_mm < 5000


def test_svg_has_group_per_valid_panel(icosphere_panels):
    _mesh, panels = icosphere_panels
    svg = build_pattern_svg(panels, seam_allowance_mm=7.0)
    root = ET.fromstring(svg)

    panel_groups = root.findall(".//svg:g[@class='panel']", _SVG_NS)
    n_valid = len([p for p in panels if not p.get("flatten_failed")])
    assert len(panel_groups) == n_valid
    assert n_valid == len(panels)  # icosphereの6分割は全パネル円盤位相のはず


def test_svg_is_valid_xml_for_various_panel_counts():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=30.0)
    for n_panels in (4, 8):
        panels = _flatten_all_panels(mesh, n_panels=n_panels, seed=1)
        svg = build_pattern_svg(panels, seam_allowance_mm=5.0)
        root = ET.fromstring(svg)  # パース失敗時は例外
        assert root.tag.endswith("svg")


# --------------------------------------------------------------------------
# 縫い代パスは本体パスの外側
# --------------------------------------------------------------------------
def test_seam_allowance_offset_is_outside_body_polygon(icosphere_panels):
    _mesh, panels = icosphere_panels
    for panel in panels:
        if panel.get("flatten_failed"):
            continue
        loop = _ensure_ccw(np.asarray(panel["boundary_loop_2d"]))
        offset = _offset_polygon(loop, 7.0)
        body_area = abs(_polygon_area(loop))
        offset_area = abs(_polygon_area(offset))
        assert offset_area > body_area


def test_svg_seam_allowance_paths_present_and_dashed(icosphere_panels):
    _mesh, panels = icosphere_panels
    svg = build_pattern_svg(panels, seam_allowance_mm=7.0)
    assert svg.count("stroke-dasharray") == len(panels)  # 各パネル1本の縫い代破線


# --------------------------------------------------------------------------
# 合印: シームごとに両パネルで同数
# --------------------------------------------------------------------------
def test_notches_appear_in_equal_counts_on_both_sides_of_each_seam(icosphere_panels):
    _mesh, panels = icosphere_panels
    seams = _detect_seams(panels)
    assert len(seams) > 0

    svg = build_pattern_svg(panels, seam_allowance_mm=7.0)
    root = ET.fromstring(svg)
    notch_groups = root.findall(".//svg:g[@class='notch']", _SVG_NS)
    assert len(notch_groups) > 0

    # notchのテキストはseam_idを表す。seam_idごとの出現回数は偶数
    # (両パネルの対応点にそれぞれ同数配置)であるはず。
    from collections import Counter

    seam_counts = Counter()
    for g in notch_groups:
        text_el = g.find("svg:text", _SVG_NS)
        assert text_el is not None
        seam_counts[text_el.text] += 1

    for seam_id, count in seam_counts.items():
        assert count % 2 == 0, f"seam {seam_id} has odd notch count {count}"


def test_detect_seams_matches_face_adjacency_panel_pairs():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=50.0)
    labels = segment_panels(mesh, n_panels=6, seed=0)

    # 面の双対グラフから真のパネル隣接ペアを求める
    true_adjacent_pairs = set()
    for a, b in mesh.face_adjacency:
        la, lb = int(labels[a]), int(labels[b])
        if la != lb:
            true_adjacent_pairs.add((min(la, lb), max(la, lb)))

    panels = _flatten_all_panels(mesh, n_panels=6, seed=0)
    seams = _detect_seams(panels)
    detected_pairs = {(min(s["panel_a"], s["panel_b"]), max(s["panel_a"], s["panel_b"])) for s in seams}

    assert detected_pairs == true_adjacent_pairs


# --------------------------------------------------------------------------
# パネル番号ラベル・布目線・凡例
# --------------------------------------------------------------------------
def test_svg_contains_panel_labels_and_grainlines(icosphere_panels):
    _mesh, panels = icosphere_panels
    svg = build_pattern_svg(panels, seam_allowance_mm=7.0, label_prefix="P")
    root = ET.fromstring(svg)

    labels_found = [
        t.text
        for t in root.findall(".//svg:g[@class='panel']/svg:text", _SVG_NS)
        if t.text and t.text.startswith("P")
    ]
    expected_labels = {f"P{p['panel_id'] + 1}" for p in panels if not p.get("flatten_failed")}
    assert set(labels_found) >= expected_labels

    grainlines = root.findall(".//svg:g[@class='grainline']", _SVG_NS)
    assert len(grainlines) == len([p for p in panels if not p.get("flatten_failed")])


def test_svg_contains_legend(icosphere_panels):
    _mesh, panels = icosphere_panels
    svg = build_pattern_svg(
        panels, seam_allowance_mm=7.0, model_name="momo", model_height_mm=100.0
    )
    root = ET.fromstring(svg)
    legend = root.find(".//svg:g[@class='legend']", _SVG_NS)
    assert legend is not None
    legend_text = "".join(t.text or "" for t in legend.findall("svg:text", _SVG_NS))
    assert "momo" in legend_text
    assert "100" in legend_text
    assert "7" in legend_text


# --------------------------------------------------------------------------
# 失敗パネルの除外
# --------------------------------------------------------------------------
def test_svg_skips_failed_panels_without_raising():
    failed_panel = {"panel_id": 0, "flatten_failed": True, "reason": "test"}
    svg = build_pattern_svg([failed_panel], seam_allowance_mm=7.0)
    root = ET.fromstring(svg)
    panel_groups = root.findall(".//svg:g[@class='panel']", _SVG_NS)
    assert len(panel_groups) == 0
    legend = root.find(".//svg:g[@class='legend']", _SVG_NS)
    legend_text = "".join(t.text or "" for t in legend.findall("svg:text", _SVG_NS))
    assert "1" in legend_text  # 失敗パネル数の注記


def test_svg_empty_panels_list_still_produces_valid_svg():
    svg = build_pattern_svg([], seam_allowance_mm=7.0)
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
