"""FastAPIエントリポイント (SPEC.md §5 API仕様)。"""
from __future__ import annotations

import base64
import io
import json
import logging
import platform
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import config, sheet
from .generators.base import GenerationParams
from .generators.mock import MockGenerator
from .jobs import EXPORT_FORMATS, EXTRA_VIEW_LABELS, STATUS_COMPLETED, JobManager
from .preprocess import InvalidImageError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _hunyuan3d_usable() -> bool:
    """hunyuan3dジェネレータが動作可能か(モデルの実ロードはせず判定)。"""
    import importlib.util

    if importlib.util.find_spec("hy3dgen") is None:
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _build_generator():
    name = config.GENERATOR
    if name == "auto":
        if _hunyuan3d_usable():
            name = "hunyuan3d"
        else:
            name = "mock"
            logger.warning(
                "IMAGE3D_GENERATOR=auto: GPU/hy3dgen が利用できないため mock で起動します。"
                "アップロード画像は3D化されず、テスト用形状が返ります。"
            )
    if name == "mock":
        return MockGenerator()
    if name == "hunyuan3d":
        from .generators.hunyuan3d import Hunyuan3DGenerator

        return Hunyuan3DGenerator()
    if name == "pixal3d":
        # Pixal3Dは専用venv (.venv-pixal3d) での起動が前提のため、autoでは解決せず
        # IMAGE3D_GENERATOR=pixal3d の明示指定でのみ使用する (SPEC.md §3.3)。
        from .generators.pixal3d import Pixal3DGenerator

        return Pixal3DGenerator()
    raise ValueError(f"Unknown generator: {name}")


job_manager = JobManager(_build_generator())


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_dirs()
    job_manager.load_history()
    await job_manager.start_worker()
    yield
    await job_manager.stop_worker()


app = FastAPI(title="Image-3D", lifespan=lifespan)


def _parse_params(params_json: Optional[str]) -> GenerationParams:
    data = {}
    if params_json:
        try:
            data = json.loads(params_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"paramsのJSONが不正です: {exc}") from exc
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="paramsはJSONオブジェクトである必要があります。")

    defaults = GenerationParams(
        steps=config.DEFAULT_STEPS,
        guidance_scale=config.DEFAULT_GUIDANCE_SCALE,
        octree_resolution=config.DEFAULT_OCTREE_RESOLUTION,
        seed=None,
        remove_bg=config.DEFAULT_REMOVE_BG,
        target_height_mm=config.DEFAULT_TARGET_HEIGHT_MM,
        max_faces=config.DEFAULT_MAX_FACES,
        color_mode="none",
        n_colors=4,
        texture_mode="none",
    )

    steps = data.get("steps", defaults.steps)
    guidance_scale = data.get("guidance_scale", defaults.guidance_scale)
    octree_resolution = data.get("octree_resolution", defaults.octree_resolution)
    seed = data.get("seed", defaults.seed)
    remove_bg = data.get("remove_bg", defaults.remove_bg)
    target_height_mm = data.get("target_height_mm", defaults.target_height_mm)
    max_faces = data.get("max_faces", defaults.max_faces)
    color_mode = data.get("color_mode", defaults.color_mode)
    n_colors = data.get("n_colors", defaults.n_colors)
    texture_mode = data.get("texture_mode", defaults.texture_mode)

    if octree_resolution not in config.ALLOWED_OCTREE_RESOLUTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"octree_resolutionは{sorted(config.ALLOWED_OCTREE_RESOLUTIONS)}のいずれかである必要があります。",
        )
    if not isinstance(steps, int) or steps <= 0:
        raise HTTPException(status_code=400, detail="stepsは正の整数である必要があります。")
    if not isinstance(target_height_mm, (int, float)) or target_height_mm <= 0:
        raise HTTPException(status_code=400, detail="target_height_mmは正の数である必要があります。")
    if not isinstance(max_faces, int) or max_faces <= 0:
        raise HTTPException(status_code=400, detail="max_facesは正の整数である必要があります。")
    if color_mode not in ("none", "color4"):
        raise HTTPException(
            status_code=400, detail="color_modeは'none'または'color4'である必要があります。"
        )
    if not isinstance(n_colors, int) or not (2 <= n_colors <= 4):
        raise HTTPException(status_code=400, detail="n_colorsは2〜4の整数である必要があります。")
    if texture_mode not in ("none", "paint"):
        raise HTTPException(
            status_code=400, detail="texture_modeは'none'または'paint'である必要があります。"
        )

    return GenerationParams(
        steps=steps,
        guidance_scale=guidance_scale,
        octree_resolution=octree_resolution,
        seed=seed,
        remove_bg=remove_bg,
        target_height_mm=target_height_mm,
        max_faces=max_faces,
        color_mode=color_mode,
        n_colors=n_colors,
        texture_mode=texture_mode,
    )


async def _read_and_validate_upload(image: UploadFile, label: str) -> bytes:
    if image.content_type not in config.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"対応していないファイル形式です({label}: {image.content_type})。PNG/JPEG/WebPを使用してください。",
        )

    data = await image.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"ファイルサイズが上限({config.MAX_UPLOAD_BYTES // (1024 * 1024)}MB)を超えています({label})。",
        )
    if len(data) == 0:
        raise HTTPException(status_code=400, detail=f"空のファイルです({label})。")

    try:
        from .preprocess import load_and_validate_image

        load_and_validate_image(data, config.MAX_UPLOAD_BYTES)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=f"{label}: {exc}") from exc

    return data


@app.post("/api/jobs")
async def create_job(
    image: UploadFile = File(...),
    params: Optional[str] = Form(None),
    image_back: Optional[UploadFile] = File(None),
    image_left: Optional[UploadFile] = File(None),
    image_right: Optional[UploadFile] = File(None),
):
    data = await _read_and_validate_upload(image, "image")

    gen_params = _parse_params(params)

    # 追加ビュー(SPEC.md §3.8 / FR-9): 任意のmultipartフィールド
    # image_back / image_left / image_right を受け付ける。
    extra_uploads = {"back": image_back, "left": image_left, "right": image_right}
    extra_images: dict[str, bytes] = {}
    for view, upload in extra_uploads.items():
        if upload is None:
            continue
        extra_images[view] = await _read_and_validate_upload(upload, f"image_{view}")

    job = await job_manager.create_job(
        data,
        gen_params,
        original_filename=image.filename,
        extra_images=extra_images or None,
    )
    return {"job_id": job.job_id}


@app.get("/api/jobs")
async def list_jobs():
    return [job.to_dict() for job in job_manager.list_jobs()]


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    return job.to_dict()


@app.get("/api/jobs/{job_id}/input")
async def get_job_input(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    path = job.input_image_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="入力画像がまだありません。")
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/model.glb")
async def get_job_model_glb(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    if job.status != STATUS_COMPLETED:
        raise HTTPException(status_code=409, detail=f"ジョブは未完了です(status={job.status})。")
    path = job.model_path("glb")
    if not path.exists():
        raise HTTPException(status_code=404, detail="モデルファイルが見つかりません。")
    return FileResponse(path, media_type="model/gltf-binary", filename=f"{job_id}.glb")


_DOWNLOAD_MEDIA_TYPES = {
    "stl": "model/stl",
    "3mf": "model/3mf",
    "obj": "text/plain",
    "glb": "model/gltf-binary",
}


@app.get("/api/jobs/{job_id}/download")
async def download_job_model(job_id: str, format: str = "stl"):
    fmt = format.lower()
    if fmt not in EXPORT_FORMATS:
        raise HTTPException(
            status_code=400, detail=f"formatは{sorted(EXPORT_FORMATS)}のいずれかである必要があります。"
        )
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    if job.status != STATUS_COMPLETED:
        raise HTTPException(status_code=409, detail=f"ジョブは未完了です(status={job.status})。")

    # カラーモード時、3MFは色ごとに分割されたマルチオブジェクト版を返す
    if fmt == "3mf" and job.is_color_mode():
        color_path = job.model_color_3mf_path()
        if color_path.exists():
            return FileResponse(
                color_path,
                media_type=_DOWNLOAD_MEDIA_TYPES[fmt],
                filename=f"{job_id}_color.3mf",
            )

    path = job.model_path(fmt)
    if not path.exists():
        raise HTTPException(status_code=404, detail="モデルファイルが見つかりません。")
    return FileResponse(
        path,
        media_type=_DOWNLOAD_MEDIA_TYPES[fmt],
        filename=f"{job_id}.{fmt}",
    )


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    ok = job_manager.delete_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    return {"deleted": True}


# --- ぬいぐるみ型紙生成 (SPEC.md §3.12 / FR-13, Phase 4a) --------------------
# `server/pattern/` は純粋モジュール(numpy/scipy/trimeshのみに依存し、
# server内の他モジュールを一切importしない)。ここではジョブディレクトリ・
# パラメータバリデーション等アプリ固有の事情を扱う薄いアダプタとして接続する。
_PATTERN_MIN_PANELS = 4
_PATTERN_MAX_PANELS = 12
_PATTERN_DEFAULT_PANELS = 6
_PATTERN_MIN_SEAM_ALLOWANCE_MM = 1
_PATTERN_MAX_SEAM_ALLOWANCE_MM = 30
_PATTERN_DEFAULT_SEAM_ALLOWANCE_MM = 7.0
_PATTERN_ARAP_ITERATIONS = 10


def _pattern_vertex_colors(mesh):
    import numpy as np
    import trimesh

    visual = getattr(mesh, "visual", None)
    if isinstance(visual, trimesh.visual.ColorVisuals) and visual.kind == "vertex":
        return np.asarray(visual.vertex_colors)
    return None


@app.post("/api/jobs/{job_id}/pattern")
async def create_job_pattern(job_id: str, body: Optional[dict] = None):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    if job.status != STATUS_COMPLETED:
        raise HTTPException(status_code=409, detail=f"ジョブは未完了です(status={job.status})。")

    body = body or {}
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="リクエストボディはJSONオブジェクトである必要があります。")

    n_panels = body.get("n_panels", _PATTERN_DEFAULT_PANELS)
    use_colors = body.get("use_colors", True)
    smooth_iterations = body.get("smooth_iterations", 10)
    seam_allowance_mm = body.get("seam_allowance_mm", _PATTERN_DEFAULT_SEAM_ALLOWANCE_MM)

    if isinstance(n_panels, bool) or not isinstance(n_panels, int) or not (
        _PATTERN_MIN_PANELS <= n_panels <= _PATTERN_MAX_PANELS
    ):
        raise HTTPException(
            status_code=400,
            detail=f"n_panelsは{_PATTERN_MIN_PANELS}〜{_PATTERN_MAX_PANELS}の整数である必要があります。",
        )
    if not isinstance(use_colors, bool):
        raise HTTPException(status_code=400, detail="use_colorsはbool値である必要があります。")
    if isinstance(smooth_iterations, bool) or not isinstance(smooth_iterations, int) or not (
        0 <= smooth_iterations <= 50
    ):
        raise HTTPException(status_code=400, detail="smooth_iterationsは0〜50の整数である必要があります。")
    if isinstance(seam_allowance_mm, bool) or not isinstance(seam_allowance_mm, (int, float)) or not (
        _PATTERN_MIN_SEAM_ALLOWANCE_MM <= seam_allowance_mm <= _PATTERN_MAX_SEAM_ALLOWANCE_MM
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"seam_allowance_mmは{_PATTERN_MIN_SEAM_ALLOWANCE_MM}〜"
                f"{_PATTERN_MAX_SEAM_ALLOWANCE_MM}の数値である必要があります。"
            ),
        )
    seam_allowance_mm = float(seam_allowance_mm)

    model_path = job.model_path("glb")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="モデルファイルが見つかりません。")

    def _run_pattern() -> dict:
        import numpy as np
        import trimesh

        from .pattern import (
            build_pattern_svg,
            build_preview_mesh,
            flatten_panel,
            panel_stats,
            prepare_mesh,
            segment_panels,
        )

        loaded = trimesh.load(model_path, file_type="glb", process=False)
        if isinstance(loaded, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            )
        else:
            mesh = loaded

        vertex_colors = _pattern_vertex_colors(mesh)

        prepared = prepare_mesh(mesh, smooth_iterations=smooth_iterations)
        prepared_colors = _pattern_vertex_colors(prepared) if vertex_colors is not None else None

        labels = segment_panels(
            prepared,
            n_panels=n_panels,
            vertex_colors=prepared_colors,
            use_colors=use_colors,
            seed=0,
        )
        stats = panel_stats(prepared, labels)
        preview_mesh = build_preview_mesh(prepared, labels)

        preview_data = preview_mesh.export(file_type="glb")
        job.pattern_preview_glb_path().write_bytes(preview_data)

        # 平坦化(Phase 4b): パネルごとにLSCM+ARAPで2D展開する。円盤位相でない
        # パネルは flatten_panel が例外を投げず flatten_failed=True を返すため、
        # 他パネルの処理を続行できる。
        panels_2d = []
        stats_by_id = {s["panel_id"]: s for s in stats}
        for panel_id in sorted(stats_by_id):
            face_idx = np.where(labels == panel_id)[0]
            flat_result = flatten_panel(prepared, face_idx, n_arap_iterations=_PATTERN_ARAP_ITERATIONS)
            flat_result["panel_id"] = int(panel_id)
            panels_2d.append(flat_result)

            panel_stat_entry = stats_by_id[panel_id]
            panel_stat_entry["flatten_failed"] = bool(flat_result.get("flatten_failed"))
            if flat_result.get("flatten_failed"):
                panel_stat_entry["flatten_failed_reason"] = flat_result.get("reason", "")
            else:
                panel_stat_entry["distortion"] = flat_result["distortion"]

        model_height_mm = float(job.stats.get("bbox_mm", [0, 0, 0])[2] or job.params.get("target_height_mm", 0) or 0)
        model_name = job.original_filename or job_id

        svg_text = build_pattern_svg(
            panels_2d,
            seam_allowance_mm=seam_allowance_mm,
            label_prefix="P",
            model_name=model_name,
            model_height_mm=model_height_mm,
        )
        job.pattern_svg_path().write_text(svg_text, encoding="utf-8")

        n_flatten_ok = sum(1 for p in panels_2d if not p.get("flatten_failed"))

        result = {
            "job_id": job_id,
            "n_panels_requested": n_panels,
            "n_panels_actual": len(stats),
            "use_colors": use_colors,
            "smooth_iterations": smooth_iterations,
            "seam_allowance_mm": seam_allowance_mm,
            "n_panels_flattened": n_flatten_ok,
            "panels": list(stats_by_id.values()),
        }
        job.pattern_json_path().write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result

    import asyncio

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, _run_pattern)
    except Exception as exc:
        logger.exception("Pattern generation failed for job %s", job_id)
        raise HTTPException(status_code=500, detail=f"型紙生成に失敗しました: {exc}") from exc

    return result


@app.get("/api/jobs/{job_id}/pattern.json")
async def get_job_pattern_json(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    path = job.pattern_json_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="型紙がまだ生成されていません。")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/jobs/{job_id}/pattern_preview.glb")
async def get_job_pattern_preview_glb(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    path = job.pattern_preview_glb_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="型紙プレビューがまだ生成されていません。")
    return FileResponse(
        path, media_type="model/gltf-binary", filename=f"{job_id}_pattern_preview.glb"
    )


@app.get("/api/jobs/{job_id}/pattern.svg")
async def get_job_pattern_svg(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    path = job.pattern_svg_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="型紙SVGがまだ生成されていません。")
    return FileResponse(path, media_type="image/svg+xml", filename=f"{job_id}_pattern.svg")


@app.post("/api/sheet/split")
async def split_sheet(image: UploadFile = File(...)):
    """キャラクターシート画像から被写体パネルを自動検出する (SPEC.md §3.8 / FR-9)。

    ジョブは作成しない同期API。数秒で結果を返す。
    """
    data = await _read_and_validate_upload(image, "image")

    from .preprocess import load_and_validate_image

    pil_image = load_and_validate_image(data, config.MAX_UPLOAD_BYTES)

    import asyncio

    loop = asyncio.get_running_loop()
    panels = await loop.run_in_executor(None, sheet.split_sheet, pil_image)
    views = sheet.suggested_views(len(panels))

    result = []
    for idx, (panel, suggested_view) in enumerate(zip(panels, views)):
        buf = io.BytesIO()
        panel.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        result.append(
            {
                "index": idx,
                "image_b64": image_b64,
                "suggested_view": suggested_view,
            }
        )

    return {"panels": result}


@app.get("/api/health")
async def health():
    gpu_info = {"available": False}
    try:
        import torch

        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "vram_total_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
                ),
            }
    except ImportError:
        pass

    from . import texture

    return {
        "status": "ok",
        "generator": job_manager.generator.name,
        "python_version": platform.python_version(),
        "gpu": gpu_info,
        "texgen_available": texture.is_available(),
    }


# --- 静的フロントエンド配信 (SPEC.md §5 `GET /`) -----------------------------
app.mount("/", StaticFiles(directory=str(config.WEB_DIR), html=True), name="web")
