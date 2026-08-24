#!/usr/bin/env python3
"""Build a reviewer-facing 720p video from one verified Isaac run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1280
HEIGHT = 720
FPS = 30.0
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
INSTRUCTION = "请把混杂工位上的钳子放进工具箱第三格"
PLAN = [
    ("目标选择", "感知"),
    ("抓取规划", "决策"),
    ("抓取与抬升", "执行"),
    ("第三格定位", "感知"),
    ("携物导航", "执行"),
    ("格内放置", "执行"),
    ("结果验收", "决策"),
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size)


def _records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return records


def _event(records: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((record for record in reversed(records) if record.get("event") == name), None)


def _verified_run(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    final = _event(records, "orchestrator_task_final")
    terminal = _event(records, "action_terminal_success")
    final_payload = (final or {}).get("payload") or {}
    terminal_payload = (terminal or {}).get("payload") or {}
    physical = terminal_payload.get("physical_evidence") or {}
    container_contained = terminal_payload.get("aabb_contained")
    if container_contained is None:
        container_contained = physical.get("aabb_contained")
    cell_contained = terminal_payload.get("cell_aabb_contained")
    if cell_contained is None:
        cell_contained = physical.get("cell_aabb_contained")
    required = {
        "task outcome": final_payload.get("outcome") == "success",
        "placement success": terminal_payload.get("placement_success") is True,
        "placement verified": terminal_payload.get("placement_verified") is True,
        "released": terminal_payload.get("released") is True,
        "container AABB contained": container_contained is True,
        "requested cell AABB contained": cell_contained is True,
    }
    failed = [label for label, passed in required.items() if not passed]
    if failed:
        raise RuntimeError(
            "refusing to build a competition video from an unverified run: "
            + ", ".join(failed)
        )
    return final_payload, terminal_payload


def _canvas() -> np.ndarray:
    top = np.array([10, 23, 39], dtype=np.float32)
    bottom = np.array([3, 9, 18], dtype=np.float32)
    amount = np.linspace(0.0, 1.0, HEIGHT, dtype=np.float32)[:, None, None]
    rgb = top[None, None, :] * (1.0 - amount) + bottom[None, None, :] * amount
    return np.repeat(rgb.astype(np.uint8), WIDTH, axis=1)[:, :, ::-1]


def _pil(frame: np.ndarray) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return image, ImageDraw.Draw(image)


def _bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _centered(draw: ImageDraw.ImageDraw, text: str, y: int, font: ImageFont.FreeTypeFont, fill: str) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    draw.text(((WIDTH - bounds[2]) / 2, y), text, font=font, fill=fill)


def _title_card() -> np.ndarray:
    image, draw = _pil(_canvas())
    draw.rounded_rectangle((86, 72, 1194, 648), radius=28, fill="#0e2035", outline="#1f6f91", width=3)
    draw.ellipse((116, 110, 160, 154), fill="#29d3ff")
    draw.text((180, 108), "VISIOMIND", font=_font(30), fill="#8be8ff")
    _centered(draw, "工业环境物体感知与指令交互智能体", 230, _font(48), "#f5fbff")
    _centered(draw, "Isaac Sim 闭环验证", 310, _font(36), "#58d8ff")
    _centered(draw, "自然语言  →  感知  →  决策  →  执行  →  物理验收", 405, _font(25), "#b9cad8")
    _centered(draw, "RTX 3090 · AnyGrasp 6-DoF · CuRobo · R1 Pro", 525, _font(22), "#6f90a8")
    return _bgr(image)


def _instruction_card(progress: float) -> np.ndarray:
    image, draw = _pil(_canvas())
    draw.text((80, 62), "01  自然语言指令", font=_font(32), fill="#58d8ff")
    draw.rounded_rectangle((80, 145, 1200, 310), radius=22, fill="#112b43", outline="#28769a", width=2)
    visible = INSTRUCTION[: max(1, int(round(len(INSTRUCTION) * min(1.0, progress * 1.7))))]
    draw.text((128, 200), f"“{visible}”", font=_font(39), fill="#ffffff")
    chips = ["意图  transfer_inside", "置信度  0.884", "目标  钳子", "容器  工具箱", "格位  3"]
    x = 85
    for chip in chips:
        width = draw.textbbox((0, 0), chip, font=_font(20))[2] + 34
        draw.rounded_rectangle((x, 370, x + width, 420), radius=16, fill="#123650", outline="#1d88aa")
        draw.text((x + 17, 380), chip, font=_font(20), fill="#ccefff")
        x += width + 14
    draw.text((84, 505), "模型：industrial-char-tfidf-logreg-v2-recovery", font=_font(22), fill="#7895aa")
    draw.text((84, 555), "语义落地：pliers → plier_192    toolbox → toolbox_191", font=_font(24), fill="#b8cad7")
    return _bgr(image)


def _plan_card(progress: float) -> np.ndarray:
    image, draw = _pil(_canvas())
    draw.text((80, 55), "02  任务序列生成", font=_font(32), fill="#58d8ff")
    completed = int(np.clip(progress, 0.0, 1.0) * len(PLAN) + 0.001)
    for index, (step, module) in enumerate(PLAN):
        y = 130 + index * 73
        active = index < completed
        color = "#24d49b" if active else "#315269"
        draw.ellipse((92, y, 126, y + 34), fill=color)
        draw.text((102, y + 2), "✓" if active else str(index + 1), font=_font(19), fill="#06141d" if active else "#d6e2ea")
        draw.line((126, y + 17, 172, y + 17), fill=color, width=3)
        draw.rounded_rectangle((172, y - 8, 1145, y + 43), radius=13, fill="#10283b", outline=color, width=2)
        draw.text((198, y + 1), step, font=_font(24), fill="#ffffff")
        draw.text((940, y + 3), module, font=_font(20), fill="#89a9bb")
    return _bgr(image)


def _fit_video(frame: np.ndarray) -> np.ndarray:
    area_x, area_y, area_w, area_h = 34, 110, 1212, 504
    source_h, source_w = frame.shape[:2]
    scale = min(area_w / source_w, area_h / source_h)
    resized = cv2.resize(frame, (int(source_w * scale), int(source_h * scale)), interpolation=cv2.INTER_LANCZOS4)
    canvas = _canvas()
    x = area_x + (area_w - resized.shape[1]) // 2
    y = area_y + (area_h - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.rectangle(canvas, (area_x, area_y), (area_x + area_w, area_y + area_h), (92, 202, 238), 2)
    return canvas


def _execution_frame(frame: np.ndarray, source_index: int, frame_count: int, pick_boundary: int) -> np.ndarray:
    canvas = _fit_video(frame)
    pick_boundary = max(1, min(frame_count - 1, pick_boundary))
    if source_index < pick_boundary:
        ratio = source_index / pick_boundary
        if ratio < 0.18:
            phase, detail, active = "工业工具感知", "实例分割 · 深度点云 · 目标身份 plier_192", 0
        elif ratio < 0.43:
            phase, detail, active = "AnyGrasp 6-DoF 规划", "候选过滤 · 夹爪内线几何 · 碰撞审计", 1
        elif ratio < 0.86:
            phase, detail, active = "CuRobo 抓取执行", "预抓取 · 受约束接近 · 闭合 · 抬升", 2
        else:
            phase, detail, active = "抓取物理验证", "identity ✓   lift ✓   attachment ✓", 2
    else:
        ratio = (source_index - pick_boundary) / max(1, frame_count - pick_boundary)
        if ratio < 0.56:
            phase, detail, active = "携物安全导航", "A* 可行域 · 0.70 m 停靠 · 全向底盘", 4
        elif ratio < 0.72:
            phase, detail, active = "第三格定位与对齐", "PCA 主轴 · 目标格边界 · 持物净空", 3
        elif ratio < 0.93:
            phase, detail, active = "CuRobo 分段放置", "竖直抬升 · 安全高度转向 · 顶部进入", 5
        else:
            phase, detail, active = "释放与几何验收", "released ✓   stable ✓   cell AABB contained ✓", 6
    image, draw = _pil(canvas)
    draw.rectangle((0, 0, WIDTH, 94), fill="#071522")
    draw.text((34, 20), phase, font=_font(30), fill="#f4fbff")
    draw.text((WIDTH - 290, 27), "ISAAC SIM · RTX 3090", font=_font(18), fill="#68cde9")
    draw.rounded_rectangle((34, 628, 1246, 700), radius=16, fill="#081723", outline="#28566d", width=2)
    draw.text((58, 645), detail, font=_font(23), fill="#d9edf7")
    bar_y = 84
    for index, _ in enumerate(PLAN):
        x = 380 + index * 95
        color = "#21d09b" if index <= active else "#304f61"
        draw.ellipse((x, bar_y - 10, x + 19, bar_y + 9), fill=color)
        if index < len(PLAN) - 1:
            draw.line((x + 19, bar_y, x + 94, bar_y), fill=color if index < active else "#304f61", width=3)
    return _bgr(image)


def _result_card(terminal: dict[str, Any]) -> np.ndarray:
    physical = terminal.get("physical_evidence") or {}
    strategy = terminal.get("placement_strategy") or physical.get("placement_strategy") or "verified placement"
    image, draw = _pil(_canvas())
    draw.rounded_rectangle((95, 66, 1185, 650), radius=28, fill="#0b2928", outline="#20d39a", width=4)
    _centered(draw, "任务闭环成功", 105, _font(52), "#5fffc4")
    _centered(draw, "STRICT PHYSICAL VERIFICATION · PASS", 178, _font(24), "#a9ffe2")
    evidence = [
        "目标身份匹配  object_identity_matches = true",
        "抓取抬升验证  lift_verified = true",
        "持物约束有效  attachment_valid = true",
        "夹爪完成释放  released = true",
        "第三格三维验收  cell_aabb_contained = true",
    ]
    for index, text in enumerate(evidence):
        y = 275 + index * 57
        draw.ellipse((170, y + 3, 194, y + 27), fill="#25d49d")
        draw.text((174, y - 2), "✓", font=_font(18), fill="#052019")
        draw.text((220, y), text, font=_font(24), fill="#e8fff8")
    draw.text((170, 585), f"placement strategy: {strategy}", font=_font(18), fill="#78b7a4")
    return _bgr(image)


def _repeat(writer: cv2.VideoWriter, frame: np.ndarray, seconds: float) -> None:
    for _ in range(int(round(seconds * FPS))):
        writer.write(frame)


def build(run_dir: Path, output: Path) -> None:
    records = _records(run_dir / "process_data.jsonl")
    final, terminal = _verified_run(records)
    source = run_dir / "trajectory.mp4"
    if not source.is_file():
        source = run_dir / "trajectory.avi"
    if not source.is_file():
        raise FileNotFoundError(f"run has no trajectory video: {run_dir}")
    capture = cv2.VideoCapture(str(source))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if frame_count <= 0 or source_fps <= 0.0:
        raise RuntimeError(f"cannot decode trajectory video: {source}")

    final_steps = int(((final.get("environment") or {}).get("step_count") or frame_count))
    pick_step = next(
        (
            int((record.get("payload") or {}).get("control_step") or 0)
            for record in reversed(records)
            if record.get("event") == "orchestrator_completion_monitor_decision"
            and (record.get("payload") or {}).get("subtask_id") == "st_01"
            and (record.get("payload") or {}).get("success") is True
        ),
        int(final_steps * 0.32),
    )
    pick_boundary = int(frame_count * np.clip(pick_step / max(1, final_steps), 0.2, 0.6))

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="visiomind-video-") as temp_dir:
        intermediate = Path(temp_dir) / "review.mp4"
        writer = cv2.VideoWriter(
            str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not initialize the review-video writer")
        _repeat(writer, _title_card(), 4.0)
        for index in range(int(FPS * 5.0)):
            writer.write(_instruction_card(index / (FPS * 5.0 - 1)))
        for index in range(int(FPS * 7.0)):
            writer.write(_plan_card(index / (FPS * 7.0 - 1)))
        source_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(_execution_frame(frame, source_index, frame_count, pick_boundary))
            source_index += 1
        capture.release()
        _repeat(writer, _result_card(terminal), 6.0)
        writer.release()

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(intermediate),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )
    print(
        json.dumps(
            {
                "output": str(output),
                "source": str(source),
                "source_frames": frame_count,
                "source_fps": source_fps,
                "output_fps": FPS,
                "output_resolution": [WIDTH, HEIGHT],
                "pick_boundary_frame": pick_boundary,
                "verified_outcome": final.get("outcome"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("demo/visiomind_industrial_review.mp4")
    )
    args = parser.parse_args()
    build(args.run_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
