import json
import csv
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
import cv2
from PIL import Image, ImageSequence
from anycalib import AnyCalib

# =========================
# 설정
# =========================
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RAW 현상 파라미터 (RAW 파일 처리는 GIF에서는 사용되지 않지만, 변수 정의는 유지)
RAW_PARAMS = dict(
    use_camera_wb=True,
    gamma=(1, 1),
    no_auto_bright=False,
    output_bps=16,
)

# 입력/출력 경로
DATA_ROOT = Path("C:/Personal_Files/2025_Summer/Sojung/96개 프롬프트의 결과_수치,시각적 평가/raw 파일 결과들/distorted_Genphoto")
DATA_DIRS = [DATA_ROOT]  # / scene_num for scene_num in SCENE_NUMBERS]

out_root = Path("./anycalib_results")
out_root.mkdir(parents=True, exist_ok=True)

# 처리 대상 파일들: GIF 목록
GIF_NAMES = [f"distorted_sample{i}.gif" for i in range(1, 97)]  # sample1.gif ~ sample96.gif

# =========================
# 4프레임 GIF용 인덱싱 규칙
# =========================
# 1번째 프레임 -> 00007
# 2번째 프레임 -> 00006
# 3번째 프레임 -> 00005
# 4번째 프레임 -> 00004
# (즉, 00007 ~ 00004만 사용)
FRAME_COUNT = 4
FRAME_FILENAMES = [f"{i:05d}" for i in range(4, 8)]  # ['00004','00005','00006','00007']

CSV_PARAM_NAMES = ["f", "cx", "cy", "k"]  # 인트린직 파라미터 이름 (k는 왜곡 계수 k1이라고 가정)
FIELDNAMES = ["gif_filename"]
for f_name in reversed(FRAME_FILENAMES):  # 00007, 00006, 00005, 00004 순서로 저장
    for p_name in CSV_PARAM_NAMES:
        FIELDNAMES.append(f"{f_name}_{p_name}")

model_plan = [
    ("anycalib_gen", "radial:1"),
]

# =========================
# 유틸
# =========================
def pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """PIL Image -> (3,H,W) float32 in [0,1] Tensor"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor.to(device=device, dtype=torch.float32)

def extract_frames(gif_path: Path, max_frames: int = FRAME_COUNT) -> List[Image.Image]:
    """
    GIF 파일에서 프레임을 추출하여 PIL 이미지 리스트로 만듬.
    4프레임 GIF를 전제로 하되, 혹시 더 많으면 앞의 4개만 사용.
    """
    try:
        gif = Image.open(gif_path)
        frames = []
        for idx, frame in enumerate(ImageSequence.Iterator(gif)):
            if idx >= max_frames:
                break
            frames.append(frame.convert("RGB"))
        return frames
    except FileNotFoundError:
        print(f"[ERROR] GIF 파일 찾을 수 없음: {gif_path}")
        return []
    except Exception as e:
        print(f"[ERROR] GIF 처리 중 오류 발생 {gif_path}: {e}")
        return []

def save_calibration_results_as_csv(all_results: List[Dict], out_dir: Path, model_id: str):
    """모든 GIF의 Calibration 결과를 하나의 CSV 파일로 저장"""
    csv_path = out_dir / f"{model_id}_all_calibration_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for row_data in all_results:
            writer.writerow(row_data)

    print(f"[SAVE] CSV  -> {csv_path} ({len(all_results)} rows saved)")

# =========================
# 메인 루프
# =========================

LOADED_MODELS = {}
for mid, cid in model_plan:
    print(f"[LOAD] Loading model {mid}...")
    LOADED_MODELS[mid] = AnyCalib(model_id=mid).to(dev).eval()

global_results_by_model = {mid: [] for mid, _ in model_plan}

for data_dir in DATA_DIRS:
    print(f"\n=======================================================")
    print(f"| PROCESSING DIRECTORY: {data_dir.name} |")
    print(f"=======================================================")

    for model_id, cam_id in model_plan:
        model = LOADED_MODELS[model_id]

        out_dir = out_root / data_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        directory_results = []

        for gif_name in GIF_NAMES:
            gif_path = data_dir / gif_name
            if not gif_path.exists():
                continue

            frames = extract_frames(gif_path, max_frames=FRAME_COUNT)
            if not frames:
                continue

            if len(frames) < FRAME_COUNT:
                print(f"[WARN] {gif_name}: 프레임 수 {len(frames)}개 (기대: {FRAME_COUNT})")

            frame_intrinsics = {}

            # 4프레임 인덱싱:
            # frame_idx=0 -> 00007
            # frame_idx=1 -> 00006
            # frame_idx=2 -> 00005
            # frame_idx=3 -> 00004
            for frame_idx, frame in enumerate(frames[:FRAME_COUNT]):
                mapped_file_num = 7 - frame_idx  # 7,6,5,4
                mapped_file_name = f"{mapped_file_num:05d}"

                img_tensor = pil_to_tensor(frame, dev)
                with torch.inference_mode():
                    output = model.predict(img_tensor, cam_id=cam_id)

                intr = output.get("intrinsics", None)

                if isinstance(intr, torch.Tensor):
                    intr_list = intr.detach().cpu().flatten().tolist()
                else:
                    intr_list = []

                if len(intr_list) >= 5:  # fx, fy, cx, cy, k1, ...
                    fx, fy, cx, cy, k1 = intr_list[0], intr_list[1], intr_list[2], intr_list[3], intr_list[4]
                    params = [fx, cx, cy, k1]  # f=fx만 사용
                else:
                    print(f"[WARN] {gif_name}, frame {frame_idx}: 파라미터 부족")
                    params = [float("nan")] * 4

                for p_name, val in zip(CSV_PARAM_NAMES, params):
                    frame_intrinsics[f"{mapped_file_name}_{p_name}"] = val

                if "radial" in cam_id and len(intr_list) >= 5:
                    # (왜곡 보정 및 저장 로직은 요청에 따라 제외)
                    pass

            row_data = {"gif_filename": gif_name}
            row_data.update(frame_intrinsics)
            directory_results.append(row_data)

        global_results_by_model[model_id].extend(directory_results)

for model_id, results in global_results_by_model.items():
    save_calibration_results_as_csv(results, out_root, model_id)

print("\n[DONE] 모든 디렉토리 및 GIF 처리, 최종 CSV 저장 완료.")