# 7프레임 gif 중 4개 프레임만을 대상으로 캘리브레이션 하는 코드. 여러개의 gif 디렉토리를 한 번에 순회하며 각각의 csv를 얻을 수 있음.
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

# RAW 현상 파라미터
# RAW_PARAMS = dict(
#     use_camera_wb=True,
#     gamma=(1, 1),
#     no_auto_bright=False,
#     output_bps=16,
# )


# 실제 사용하시는 3개의 폴더 경로로 아래 문자열들을 수정해 주세요.
DATA_DIRS = [
    Path("path"),
]

out_root = Path("./anycalib_results_496")
out_root.mkdir(parents=True, exist_ok=True)

# 처리 대상 파일들: GIF 목록
GIF_NAMES = [f"sample{i}.gif" for i in range(1, 497)]  # sample1.gif ~ sample96.gif

# =========================
# 4프레임 GIF용 인덱싱 규칙
# =========================
FRAME_COUNT = 4
FRAME_FILENAMES = [f"{i:05d}" for i in range(4, 8)]  # ['00004','00005','00006','00007']

# [수정 2] 파라미터에 k2 추가, AnyCalib 출력 기준(fx, fy)에 맞춰 6개 컬럼 구성
CSV_PARAM_NAMES = ["fx", "fy", "cx", "cy", "k1", "k2"] 
FIELDNAMES = ["gif_filename"]

for f_name in reversed(FRAME_FILENAMES):  # 00007, 00006, 00005, 00004 순서
    for p_name in CSV_PARAM_NAMES:
        FIELDNAMES.append(f"{f_name}_{p_name}")

# [수정 4] 사용하는 모델을 radial:2 로 변경
model_plan = [
    ("anycalib_gen", "radial:2"),
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
    """GIF 파일에서 앞의 4개 프레임만 추출"""
    try:
        gif = Image.open(gif_path)
        frames = []
        for idx, frame in enumerate(ImageSequence.Iterator(gif)):
            if idx >= max_frames:
                break
            frames.append(frame.convert("RGB"))
        return frames
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"[ERROR] GIF 처리 중 오류 발생 {gif_path}: {e}")
        return []

# [수정 5] 저장 파일명에 폴더명(dir_name)이 포함되도록 파라미터 추가
def save_calibration_results_as_csv(all_results: List[Dict], out_dir: Path, model_id: str, dir_name: str):
    """특정 디렉토리의 Calibration 결과를 해당 폴더 안의 개별 CSV로 저장"""
    csv_path = out_dir / f"{model_id}_{dir_name}_calibration_results.csv"

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

# 각 데이터 디렉토리(gif 뭉치)별로 루프 실행
for data_dir in DATA_DIRS:
    print(f"\n=======================================================")
    print(f"| PROCESSING DIRECTORY: {data_dir.name} |")
    print(f"=======================================================")

    # 폴더가 존재하는지 확인 (경로 오타 방지)
    if not data_dir.exists():
        print(f"[SKIP] 해당 디렉토리를 찾을 수 없습니다: {data_dir}")
        continue

    for model_id, cam_id in model_plan:
        model = LOADED_MODELS[model_id]

        # 개별 디렉토리에 대한 결과 저장 폴더 생성
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

            for frame_idx, frame in enumerate(frames[:FRAME_COUNT]):
                mapped_file_num = 7 - frame_idx  # 7, 6, 5, 4
                mapped_file_name = f"{mapped_file_num:05d}"

                img_tensor = pil_to_tensor(frame, dev)
                with torch.inference_mode():
                    output = model.predict(img_tensor, cam_id=cam_id)

                intr = output.get("intrinsics", None)

                if isinstance(intr, torch.Tensor):
                    intr_list = intr.detach().cpu().flatten().tolist()
                else:
                    intr_list = []

                # [수정 6] 6개의 파라미터 (fx, fy, cx, cy, k1, k2) 추출
                if len(intr_list) >= 6: 
                    fx, fy, cx, cy, k1, k2 = intr_list[0:6]
                    params = [fx, fy, cx, cy, k1, k2]
                else:
                    print(f"[WARN] {gif_name}, frame {frame_idx}: 파라미터 부족")
                    params = [float("nan")] * 6

                for p_name, val in zip(CSV_PARAM_NAMES, params):
                    frame_intrinsics[f"{mapped_file_name}_{p_name}"] = val

            row_data = {"gif_filename": gif_name}
            row_data.update(frame_intrinsics)
            directory_results.append(row_data)

        # [수정 8] 글로벌 리스트에 모으지 않고, 각 디렉토리 처리가 끝날 때마다 개별 CSV로 저장
        save_calibration_results_as_csv(directory_results, out_dir, model_id, data_dir.name)

print("\n[DONE] 모든 디렉토리 및 GIF 처리, 최종 CSV 개별 저장 완료.")