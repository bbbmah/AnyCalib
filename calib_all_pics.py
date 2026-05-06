# 임의의 이미지를 네 개의 카메라 모델로 캘리브레이션 하는 코드.
import os
import csv
from pathlib import Path
import numpy as np
import torch
import rawpy
from anycalib import AnyCalib

# ===========================================================
# 설정
# ===========================================================
BASE_DIR = Path("/home/work/Hwang/datasets/SR-RAW/train")  # 베이스 경로
OUT_ROOT = Path("./anycalib_batch_results")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# AnyCalib 모델 4종
MODEL_IDS = ["anycalib_pinhole", "anycalib_gen", "anycalib_dist", "anycalib_edit"]
CAM_ID = "radial:1"  # 왜곡계수 k1 하나만

# RAW 처리 옵션
RAW_PARAMS = dict(
    use_camera_wb=True,
    gamma=(1, 1),
    no_auto_bright=True,
    output_bps=16,
)

# GPU 사용 가능 시 CUDA
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================================================
# 유틸 함수
# ===========================================================
def load_raw_as_tensor(path: Path, params: dict, device: torch.device):
    """RAW 파일(.ARW) -> (3,H,W) float32 [0,1] Tensor"""
    with rawpy.imread(str(path)) as raw:
        rgb16 = raw.postprocess(**params)
    rgb = rgb16.astype(np.float32) / 65535.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    return tensor.to(device=device, dtype=torch.float32)

def predict_intrinsics(model: AnyCalib, image_path: Path, cam_id: str):
    """단일 이미지에서 intrinsics(fx, fy, cx, cy, k1) 추출"""
    img = load_raw_as_tensor(image_path, RAW_PARAMS, DEV)
    with torch.inference_mode():
        output = model.predict(img, cam_id=cam_id)
    intr = output.get("intrinsics")
    if intr is None:
        return [None] * 5
    intr = intr.detach().cpu().numpy().tolist()
    # 모델마다 길이가 다르므로 5개만 (fx, fy, cx, cy, k1)
    values = intr[:5] + [None] * max(0, 5 - len(intr))
    return values

# ===========================================================
# 메인 루프
# ===========================================================
for model_id in MODEL_IDS:
    print(f"\n===== Running model: {model_id} =====")

    # 모델 로드
    model = AnyCalib(model_id=model_id).to(DEV)
    model.eval()

    # 출력 파일 초기화
    out_csv = OUT_ROOT / f"{model_id}_intrinsics.csv"
    fieldnames = ["dir"]
    # 열 이름 정의 (00001_fx, ..., 00007_k1)
    for i in range(1, 8):
        if model_id == "anycalib_pinhole":
            # pinhole은 왜곡계수 없음
            fieldnames += [f"{i:05d}_{key}" for key in ["fx", "fy", "cx", "cy"]]
        else:
            fieldnames += [f"{i:05d}_{key}" for key in ["fx", "fy", "cx", "cy", "k1"]]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # 하위 디렉토리 순회
        for subdir in sorted(BASE_DIR.iterdir()):
            if not subdir.is_dir():
                continue
            dir_name = subdir.name
            row = {"dir": dir_name}

            for i in range(1, 8):
                img_path = subdir / f"{i:05d}.ARW"
                if not img_path.exists():
                    # 빈 슬롯은 None
                    for key in fieldnames:
                        if key.startswith(f"{i:05d}_"):
                            row[key] = None
                    continue

                try:
                    values = predict_intrinsics(model, img_path, cam_id=CAM_ID)
                except Exception as e:
                    print(f"[ERROR] {dir_name}/{img_path.name}: {e}")
                    values = [None] * (4 if model_id == "anycalib_pinhole" else 5)

                keys = ["fx", "fy", "cx", "cy"] if model_id == "anycalib_pinhole" else ["fx", "fy", "cx", "cy", "k1"]
                for key, val in zip(keys, values):
                    row[f"{i:05d}_{key}"] = val

                print(f"{dir_name}/{img_path.name} → {values}")

            writer.writerow(row)

    print(f"[SAVE] {model_id} results -> {out_csv}")

print("\n[DONE] 모든 모델(4종) × 모든 디렉토리 처리 완료.")