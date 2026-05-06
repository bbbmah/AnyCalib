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

# 입력/출력 경로
DATA_ROOT = Path("/home/work/Hwang/Generative_Ph/generative-photography/inference_output/genphoto_focal_length/2025.11.15")
DATA_DIRS = [DATA_ROOT]

out_root = Path("./anycalib_results")
out_root.mkdir(parents=True, exist_ok=True)

# 처리 대상 파일들: GIF 목록 
GIF_NAMES = [f"sample{i}.gif" for i in range(1, 97)] # sample1.gif ~ sample96.gif

# 프레임 이름 매핑
FRAME_FILENAMES = [f"{i:05d}" for i in range(1, 8)]

# 결과 CSV 컬럼 이름 매핑: radial:2에 맞춰 6개의 파라미터 지정
CSV_PARAM_NAMES = ["fx", "fy", "cx", "cy", "k1", "k2"] 
FIELDNAMES = ["gif_filename"]
for f_name in reversed(FRAME_FILENAMES): 
    for p_name in CSV_PARAM_NAMES:       
        FIELDNAMES.append(f"{f_name}_{p_name}")

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

def extract_frames(gif_path: Path) -> List[Image.Image]:
    """GIF 파일에서 프레임을 추출하여 PIL 이미지 리스트로 만듬"""
    try:
        gif = Image.open(gif_path)
        frames = [frame.convert('RGB') for frame in ImageSequence.Iterator(gif)]
        return frames
    except FileNotFoundError:
        print(f"[ERROR] GIF 파일 찾을 수 없음: {gif_path}")
        return []
    except Exception as e:
        print(f"[ERROR] GIF 처리 중 오류 발생 {gif_path}: {e}")
        return []

def save_calibration_results_as_csv(all_results: List[Dict], out_dir: Path, model_id: str):
    """모든 GIF의 Calibration 결과를 하나의 CSV 파일로 저장"""
    csv_path = out_dir / f"{model_id}_radial2_calibration_results.csv"
    
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

# 디렉토리별 결과를 저장할 최종 리스트
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
            
            frames = extract_frames(gif_path)
            if not frames:
                continue

            frame_intrinsics = {}
            
            for frame_idx, frame in enumerate(frames):
                mapped_file_num = 7 - frame_idx
                mapped_file_name = f"{mapped_file_num:05d}" 

                # Calibration 수행
                img_tensor = pil_to_tensor(frame, dev)
                with torch.inference_mode():
                    output = model.predict(img_tensor, cam_id=cam_id)
                
                intr = output.get("intrinsics", None)
                
                if isinstance(intr, torch.Tensor):
                    intr_list = intr.detach().cpu().flatten().tolist()
                else:
                    intr_list = []
                    
                # [수정됨] 6개의 파라미터 (fx, fy, cx, cy, k1, k2) 추출
                if len(intr_list) >= 6: 
                    fx, fy, cx, cy, k1, k2 = intr_list[0:6]
                    params = [fx, fy, cx, cy, k1, k2] 
                else:
                    print(f"[WARN] {gif_name}, frame {frame_idx}: 파라미터 부족")
                    params = [float("nan")] * 6

                # 결과 딕셔너리에 저장
                for p_name, val in zip(CSV_PARAM_NAMES, params):
                    frame_intrinsics[f"{mapped_file_name}_{p_name}"] = val
                    
            row_data = {"gif_filename": gif_name}
            row_data.update(frame_intrinsics) 
            directory_results.append(row_data)

        global_results_by_model[model_id].extend(directory_results)

# 최종 CSV 저장 루프 
for model_id, results in global_results_by_model.items():
    save_calibration_results_as_csv(results, out_root, model_id)

print("\n[DONE] 모든 디렉토리 및 GIF 처리, 최종 CSV 저장 완료.")