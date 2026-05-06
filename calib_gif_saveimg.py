# 7 프레임 gif 파일을 순회하며 7개 프레임에 대한 radial:1 방식을 캘리브레이션, csv 파일을 저장하는 코드.
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
DATA_ROOT = Path("/home/work/Hwang/Generative_Ph/generative-photography/inference_output/genphoto_focal_length/2025.11.15")
# SCENE_NUMBERS = [
#     "00068", "00053", "00563", "00376", "00069", "00564", "00043", "00284", "00531", "00245", "00118", "00381", "00400", "00380", "00428", "00388", "00405", "00488", "00045", "00241", "00576", "00533", "00431", "00571", "00485", "00344", "00330", "00384", "00328", "00383", "00229", "00331", "00391", "00110"
# ]
DATA_DIRS = [DATA_ROOT]# / scene_num for scene_num in SCENE_NUMBERS]

out_root = Path("./anycalib_results")
out_root.mkdir(parents=True, exist_ok=True)

#  수정된 처리 대상 파일들: GIF 목록 
GIF_NAMES = [f"sample{i}.gif" for i in range(1, 97)] # sample1.gif ~ sample96.gif

# 프레임 이름 매핑 (7번째 프레임 -> 00001, 1번째 프레임 -> 00007)
FRAME_FILENAMES = [f"{i:05d}" for i in range(1, 8)] # ['00001', '00002', ..., '00007']
# 결과 CSV 컬럼 이름 매핑: '00001_f', '00001_cx', ... '00007_k'
CSV_PARAM_NAMES = ["f", "cx", "cy", "k"] # 인트린직 파라미터 이름 (k는 왜곡 계수 k1이라고 가정)
FIELDNAMES = ["gif_filename"]
for f_name in reversed(FRAME_FILENAMES): # 00007, 00006, ... 00001 순서로 처리 후 
    for p_name in CSV_PARAM_NAMES:       # 00001_f, 00001_cx, ... 순서로 저장
        FIELDNAMES.append(f"{f_name}_{p_name}")

model_plan = [
    #("anycalib_pinhole", "pinhole"),
    ("anycalib_gen", "radial:1"),
    #("anycalib_dist", "radial:1"),
    #("anycalib_edit", "radial:1"),
]

# =========================
# 유틸
# =========================
# pil_to_tensor 함수 (RAW 대신 GIF 프레임 처리용으로 유지)
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
        # 프레임을 RGB로 변환하여 리스트로 저장
        frames = [frame.convert('RGB') for frame in ImageSequence.Iterator(gif)]
        return frames
    except FileNotFoundError:
        print(f"[ERROR] GIF 파일 찾을 수 없음: {gif_path}")
        return []
    except Exception as e:
        print(f"[ERROR] GIF 처리 중 오류 발생 {gif_path}: {e}")
        return []

# CSV 저장 함수: 결과를 하나의 거대한 CSV 파일로 저장하도록 수정
def save_calibration_results_as_csv(all_results: List[Dict], out_dir: Path, model_id: str):
    """모든 GIF의 Calibration 결과를 하나의 CSV 파일로 저장"""
    
    csv_path = out_dir / f"{model_id}_all_calibration_results.csv"
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        
        # 각 GIF 파일의 결과를 행으로 작성
        for row_data in all_results:
            writer.writerow(row_data)
            
    print(f"[SAVE] CSV  -> {csv_path} ({len(all_results)} rows saved)")


# =========================
# 메인 루프
# =========================
#  개선 사항: 모델 로드를 외부로 빼서 한 번만 수행 
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

    # 모델 루프
    for model_id, cam_id in model_plan:
        model = LOADED_MODELS[model_id]

        # 결과 저장 폴더 설정
        out_dir = out_root / data_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 보정된 이미지 저장용 폴더 (요청에 따라 파일 저장 로직은 삭제)
        # undistorted_dir = out_dir / "undistorted_images" 
        # undistorted_dir.mkdir(parents=True, exist_ok=True)

        directory_results = []
        
        # GIF 파일 루프
        for gif_name in GIF_NAMES:
            gif_path = data_dir / gif_name
            if not gif_path.exists():
                # print(f"[SKIP] GIF 파일 없음: {gif_path}")
                continue
            
            frames = extract_frames(gif_path)
            if not frames:
                continue

            #  프레임 순서 처리: 1번째 프레임 -> 7번째 프레임 (인덱스 0 -> 6) 
            #  파일 이름 매핑: 1번째 프레임(frames[0])은 00007.jpg에 매핑되어야 함.
            #  frames 리스트를 역순으로 순회해야 파일 이름 00007, 00006, ... 순서와 맞음.
            
            frame_intrinsics = {}
            
            # frames를 역순으로 순회 (frames[0] -> frames[6])
            # frame_idx는 0부터 6까지
            for frame_idx, frame in enumerate(frames):
                
                # 프레임의 역순 매핑 계산
                # 1번째 프레임(idx 0) -> 7 (00007)
                # 7번째 프레임(idx 6) -> 1 (00001)
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
                    
                # 4개의 파라미터 (f, cx, cy, k1) 추출
                if len(intr_list) >= 5: # fx, fy, cx, cy, k1, ...
                    fx, fy, cx, cy, k1 = intr_list[0], intr_list[1], intr_list[2], intr_list[3], intr_list[4]
                    
                    #  CSV 저장 형식에 맞게 4개의 파라미터만 사용 (fx, cx, cy, k1)
                    # *참고: AnyCalib은 fx, fy를 분리하지만, 요청에 따라 f=fx만 사용*
                    params = [fx, cx, cy, k1] 
                else:
                    print(f"[WARN] {gif_name}, frame {frame_idx}: 파라미터 부족")
                    params = [float("nan")] * 4

                # 결과 딕셔너리에 저장
                for p_name, val in zip(CSV_PARAM_NAMES, params):
                    frame_intrinsics[f"{mapped_file_name}_{p_name}"] = val
                    
                #  왜곡 보정 및 이미지 저장 (요청에 따라 로직만 남김)
                if "radial" in cam_id and len(intr_list) >= 5:
                     # (왜곡 보정 및 저장 로직은 복잡성 및 요청 제외로 인해 주석 처리/단순화)
                     pass

            #  최종 CSV 행 데이터 구성 
            row_data = {"gif_filename": gif_name}
            # row_data에 frame_intrinsics의 모든 키-값 쌍을 추가
            row_data.update(frame_intrinsics) 
            directory_results.append(row_data)

        # 디렉토리별 결과를 최종 글로벌 리스트에 추가
        global_results_by_model[model_id].extend(directory_results)

#  최종 CSV 저장 루프 
for model_id, results in global_results_by_model.items():
    save_calibration_results_as_csv(results, out_root, model_id)

print("\n[DONE] 모든 디렉토리 및 GIF 처리, 최종 CSV 저장 완료.")