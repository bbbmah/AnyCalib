import json
import csv
from pathlib import Path
import numpy as np
import torch
import rawpy
import cv2
from anycalib import AnyCalib

# =========================
# 설정
# =========================
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RAW 현상 파라미터(선형 톤, 오토브라이트 끔, 16bit)
RAW_PARAMS = dict(
    use_camera_wb=True,
    gamma=(1, 1),
    no_auto_bright=False,
    output_bps=16,
)

# 입력/출력 경로
data_dir = Path("/home/work/Hwang/datasets/SR-RAW/train/00063")
out_root = Path("./anycalib_results")
out_root.mkdir(parents=True, exist_ok=True)

# 처리 대상 파일들
filenames = [f"{i:05d}.ARW" for i in range(1, 8)]  # 00001.ARW ~ 00007.ARW

# 사용할 모델 목록과 cam_id 매핑
model_plan = [
    ("anycalib_pinhole", "pinhole"),
    ("anycalib_gen", "radial:4"),
    ("anycalib_dist", "radial:4"),
    ("anycalib_edit", "radial:4"),
]

# =========================
# 유틸
# =========================
def load_raw_as_tensor(path: Path, params: dict, device: torch.device):
    """RAW(.ARW) -> (3,H,W) float32 in [0,1] Tensor"""
    with rawpy.imread(str(path)) as raw:
        rgb16 = raw.postprocess(**params)  # (H,W,3) uint16
    rgb = rgb16.astype(np.float32) / 65535.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    return tensor.to(device=device, dtype=torch.float32)

def tensor_to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {k: tensor_to_list(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [tensor_to_list(t) for t in x]
    return x

def save_as_json_csv(results, out_dir: Path, stem: str):
    """[{filename, intrinsics:[...]}] 형태를 JSON/CSV로 저장"""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] JSON -> {json_path}")

    # CSV
    max_len = max((len(r["intrinsics"]) for r in results), default=0)
    fieldnames = ["filename"] + [f"param_{i}" for i in range(max_len)]
    csv_path = out_dir / f"{stem}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {"filename": r["filename"]}
            for i, v in enumerate(r["intrinsics"]):
                row[f"param_{i}"] = v
            writer.writerow(row)
    print(f"[SAVE] CSV  -> {csv_path}")

# =========================
# 메인 루프
# =========================
for model_id, cam_id in model_plan:
    print(f"\n===== Model: {model_id} | cam_id: {cam_id} =====")
    # 모델 로드
    model = AnyCalib(model_id=model_id).to(dev)

    # 모델별 결과 저장용 폴더
    out_dir = out_root / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 보정된 이미지 저장용 폴더
    undistorted_dir = out_dir / "undistorted_images"
    undistorted_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name in filenames:
        raw_path = data_dir / name
        if not raw_path.exists():
            print(f"[SKIP] 파일 없음: {raw_path}")
            continue
            
        # RAW 이미지를 numpy 배열로 변환
        with rawpy.imread(str(raw_path)) as raw:
            rgb16 = raw.postprocess(**RAW_PARAMS)
        img_np = rgb16.astype(np.float32) / 65535.0
        img_bgr = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        # AnyCalib 모델로 파라미터 예측
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous().to(dev)
        with torch.inference_mode():
            output = model.predict(img_tensor, cam_id=cam_id)
            
        intr = output.get("intrinsics", None)
        
        # 파라미터 추출 및 변환
        if isinstance(intr, torch.Tensor):
            intr = intr.detach().cpu().flatten().tolist()
        elif intr is None:
            print(f"[WARN] intrinsics 없음: {name}")
            intr = []
            
        # Pinhole 모델은 4개 파라미터 (fx, fy, cx, cy)
        # Radial 모델은 4개 + k1~k4
        
        # 왜곡 보정
        if "radial" in cam_id and len(intr) > 4:
            # 5번째 이후 값들이 왜곡 계수
            fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
            dist_coeffs_list = intr[4:]

            camera_matrix = np.array([[fx, 0, cx],
                                      [0, fy, cy],
                                      [0, 0, 1]], dtype=np.float32)
            
            # 왜곡 계수 배열 (k1, k2, p1, p2, k3, k4, ...)
            # AnyCalib의 `simple_radial:4`는 k1~k4만 반환하므로 p1,p2는 0으로 설정
            dist_coeffs = np.zeros(8, dtype=np.float32)
            for i, val in enumerate(dist_coeffs_list):
                if i < 4:
                    # k1, k2, k3, k4를 순서대로 매핑
                    # AnyCalib의 k3, k4는 OpenCV의 k3, k4에 매핑 (k3가 4번째, k4가 5번째)
                    if i < 2:
                        dist_coeffs[i] = val
                    elif i < 4:
                        dist_coeffs[i+2] = val
            
            # cv2.undistort로 보정
            undistorted_img = cv2.undistort(img_bgr, camera_matrix, dist_coeffs, None, camera_matrix)
            
            # 보정된 이미지 저장
            save_path = undistorted_dir / name.replace(".ARW", "_undistorted.png")
            cv2.imwrite(str(save_path), undistorted_img)
            print(f"[SAVE] Undistorted image -> {save_path}")

        elif "pinhole" in cam_id:
            print(f"[INFO] Pinhole 모델은 왜곡 계수가 없어 이미지 보정을 수행하지 않습니다.")
        else:
            print(f"[WARN] 지원하지 않는 cam_id 또는 파라미터 부족: {cam_id}")

        results.append({"filename": name, "intrinsics": intr})

    # 모델별 결과 저장
    save_as_json_csv(results, out_dir, stem=f"{model_id}_intrinsics")

print("\n[DONE] 모든 모델 처리 및 이미지 보정 완료.")