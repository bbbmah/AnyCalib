import json
import csv
from pathlib import Path
import numpy as np
import torch
import rawpy
from anycalib import AnyCalib

# =========================
# 설정
# =========================
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RAW 현상 파라미터(선형 톤, 오토브라이트 끔, 16bit)
RAW_PARAMS = dict(
    use_camera_wb=True,
    gamma=(1, 1),
    no_auto_bright=True,
    output_bps=16,
)

# 입력/출력 경로
data_dir = Path("/home/work/Hwang/datasets/SR-RAW/train/00046")
out_root = Path("./anycalib_results")
out_root.mkdir(parents=True, exist_ok=True)

# 처리 대상 파일들
filenames = [f"{i:05d}.ARW" for i in range(1, 8)]  # 00001.ARW ~ 00007.ARW

# 사용할 모델 목록과 cam_id 매핑
model_plan = [
    ("anycalib_pinhole", "pinhole"),
    ("anycalib_gen", "simple_radial:4"),
    ("anycalib_dist", "simple_radial:4"),
    ("anycalib_edit", "simple_radial:4"),
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

    results = []
    for name in filenames:
        raw_path = data_dir / name
        if not raw_path.exists():
            print(f"[SKIP] 파일 없음: {raw_path}")
            continue

        img = load_raw_as_tensor(raw_path, RAW_PARAMS, dev)

        with torch.inference_mode():
            output = model.predict(img, cam_id=cam_id)

        intr = output.get("intrinsics", None)
        if isinstance(intr, torch.Tensor):
            intr = [float(x) for x in intr.detach().cpu().flatten()]
        elif intr is None:
            print(f"[WARN] intrinsics 없음: {name}")
            intr = []

        # 화면 출력
        print(f"{name}: intrinsics = {intr}")

        results.append({"filename": name, "intrinsics": intr})

    # 모델별 결과 저장
    save_as_json_csv(results, out_dir, stem=f"{model_id}_intrinsics")

print("\n[DONE] 4개 모델 × 7장 결과 저장 완료.")
