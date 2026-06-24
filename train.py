"""
TEKNOFEST 2026 - YOLO Eğitim + Başarı Test Sistemi
pip install ultralytics torch torchvision torchaudio pyyaml
"""

from ultralytics import YOLO
from pathlib import Path
import yaml
import torch
import logging
import sys
from datetime import datetime
from typing import Dict


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("YOLO")


# =========================================================
# DEVICE KONTROL
# =========================================================

def get_device() -> str:
    """
    GPU varsa CUDA kullan.
    Yoksa CPU kullan.
    """

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA aktif -> {gpu_name}")
        return "0"

    logger.warning("CUDA bulunamadı -> CPU kullanılacak")
    return "cpu"


def print_system_info():
    """Sistem bilgisi göster."""

    print("\n========== SİSTEM BİLGİSİ ==========")

    print(f"Python : {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")

    if torch.cuda.is_available():

        print(f"CUDA   : {torch.version.cuda}")
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

        vram = (
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3
        )

        print(f"VRAM   : {vram:.2f} GB")

    else:
        print("GPU    : Kullanılamıyor")

    print("====================================\n")


# =========================================================
# YAML DOĞRULAMA
# =========================================================

def validate_yaml(data_yaml: str) -> bool:
    """
    data.yaml doğrulama.
    """

    yaml_path = Path(data_yaml)

    if not yaml_path.exists():
        logger.error(f"data.yaml bulunamadı -> {data_yaml}")
        return False

    try:

        with open(yaml_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        required_keys = ["train", "val", "names"]

        for key in required_keys:

            if key not in data:
                logger.error(f"Eksik alan -> {key}")
                return False

        logger.info("data.yaml doğrulandı")
        return True

    except Exception as error:
        logger.error(f"YAML hatası -> {error}")
        return False


# =========================================================
# MODEL CLASS NAMES
# =========================================================

def get_model_class_names(model: YOLO) -> Dict[int, str]:
    """
    Model class isimlerini güvenli al.
    """

    names = getattr(model, "names", {})

    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}

    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}

    return {}


# =========================================================
# OTOMATİK BATCH SIZE
# =========================================================

def get_optimal_batch_size() -> int:
    """
    GPU belleğine göre batch size belirle.
    """

    if not torch.cuda.is_available():
        return 4

    total_vram = (
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3
    )

    if total_vram >= 16:
        return 32

    if total_vram >= 10:
        return 16

    if total_vram >= 6:
        return 8

    return 4


# =========================================================
# MODEL EĞİTİMİ
# =========================================================

def train_custom_model(
    data_yaml: str = "data/data.yaml",
    model_name: str = "yolo11n.pt",
    epochs: int = 100,
    imgsz: int = 640,
):
    """
    YOLO model eğitimi.
    """

    print_system_info()

    if not validate_yaml(data_yaml):
        return None

    device = get_device()
    batch_size = get_optimal_batch_size()

    logger.info("Model yükleniyor...")

    model = YOLO(model_name)

    logger.info("Eğitim başladı...")

    try:

        results = model.train(

            # Dataset
            data=data_yaml,

            # Temel ayarlar
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,

            # Performans
            workers=4,
            cache=True,
            amp=True,

            # Eğitim parametreleri
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.0001,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,

            # Scheduler
            cos_lr=True,

            # Early stopping
            patience=25,

            # Augmentation
            mosaic=1.0,
            mixup=0.2,
            fliplr=0.5,
            flipud=0.5,
            degrees=10,
            scale=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,

            # Kayıt
            save=True,
            save_period=10,
            pretrained=True,
            verbose=True,
            name=f"custom_drone_"
                 f"{datetime.now().strftime('%Y%m%d_%H%M')}"
        )

        logger.info("Eğitim tamamlandı")

        return results

    except torch.cuda.OutOfMemoryError:

        logger.error("GPU belleği yetersiz")

        print("\nÇÖZÜM ÖNERİLERİ:")
        print("- batch size düşür")
        print("- imgsz=512 kullan")

    except Exception as error:

        logger.exception(f"Eğitim hatası -> {error}")

    return None


# =========================================================
# MODEL BAŞARI TESTİ
# =========================================================

def evaluate_model(
    model_path: str,
    data_yaml: str = "data/data.yaml",
):
    """
    Eğitilmiş modeli test et.
    """

    model_file = Path(model_path)

    if not model_file.exists():
        logger.error(f"Model bulunamadı -> {model_path}")
        return

    logger.info("Model yükleniyor...")

    try:

        model = YOLO(model_path)

        logger.info("Validation başladı...")

        metrics = model.val(

            data=data_yaml,
            split="val",
            imgsz=640,
            batch=get_optimal_batch_size(),
            device=get_device(),
            verbose=True
        )

        print("\n========== MODEL BAŞARI RAPORU ==========\n")

        print(f"mAP@0.50       : {metrics.box.map50:.4f}")
        print(f"mAP@0.50:0.95  : {metrics.box.map:.4f}")
        print(f"Precision      : {metrics.box.mp:.4f}")
        print(f"Recall         : {metrics.box.mr:.4f}")

        print("\n=========================================\n")

        print("YORUM:")

        if metrics.box.map50 >= 0.90:
            print("Mükemmel model")

        elif metrics.box.map50 >= 0.75:
            print("Çok iyi model")

        elif metrics.box.map50 >= 0.60:
            print("Orta seviye model")

        else:
            print("Dataset geliştirilmeli")

    except Exception as error:

        logger.exception(f"Validation hatası -> {error}")


# =========================================================
# ANA MENÜ
# =========================================================

def main():

    print("""
╔══════════════════════════════════════════════╗
║      TEKNOFEST 2026 YOLO TRAIN SYSTEM       ║
╚══════════════════════════════════════════════╝

1. Model Eğit
2. Başarı Testi (Validation)
""")

    choice = input("Seçim yap: ").strip()

    if choice == "1":

        train_custom_model()

    elif choice == "2":

        model_path = input(
            "\nModel yolu gir:\n"
            "Örnek:\n"
            "runs/detect/custom_drone_xxx/weights/best.pt\n\n> "
        ).strip()

        evaluate_model(model_path)

    else:
        print("Hatalı seçim")


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    main()