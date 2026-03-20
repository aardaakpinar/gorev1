from ultralytics import YOLO
import yaml
from pathlib import Path

def create_dataset_yaml():
    """dataset.yaml oluştur"""
    
    data = {
        'path': str(Path.cwd()),
        'train': 'data/train',
        'val': 'data/valid',
        'test': 'data/test',
        'nc': 6,  # Sınıf sayısı
        'names': ['Trucks', 'UAI', 'UAP', 'bicycle', 'car', 'person']
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(data, f)
    
    print("✅ data.yaml oluşturuldu")

def train_custom_model():
    """Custom YOLO modeli eğit"""
    
    print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║    TEKNOFEST 2026 - CUSTOM YOLO MODEL EĞİTİMİ                  ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝

ÖN HAZIRLIKLAR:
1. Drone görüntülerinizi toplayın (100-500 görüntü minimum)
2. Etiketleme yapın:
   - Roboflow (https://roboflow.com/)
   - CVAT (https://www.cvat.ai/)
   - Makesense.ai
   
3. Klasör yapısı:
   data/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── labels/
       ├── train/
       ├── val/
       └── test/

4. data.yaml dosyası:
   path: /path/to/data
   train: images/train
   val: images/val
   nc: 4
   names: [Taşıt, İnsan, UAP, UAİ]
    """)
    
    print("\n📌 ADIM 1: Pre-trained modeli yükle")
    model = YOLO('yolov8n.pt')  # Medium model
    print("✅ Model yüklendi")
    
    print("\n📌 ADIM 2: Modeli eğit")
    print("   (Bu işlem 1-2 saat sürebilir)\n")
    
    try:
        results = model.train(
            data='data.yaml',           # Dataset konfigürasyonu
            epochs=100,                 # Kaç kez döndürülsün
            imgsz=640,                  # Görüntü boyutu
            device='cpu',               # GPU ID (0 = ilk GPU)
            batch=16,                   # Batch size (GPU belleğine göre)
            patience=20,                # Early stopping
            save=True,                  # Modeli kaydet
            name='custom_drone_v1',     # Model adı
            lr0=0.001,                  # Learning rate
            momentum=0.937,             # Momentum
            weight_decay=0.0005,        # Weight decay
            augment=True,               # Data augmentation
            mosaic=1.0,                 # Mosaic augmentation
            flipud=0.5,                 # Dikey çevirme (%50)
            fliplr=0.5,                 # Yatay çevirme (%50)
            degrees=15,                 # Döndürme derecesi
        )
        
        print("\n✅ Eğitim tamamlandı!")
        print(f"\n📊 Sonuçlar: runs/detect/custom_drone_v1/")
        print(f"   - best.pt    (En iyi model)")
        print(f"   - last.pt    (Son model)")
        print(f"   - results.png (Grafikler)")
        
    except Exception as e:
        print(f"❌ Eğitim hatası: {e}")
        print("\nÇözüm:")
        print("1. GPU'nuz var mı? (nvidia-smi)")
        print("2. data.yaml dosyası doğru mu?")
        print("3. Eğitim veri seti mevcut mu?")

def evaluate_model(model_path: str):
    """Eğitilmiş modeli değerlendir"""
    
    print(f"\n📊 Model Değerlendirmesi: {model_path}\n")
    
    model = YOLO(model_path)
    metrics = model.val()  # Validation set'te test et
    
    print("\n✅ Metrikleri Kontrol Et:")
    print(f"   mAP@0.5: {metrics.box.map50:.3f}")
    print(f"   mAP@0.5:0.95: {metrics.box.map:.3f}")
    print(f"   Precision: {metrics.box.p.mean():.3f}")
    print(f"   Recall: {metrics.box.r.mean():.3f}")

def use_trained_model(image_path: str, model_path: str = 'runs/detect/custom_drone_v1/weights/best.pt'):
    """Eğitilmiş modeli kullan"""
    
    import cv2
    
    print(f"\n🎬 Eğitilmiş Model ile Tespit\n")
    
    model = YOLO(model_path)
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"❌ Görüntü yüklenemedi: {image_path}")
        return
    
    print(f"📸 Görüntü: {image_path}")
    results = model.predict(frame, conf=0.3)
    
    print(f"✅ {len(results[0].boxes)} nesne tespit edildi\n")
    
    for i, box in enumerate(results[0].boxes, 1):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_names = ['Taşıt', 'İnsan', 'UAP', 'UAİ']
        print(f"{i}. {class_names[cls]} ({conf:.1%})")

def main():
    """Main fonksiyon"""
    
    print("""
SEÇENEKLER:
1. data.yaml oluştur
2. Modeli eğit
3. Modeli değerlendir
4. Eğitilmiş modelle tespit yap
    """)
    
    choice = input("Seçeneği gir (1-4): ").strip()
    
    if choice == '1':
        create_dataset_yaml()
    elif choice == '2':
        train_custom_model()
    elif choice == '3':
        model_path = input("Model yolu gir (default: runs/detect/custom_drone_v1/weights/best.pt): ").strip()
        if not model_path:
            model_path = 'runs/detect/custom_drone_v1/weights/best.pt'
        evaluate_model(model_path)
    elif choice == '4':
        image_path = input("Görüntü yolu gir: ").strip()
        model_path = input("Model yolu gir (default: best.pt): ").strip()
        if not model_path:
            model_path = 'runs/detect/custom_drone_v1/weights/best.pt'
        use_trained_model(image_path, model_path)
    else:
        print("❌ Hatalı seçenek")

if __name__ == '__main__':
    main()
