from ultralytics import YOLO

try:
    model = YOLO("best (15).pt")
    print("\n✅ Model loaded successfully!")
    print("📋 Classes (IDs) in the model:")
    print("-" * 30)
    for id, name in model.names.items():
        print(f"ID {id}: {name}")
    print("-" * 30)
except Exception as e:
    print(f"\n❌ Error loading model: {e}")
