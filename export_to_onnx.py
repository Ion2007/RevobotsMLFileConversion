import argparse
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Export YOLO .pt model to ONNX for Sophgo CV181x / reCamera")
    parser.add_argument("--model", default="yolo11n.pt", help="Path to .pt file")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=11)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        from ultralytics import YOLO

    model = YOLO(args.model)

    export_path = model.export(
        format="onnx",
        imgsz=(args.imgsz, args.imgsz),
        opset=args.opset,
        simplify=True,
        dynamic=False,
        batch=1,
        half=False,
        int8=False,
        nms=False,
    )

    print(f"\nExported ONNX model to: {export_path}")

if __name__ == "__main__":
    main()