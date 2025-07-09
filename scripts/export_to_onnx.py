import os
import yaml
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from patchmatch.models import PatchMatchTripletNetwork


def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def export_to_onnx(model, dummy_input, export_path):
    print(f"[INFO] Exporting model to {export_path}")
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        input_names=["input"],
        output_names=["embedding"],
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}}
    )
    print("[INFO] ONNX export completed.")


def apply_quantization(onnx_path, quantized_path):
    print(f"[INFO] Applying quantization to {onnx_path}")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QInt8
    )
    print(f"[INFO] Quantized model saved to {quantized_path}")


def validate_onnx_model(onnx_path):
    print(f"[INFO] Validating ONNX model at {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[INFO] ONNX model is valid.")


def main():
    config = load_config()

    if not config.get("onnx", {}).get("export_enabled", False):
        print("[WARN] ONNX export not enabled in config.yaml. Skipping.")
        return

    model_config = config["model"]
    onnx_config = config["onnx"]
    input_size = tuple(model_config.get("input_shape", [40, 40, 1])[:2])  # (H, W)

    # Load model
    full_model = PatchMatchTripletNetwork(embedding_dim=model_config["embedding_dim"])
    checkpoint = torch.load(onnx_config["checkpoint_path"], map_location="cpu")
    full_model.load_state_dict(checkpoint["model_state"])
    model = full_model.encoder.eval()
    print(f"[INFO] Loaded model from {onnx_config['checkpoint_path']}")

    # Create dummy input
    dummy_input = torch.randn(1, 1, *input_size)

    # Export to ONNX
    export_to_onnx(model, dummy_input, onnx_config["output_onnx"])

    # Validate ONNX
    validate_onnx_model(onnx_config["output_onnx"])

    # Apply quantization
    if onnx_config.get("quantize", False):
        apply_quantization(onnx_config["output_onnx"], onnx_config["output_quant_onnx"])
        validate_onnx_model(onnx_config["output_quant_onnx"])


if __name__ == "__main__":
    main()
