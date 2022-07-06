# yolox_onnx_modifier

attach nms layer to yolox onnx file.

## Install

```bash
pip install git+ssh://git@github.com/wep21/yolox_onnx_modifier.git
```

## Create yolox onnx file

See <https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime>.
You need to specify `--decode_in_inference` option.

## Usage

```bash
yolox_onnx_modifier <your yolox onnx file> -o <output onnx file>
```
