import argparse
import logging

import numpy as np
import onnx
import onnx_graphsurgeon as gs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Convert yolox onnx for autoware.universe",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "model",
        type=str,
        help="onnx file to modify",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        type=str,
        help="onnx file to save",
    )
    parser.add_argument(
        "--iou-threshold",
        default=0.3,
        type=float,
        help="iou threshold",
    )
    parser.add_argument(
        "--score-threshold",
        default=0.3,
        type=float,
        help="score threshold",
    )
    parser.add_argument(
        "--max-output-boxes",
        default=100,
        type=int,
        help="max output boxes",
    )

    args = parser.parse_args()

    output = args.output
    if output is None:
        output = args.model

    graph = gs.import_onnx(onnx.load(args.model))
    scatter_node = graph.outputs[0].inputs[0]
    box_slice_starts = gs.Constant(name="box_slice_starts", values=np.array([0], dtype=np.int64))
    box_slice_ends = gs.Constant(name="box_slice_ends", values=np.array([4], dtype=np.int64))
    box_slice_axes = gs.Constant(name="box_slice_axes", values=np.array([2], dtype=np.int64))
    box_slice_steps = gs.Constant(name="box_slice_steps", values=np.array([1], dtype=np.int64))
    boxes = gs.Variable(
        name="boxes",
        dtype=np.float32,
        shape=(scatter_node.outputs[0].shape[0], scatter_node.outputs[0].shape[1], 4),
    )
    box_slice_node = gs.Node(
        "Slice",
        inputs=[
            scatter_node.outputs[0],
            box_slice_starts,
            box_slice_ends,
            box_slice_axes,
            box_slice_steps,
        ],
        outputs=[boxes],
    )
    graph.nodes.append(box_slice_node)
    class_slice_starts = gs.Constant(
        name="class_slice_starts", values=np.array([5], dtype=np.int64)
    )
    class_slice_ends = gs.Constant(
        name="class_slice_ends", values=np.array([scatter_node.outputs[0].shape[2]], dtype=np.int64)
    )
    class_slice_axes = gs.Constant(name="class_slice_axes", values=np.array([2], dtype=np.int64))
    class_slice_steps = gs.Constant(name="class_slice_steps", values=np.array([1], dtype=np.int64))
    classes = gs.Variable(
        name="classes",
        dtype=np.float32,
        shape=(
            scatter_node.outputs[0].shape[0],
            scatter_node.outputs[0].shape[1],
            scatter_node.outputs[0].shape[2] - 5,
        ),
    )
    class_slice_node = gs.Node(
        "Slice",
        inputs=[
            scatter_node.outputs[0],
            class_slice_starts,
            class_slice_ends,
            class_slice_axes,
            class_slice_steps,
        ],
        outputs=[classes],
    )
    graph.nodes.append(class_slice_node)
    gather_indices = gs.Constant(name="gather_indices", values=np.array(4, dtype=np.int64))
    gather_output = gs.Variable(
        name="gather_output",
        dtype=np.float32,
        shape=(scatter_node.outputs[0].shape[0], scatter_node.outputs[0].shape[1]),
    )
    gather_node = gs.Node(
        "Gather",
        attrs={
            "axis": 2,
        },
        inputs=[scatter_node.outputs[0], gather_indices],
        outputs=[gather_output],
    )
    graph.nodes.append(gather_node)
    unsqueeze_output = gs.Variable(
        name="unsqueeze_output",
        dtype=np.float32,
        shape=(scatter_node.outputs[0].shape[0], scatter_node.outputs[0].shape[1], 1),
    )
    unsqueeze_node = gs.Node(
        "Unsqueeze",
        attrs={
            "axes": [2],
        },
        inputs=[gather_output],
        outputs=[unsqueeze_output],
    )
    graph.nodes.append(unsqueeze_node)
    scores = gs.Variable(
        name="scores",
        dtype=np.float32,
        shape=(
            scatter_node.outputs[0].shape[0],
            scatter_node.outputs[0].shape[1],
            scatter_node.outputs[0].shape[2] - 5,
        ),
    )
    mul_node = gs.Node(
        "Mul",
        inputs=[
            unsqueeze_output,
            classes,
        ],
        outputs=[scores],
    )
    graph.nodes.append(mul_node)
    num_detections = gs.Variable(
        name="num_detections", dtype=np.int32, shape=(scatter_node.outputs[0].shape[0], 1)
    )
    detection_boxes = gs.Variable(
        name="detection_boxes",
        dtype=np.float32,
        shape=(scatter_node.outputs[0].shape[0], args.max_output_boxes, 4),
    )
    detection_scores = gs.Variable(
        name="detection_scores",
        dtype=np.float32,
        shape=(scatter_node.outputs[0].shape[0], args.max_output_boxes),
    )
    detection_classes = gs.Variable(
        name="detection_classes",
        dtype=np.int32,
        shape=(scatter_node.outputs[0].shape[0], args.max_output_boxes),
    )
    nms_node = gs.Node(
        "EfficientNMS_TRT",
        attrs={
            "background_class": -1,
            "box_coding": 1,
            "iou_threshold": args.iou_threshold,
            "max_output_boxes": args.max_output_boxes,
            "plugin_namespace": "",
            "plugin_version": "1",
            "score_activation": 0,
            "score_threshold": args.score_threshold,
        },
        inputs=[
            boxes,
            scores,
        ],
        outputs=[
            num_detections,
            detection_boxes,
            detection_scores,
            detection_classes,
        ],
    )
    graph.nodes.append(nms_node)
    graph.outputs = [
        num_detections,
        detection_boxes,
        detection_scores,
        detection_classes,
    ]
    graph.cleanup().toposort()
    onnx.save_model(gs.export_onnx(graph), output)

    logger.info(f"Saving the ONNX model to {output}")


if __name__ == "__main__":
    main()
