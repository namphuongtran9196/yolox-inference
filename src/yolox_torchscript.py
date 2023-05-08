#
# Created on Fri Mar 17 2023
#
# Copyright (c) 2023 Freelancer
# Contact: namphuongtran9196@gmail.com
#

import argparse

import cv2
import numpy as np
import torch

COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def preproc(
    img: np.ndarray,
    input_size: tuple,
    swap: tuple = (2, 0, 1),
) -> np.ndarray:
    """Preprocess an image before network input.

    Args:
        img (np.ndarray): an image of shape (H, W, C) in BGR order.
        input_size (tuple): a tuple of (input_w, input_h).
        swap (tuple, optional): a tuple of channel swap. Defaults to (2, 0, 1).

    Returns:
        np.ndarray: a preprocessed image of shape swap. Defaults to (C, H, W).
    """

    # Create a padded copy of the image
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114
    # Compute the aspect ratio of the image
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    # Resize and convert to uint8
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    # Transpose
    padded_img = np.ascontiguousarray(padded_img.transpose(swap), dtype=np.float32)

    return padded_img, r


def non_max_suppression_fast(
    boxes: np.ndarray,
    overlapThresh: float,
) -> list:
    """Non-maximum suppression.

    Args:
        boxes (np.ndarray): list of bounding boxes with shape (N, 4).
        overlapThresh (float): overlap threshold.

    Returns:
        list: indexes of the selected bounding boxes.
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return pick


def postprocess(
    predictions: np.ndarray,
    img_size: tuple,
    ratio: float,
    p6: bool = False,
    max_det: int = 100,
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.5,
) -> tuple:
    """Postprocess the predictions.
    Args:
        predictions (np.ndarray): model predictions.
        img_size (tuple): image size.
        ratio (float): ratio of image resize.
        p6 (bool, optional): whether to use P6. Defaults to False.
        max_det (int, optional): maximum number of detections. Defaults to 100.
        confidence_threshold (float, optional): confidence threshold. Defaults to 0.5.
        nms_threshold (float, optional): nms threshold. Defaults to 0.5.

    Returns:
        tuple: a tuple of (boxes, scores, labels).
    """

    # Bounding box decoding
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    predictions[..., :2] = (predictions[..., :2] + grids) * expanded_strides
    predictions[..., 2:4] = np.exp(predictions[..., 2:4]) * expanded_strides

    boxes = predictions[..., :4]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[..., 0] = boxes[..., 0] - boxes[..., 2] / 2.0
    boxes_xyxy[..., 1] = boxes[..., 1] - boxes[..., 3] / 2.0
    boxes_xyxy[..., 2] = boxes[..., 0] + boxes[..., 2] / 2.0
    boxes_xyxy[..., 3] = boxes[..., 1] + boxes[..., 3] / 2.0
    boxes_xyxy /= ratio

    # Class confidence and class prediction
    scores = predictions[..., 4:5] * predictions[..., 5:]
    scores = scores.detach().cpu().numpy()
    class_pred = np.argmax(scores, axis=-1, keepdims=True)
    class_conf = np.max(scores, axis=-1, keepdims=True)

    predictions = np.concatenate([boxes_xyxy, class_conf, class_pred], axis=-1)

    # Batch non-maximum suppression
    boxes_xyxy_ret = np.zeros((predictions.shape[0], max_det, 4))
    class_conf_ret = np.zeros((predictions.shape[0], max_det))
    class_pred_ret = np.zeros((predictions.shape[0], max_det))

    # Loop over each image
    for i in range(predictions.shape[0]):
        pred = predictions[i]
        pred = pred[pred[..., 4] > confidence_threshold]
        bboxes = pred[..., :4]
        c_conf = pred[..., 4]
        c_pred = pred[..., 5]
        # TODO: add support multi-class
        pick = non_max_suppression_fast(bboxes, nms_threshold)
        boxes_xyxy_ret[i][range(len(pick))] = bboxes[pick]
        class_conf_ret[i][range(len(pick))] = c_conf[pick]
        class_pred_ret[i][range(len(pick))] = c_pred[pick]

    return (boxes_xyxy_ret, class_conf_ret, class_pred_ret)


def main(args):
    # Load model
    model = torch.jit.load(args.model_path)
    model.eval()

    # Load sample
    img = cv2.imread(args.image_path)
    original = img.copy()

    # Preprocess
    tensor_img, ratio = preproc(img, input_size=(args.input_size, args.input_size))
    tensor_img = torch.from_numpy(tensor_img).unsqueeze(0).float()

    # Inference
    with torch.no_grad():
        outputs = model(tensor_img)
    # Postprocess
    boxes, confidence, class_index = postprocess(
        outputs,
        (args.input_size, args.input_size),
        ratio,
        confidence_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
    )

    # Visualize
    for box, sco, cls in zip(boxes[0], confidence[0], class_index[0]):
        cv2.rectangle(
            original,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            original,
            "{}:{:.2f}".format(COCO_CLASSES[int(cls)], sco),
            (int(box[0]), int(box[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    cv2.imshow("result", original)
    cv2.waitKey(0)


def make_parser():
    parser = argparse.ArgumentParser("YoloX fast inference!")
    parser.add_argument(
        "-img",
        "--image_path",
        type=str,
        default="../assets/dog.jpg",
        help="image folder path",
    )
    parser.add_argument(
        "-model",
        "--model_path",
        type=str,
        default="../models/torchscript/yolox_s.pt",
    )
    parser.add_argument(
        "-confthre",
        "--conf_threshold",
        type=float,
        default=0.3,
        help="confidence threshold",
    )
    parser.add_argument(
        "-nmsthre",
        "--nms_threshold",
        type=float,
        default=0.5,
        help="Non Maximum Suppression threshold",
    )
    parser.add_argument(
        "-ims",
        "--input_size",
        type=int,
        default=640,
        help="input size",
    )
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
