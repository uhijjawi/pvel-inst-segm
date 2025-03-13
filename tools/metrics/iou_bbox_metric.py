from typing import Sequence, List

from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

import numpy as np
from collections.abc import Iterable

def _compute_iou(gt_bbox: np.ndarray, pred_bbox: np.ndarray):
    # Computes IoU between two bounding boxes following the COCO format [x, y, w, h]
    # Compute intersection
    x1 = max(gt_bbox[0], pred_bbox[0])
    y1 = max(gt_bbox[1], pred_bbox[1])
    x2 = min(gt_bbox[0] + gt_bbox[2], pred_bbox[0] + pred_bbox[2])
    y2 = min(gt_bbox[1] + gt_bbox[3], pred_bbox[1] + pred_bbox[3])
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    # Compute union
    y_true_area = (gt_bbox[2] + 1) * (gt_bbox[3] + 1)
    y_pred_area = (pred_bbox[2] + 1) * (pred_bbox[3] + 1)
    union = y_true_area + y_pred_area - intersection
    return intersection / (union + 1e-6)

@METRICS.register_module()  # register the Accuracy class to the METRICS registry
class IoUBBoxMetric(BaseMetric):
    """ BBox F1-Score Evaluator

    Default prefix: BBox F1-Score

    Metrics:
        - f1 (float): BBox F1-Score
        - f1_thr (float): BBox F1-Score thresholded
    """

    default_prefix = 'IoU/BBox'  # set default_prefix
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        super().__init__()

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]):
        """Process one batch of data and predictions. The processed
        Results should be stored in `self.results`, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            result = {}
            result['pred'] = data_sample['pred_instances']['bboxes'].detach().cpu().numpy()
            result['gt'] = data_sample['gt_instances']['bboxes'].detach().cpu().numpy()

        self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        f1 = np.nan
        mIoU = np.nan
        m = {
            'f1': [],
            'mIoU': [],
        }
        tp = 0
        fp = 0
        fn = 0
        matched_gt_indices = set()
        results = [r for r in results if len(r['pred']) > 0]
        for r in results:
            for pred in r['pred']:
                best_iou = 0.0
                best_gt_index = -1
                for gt_index, gt in enumerate(r['gt']):
                    iou = _compute_iou(gt, pred)
                    m['mIoU'].append(iou)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_index = gt_index
                if best_iou > self.threshold:
                    tp += 1
                    matched_gt_indices.add(best_gt_index)
                else:
                    fp += 1
            fn = len(r['gt']) - len(matched_gt_indices)
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            _f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            m['f1'].append(_f1)
        f1 = np.mean(m['f1'])
        mIoU = np.mean(m['mIoU'])
        return {f"f1@thr={self.threshold}": f1, 'mIoU': mIoU}