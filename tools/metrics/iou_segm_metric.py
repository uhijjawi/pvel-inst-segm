from typing import Sequence, List

from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

import numpy as np
from collections.abc import Iterable
import torch

def _compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """
    Calculate Intersection over Union (IoU) for a single pair of predicted and ground truth masks.

    Args:
        pred_mask (np.ndarray): Predicted segmentation mask.
        gt_mask (np.ndarray): Ground truth segmentation mask.

    Returns:
        float: IoU score.
    """
    # Computes IoU between two masks 
    intersection = np.logical_and(pred_mask, gt_mask).sum() # Compute intersection
    union = np.logical_or(pred_mask, gt_mask).sum() # Compute union
    
    iou = intersection / union if union != 0 else 0
    return iou

def _compute_miou(pred_masks: np.ndarray, gt_masks: np.ndarray):
    """
    Calculate Mean Intersection over Union (mIoU) for multiple pairs of predicted and ground truth masks.

    Args:
        pred_masks (list of np.ndarray): List of predicted segmentation masks.
        gt_masks (list of np.ndarray): List of ground truth segmentation masks.

    Returns:
        float: mIoU score.
    """
    iou_scores = [_compute_iou(pred_mask, gt_mask) for pred_mask, gt_mask in zip(pred_masks, gt_masks)]
    miou = np.mean(iou_scores)
    return miou


def calculate_best_miou(pred_masks, gt_masks):
    best_ious = []
    for gt in gt_masks:
        best_iou = 0.0
        for pred in pred_masks:
            iou = _compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
        best_ious.append(best_iou)
    miou = np.mean(best_ious)
    return best_ious, miou


@METRICS.register_module()  # register the Accuracy class to the METRICS registry
class IoUSegmMetric(BaseMetric):
    """ Mask F1-Score Evaluator

    Default prefix: Mask F1-Score

    Metrics:
        - f1 (float): Mask F1-Score
        - f1_thr (float): Mask F1-Score thresholded
    """

    default_prefix = 'IoU/Segm'  # set default_prefix
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
            # Assuming BitmapMasks has a method to get the mask data as a NumPy array
            gt_masks_np = data_sample['gt_instances']['masks'].to_ndarray()  # Replace with the actual method to get the mask data
            gt_masks_tensor = torch.tensor(gt_masks_np)
            result['gt'] = gt_masks_tensor.detach().cpu().numpy()

            result['pred'] = data_sample['pred_instances']['masks'].detach().cpu().numpy()
            # result['gt'] = data_sample['gt_instances']['masks'].detach().cpu().numpy()

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
        results = [r for r in results if len(r['pred']) > 0] # filters the results list to include only those results that have at least one prediction mask (r['pred']).
        for r in results: # iterates over each result in the filtered results list.
            best_ious, miou_per_r = calculate_best_miou(r['pred'], r['gt'])
            m['mIoU'].append(miou_per_r)
            
        mIoU = np.mean(m['mIoU']) # for all results 
        print("HI FROM IOU_SEGM_METRIC.PY . THIS IS THE CALCULATED MIOU: ", mIoU)
        return {f"f1@thr={self.threshold}": f1, 'mIoU': mIoU}