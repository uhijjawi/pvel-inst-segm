from typing import Sequence, List, Union

from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from mmengine.utils import is_str
import torch
import torch.nn.functional as F
import numpy as np
from collections.abc import Iterable


def _label_to_onehot(label: np.ndarray, num_classes: int):
    """Convert a label to onehot format tensor.
    """
    res = np.eye(num_classes)[np.array(label).reshape(-1)]
    return res.reshape(list(label.shape)+[num_classes])

@METRICS.register_module()  # register the Accuracy class to the METRICS registry
class ClassificationMetric(BaseMetric):
    """ Classification Metric Evaluator

    Default prefix: Classification

    Metrics:
        - accuracy-top1 (float): classification accuracy top-1
        - accuracy-top2 (float): classification accuracy top-2
        - precision (float): classification precision
        - recall (float): classification recall
        - f1 (float): classification f1 score
    """

    default_prefix = 'Classification'  # set default_prefix
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
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
            if len(data_sample['gt_instances']['labels']) == 0:
                continue
            result['gt'] = data_sample['gt_instances']['labels'].detach().cpu().numpy()
            result['pred'] = data_sample['pred_instances']['labels'].detach().cpu().numpy()
            result['pred_onehot'] = _label_to_onehot(result['pred'], self.num_classes)
            result['gt_onehot'] = _label_to_onehot(result['gt'], self.num_classes)

        self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        acc_top1 = np.nan
        acc_top2 = np.nan
        precision = np.nan
        recall = np.nan
        f1 = np.nan
        m = {
            'accuracy/top1': [],
            'accuracy/top2': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        # filter out empty results
        results = [r for r in results if bool(r)]
        results = [r for r in results if len(r['gt']) > 0 and len(r['pred']) > 0] # filter out empty predictions
        for r in results:
            gt = r['gt'][0] if isinstance(r['gt'], Iterable) else r['gt']
            pred = r['pred'][0] if isinstance(r['pred'], Iterable) else r['pred']
            pred2 = np.array(r['pred'])[1] if len(r['pred']) > 1 else None
            # top-1 accuracy
            correct = pred == gt
            m['accuracy/top1'].append(correct.astype(np.float32))
            # top-2 accuracy
            if pred2 is not None:
                correct_top2 = np.any(np.array([pred, pred2]) == gt)
                m['accuracy/top2'].append(correct_top2.astype(np.float32))
            # precision, recall, f1
            pred_onehot = r['pred_onehot'][0] if isinstance(r['pred_onehot'], Iterable) else r['pred_onehot']
            gt_onehot = r['gt_onehot'][0] if isinstance(r['gt_onehot'], Iterable) else r['gt_onehot']
            tp = pred_onehot * gt_onehot
            tp_sum = np.sum(tp, axis=0)
            pred_sum = np.sum(pred_onehot, axis=0)
            gt_sum = np.sum(gt_onehot, axis=0)
            _precision = tp_sum / np.clip(pred_sum, 1, None)
            _recall = tp_sum / np.clip(gt_sum, 1, None)
            _f1 = 2 * _precision * _recall / (np.clip(_precision + _recall, np.finfo(float).eps, None))
            m['precision'].append(_precision)
            m['recall'].append(_recall)
            m['f1'].append(_f1)
                    
        acc_top1 = np.mean(m['accuracy/top1'])
        acc_top2 = np.mean(m['accuracy/top2']) if len(m['accuracy/top2']) > 0 else np.nan
        precision = np.mean(m['precision']) if len(m['precision']) > 0 else np.nan
        recall = np.mean(m['recall']) if len(m['recall']) > 0 else np.nan
        f1 = np.mean(m['f1']) if len(m['f1']) > 0 else np.nan
        # Clip the values to avoid numerical instability TODO: remove when fixed
        acc_top1 = np.clip(acc_top1, 0, 1)
        acc_top2 = np.clip(acc_top2, 0, 1)
        precision = np.clip(precision, 0, 1)
        recall = np.clip(recall, 0, 1)
        f1 = np.clip(f1, 0, 1)
        return {'accuracy/top1': acc_top1, 'accuracy/top2': acc_top2, 'precision': precision, 'recall': recall, 'f1': f1}