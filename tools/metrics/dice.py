from typing import Sequence, List

from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

import numpy as np
from collections.abc import Iterable


@METRICS.register_module()  # register the Accuracy class to the METRICS registry
class Dice(BaseMetric):
    """ Dice score Evaluator

    Default prefix: Dice

    Metrics:
        - Dice (float): Dice score
    """

    default_prefix = 'Dice'  # set default_prefix

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
            result['pred'] = data_sample['pred_instances']['masks']
            result['gt'] = data_sample['gt_instances']['masks'].masks

        self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        # Compute mask Dice
        dice = np.nan
        m = []
        results = [r for r in results if len(r['pred']) > 0]
        for r in results:
            for gt, pred in zip(r['gt'], r['pred']):
                pred = np.array(pred, dtype=np.uint8)
                gt = np.array(gt, dtype=np.uint8)
                if pred.shape != gt.shape:
                    print(
                        f"Prediction and GT masks have different shapes: {pred.shape} and {gt.shape}."
                    )
                    continue # or m.append(0)?
                intersection = np.logical_and(pred, gt)
                dice = 2 * intersection.sum() / (pred.sum() + gt.sum())
                m.append(dice)
        dice = np.mean(m)
        return {'dice': dice}