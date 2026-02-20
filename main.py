import torch
import matplotlib.pyplot as plt
import numpy as np

y_true = torch.zeros((10, 10))
y_true[:, 3] = 1

y_pred = y_true.clone()
y_pred[:, 3][5] = 0
# print(f"{y_pred=}")
# print(f"{y_true=}")

overlap = (y_pred * y_true).sum()
total_true = y_true.sum()
total_pred = y_pred.sum()

dice_score = (2 * overlap) / (total_true + total_pred)
# print(f"{dice_score=}")


