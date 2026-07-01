import math

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from tqdm import tqdm


class MultiTaskLossWrapper(nn.Module):
    """Multi-task loss wrapper supporting common task-weighting strategies."""

    def __init__(
        self,
        strategy="equal",
        num_tasks=1,
        temp=2.0,
        alpha=1.5,
        entropy_lambda=0.01,
    ):
        super().__init__()
        self.strategy = strategy
        self.num_tasks = num_tasks
        self.temp = temp
        self.alpha = alpha
        self.entropy_lambda = entropy_lambda

        if strategy in ["uncertainty", "uncertainty_v2"]:
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))
            self.register_parameter("weights", None)
        elif strategy == "gradnorm":
            self.weights = nn.Parameter(torch.ones(num_tasks))
            self.register_parameter("log_vars", None)
        else:
            self.register_parameter("log_vars", None)
            self.register_parameter("weights", None)

        self.register_buffer("prev_loss", None)
        self.register_buffer("prev_prev_loss", None)
        self.register_buffer("initial_losses", None)

    @staticmethod
    def _as_task_tensor(losses):
        if losses is None:
            return None
        if isinstance(losses, torch.Tensor):
            return losses.reshape(-1) if losses.ndim == 0 else losses
        return torch.stack(list(losses))

    def forward(self, losses=None, epoch=None, loss_matrix=None, mask=None, attn_logits=None):
        losses = self._as_task_tensor(losses)

        if self.strategy == "equal":
            return losses.mean()

        if self.strategy == "uncertainty":
            precision = torch.exp(-self.log_vars)
            return torch.sum(precision * losses + self.log_vars)

        if self.strategy == "uncertainty_v2":
            weights = torch.softmax(self.log_vars / self.temp, dim=0)
            return (weights * losses).sum() + 0.001 * torch.sum(self.log_vars)

        if self.strategy == "dwa":
            if epoch is None or epoch <= 2 or self.prev_loss is None or self.prev_prev_loss is None:
                return losses.mean()
            ratio = self.prev_loss / (self.prev_prev_loss + 1e-8)
            weights = self.num_tasks * torch.softmax(ratio / self.temp, dim=0)
            return (weights * losses).sum() / self.num_tasks

        if self.strategy == "gradnorm":
            if epoch == 1 and self.initial_losses is None:
                self.initial_losses = losses.detach()
            return losses.mean()

        if self.strategy == "pcgrad":
            return losses.mean()

        if self.strategy == "attn":
            if loss_matrix is None or mask is None or attn_logits is None:
                raise ValueError("attn strategy requires loss_matrix, mask, and attn_logits.")

            eps = 1e-8
            weights = torch.softmax(attn_logits / self.temp, dim=1)
            weights = weights * mask
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=eps)

            weighted_loss = loss_matrix * weights * mask
            main_loss = weighted_loss.sum() / mask.sum().clamp(min=1.0)

            num_tasks = weights.size(1)
            uniform = torch.full_like(weights, 1.0 / num_tasks)
            kl = (weights * (torch.log(weights + eps) - torch.log(uniform + eps))).sum(dim=1).mean()
            return main_loss + self.entropy_lambda * kl

        raise ValueError(f"Unknown mtl_loss: {self.strategy}")

    def update_dwa_history(self, losses, epoch):
        """Update the loss history used by DWA."""
        if epoch == 1:
            self.prev_loss = losses.detach()
        elif epoch == 2:
            self.prev_prev_loss = self.prev_loss
            self.prev_loss = losses.detach()
        elif epoch > 2:
            self.prev_prev_loss = self.prev_loss
            self.prev_loss = losses.detach()

    def compute_gradnorm_loss(self, task_losses, task_grad_norms):
        """Compute the auxiliary GradNorm objective."""
        if self.initial_losses is None or self.strategy != "gradnorm":
            return None

        loss_ratio = task_losses.detach() / (self.initial_losses + 1e-8)
        target_ratio = loss_ratio ** self.alpha
        avg_grad_norm = task_grad_norms.mean().detach()
        target_grad_norms = avg_grad_norm * target_ratio
        return torch.abs(self.weights * task_grad_norms - target_grad_norms).sum()

    def apply_pcgrad(self, task_gradients):
        """Project conflicting task gradients following PCGrad."""
        projected_gradients = []

        for i, grad in enumerate(task_gradients):
            grad_i = grad.clone()
            for j, grad_j in enumerate(task_gradients):
                if i == j:
                    continue
                dot_product = torch.dot(grad_i.view(-1), grad_j.view(-1))
                if dot_product < 0:
                    grad_i = grad_i - (dot_product / (torch.norm(grad_j) ** 2 + 1e-8)) * grad_j
            projected_gradients.append(grad_i)

        return projected_gradients


def _set_return_attn(model, enabled):
    if hasattr(model, "return_attn"):
        model.return_attn = enabled


def _parse_model_output(output):
    pred = output[0] if isinstance(output, (tuple, list)) else output
    attn_logits = None

    if isinstance(output, (tuple, list)):
        if len(output) == 2 and not isinstance(output[1], dict):
            attn_logits = output[1]
        elif len(output) >= 3:
            attn_logits = output[-1]

    return pred, attn_logits


def _flatten_model_gradients(model):
    grad_vec = []
    for param in model.parameters():
        if param.grad is not None:
            grad_vec.append(param.grad.view(-1))
        else:
            grad_vec.append(torch.zeros_like(param.view(-1)))
    return torch.cat(grad_vec)


def _assign_flat_gradient(model, flat_grad):
    offset = 0
    for param in model.parameters():
        param_size = param.numel()
        param.grad = flat_grad[offset:offset + param_size].view(param.shape).clone()
        offset += param_size


def _masked_mean(loss_matrix, mask):
    return (loss_matrix * mask).sum() / mask.sum().clamp(min=1.0)


def train_cls(model, device, loader, optimizer, scheduler, criterion, writer, epoch, mtl_wrapper=None):
    model.train()
    total_loss = 0.0

    num_tasks = model.num_tasks if hasattr(model, "num_tasks") else loader.dataset[0].y.size(0)
    strategy = mtl_wrapper.strategy if mtl_wrapper is not None else "equal"

    task_loss_sum = torch.zeros(num_tasks, dtype=torch.float64, device=device)
    task_count = torch.zeros(num_tasks, dtype=torch.float64, device=device)

    for batch in tqdm(loader, desc="Train Iteration"):
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad()

        _set_return_attn(model, strategy == "attn")
        pred, attn_logits = _parse_model_output(model(batch))

        y = batch.y
        is_valid = torch.isfinite(y).float()
        y_clean = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        loss_matrix = criterion(pred.double(), y_clean.double())

        per_task_losses = torch.zeros(num_tasks, device=device, dtype=loss_matrix.dtype)
        valid_task_ids = []
        for task_id in range(num_tasks):
            mask_t = is_valid[:, task_id]
            if mask_t.sum() > 0:
                task_loss = (loss_matrix[:, task_id] * mask_t).sum() / mask_t.sum()
                per_task_losses[task_id] = task_loss
                valid_task_ids.append(task_id)

                if num_tasks < 30:
                    task_loss_sum[task_id] += (loss_matrix[:, task_id] * mask_t).sum().double()
                    task_count[task_id] += mask_t.sum().double()

        if strategy == "pcgrad":
            task_gradients = []
            for task_id in valid_task_ids:
                optimizer.zero_grad()
                per_task_losses[task_id].backward(retain_graph=True)
                task_gradients.append(_flatten_model_gradients(model))

            optimizer.zero_grad()
            if task_gradients:
                projected_gradients = mtl_wrapper.apply_pcgrad(task_gradients)
                avg_grad = sum(projected_gradients) / len(projected_gradients)
                _assign_flat_gradient(model, avg_grad)

            total_loss_val = per_task_losses[valid_task_ids].mean().item() if valid_task_ids else 0.0
            optimizer.step()

        elif strategy == "gradnorm":
            task_grad_norms = torch.zeros(num_tasks, device=device)
            for task_id in valid_task_ids:
                optimizer.zero_grad()
                per_task_losses[task_id].backward(retain_graph=True)

                grad_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += torch.norm(param.grad) ** 2
                task_grad_norms[task_id] = torch.sqrt(grad_norm + 1e-8)

            gradnorm_loss = mtl_wrapper.compute_gradnorm_loss(per_task_losses, task_grad_norms)
            if gradnorm_loss is not None:
                optimizer.zero_grad()
                gradnorm_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            loss = mtl_wrapper(per_task_losses, epoch=epoch) if mtl_wrapper is not None else _masked_mean(loss_matrix, is_valid)
            loss.backward()
            total_loss_val = loss.item()

        else:
            if mtl_wrapper is not None:
                if strategy == "attn":
                    loss = mtl_wrapper(
                        epoch=epoch,
                        loss_matrix=loss_matrix,
                        mask=is_valid,
                        attn_logits=attn_logits,
                    )
                else:
                    loss = mtl_wrapper(per_task_losses, epoch=epoch)
            else:
                loss = _masked_mean(loss_matrix, is_valid)

            loss.backward()
            total_loss_val = loss.item()

        if strategy != "pcgrad":
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += total_loss_val

        if strategy == "dwa" and mtl_wrapper is not None:
            mtl_wrapper.update_dwa_history(per_task_losses.detach(), epoch)

    if writer is not None and num_tasks < 30:
        for task_id in range(num_tasks):
            epoch_task_loss = (task_loss_sum[task_id] / task_count[task_id]).item() if task_count[task_id] > 0 else float("nan")
            writer.add_scalar(f"Loss/Train_Task_{task_id + 1}_Loss", epoch_task_loss, epoch)

    return total_loss


def train_reg(model, device, loader, optimizer, scheduler, criterion, writer, epoch):
    model.train()
    total_loss = 0.0

    first_batch = next(iter(loader))
    num_tasks = first_batch.y.size(1)

    task_loss_sum = torch.zeros(num_tasks, dtype=torch.float64, device=device)
    task_count = torch.zeros(num_tasks, dtype=torch.float64, device=device)

    for batch in tqdm(loader, desc="Train Iteration"):
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad()

        pred, _ = _parse_model_output(model(batch))
        y = batch.y

        loss_matrix = (pred - y) ** 2
        for task_id in range(num_tasks):
            task_loss_sum[task_id] += loss_matrix[:, task_id].sum()
            task_count[task_id] += loss_matrix[:, task_id].shape[0]

        loss = loss_matrix.mean()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()

    if writer is not None:
        for task_id in range(num_tasks):
            epoch_task_loss = (task_loss_sum[task_id] / task_count[task_id]).item() if task_count[task_id] > 0 else float("nan")
            writer.add_scalar(f"Loss/Train_Task_{task_id + 1}_Loss", epoch_task_loss, epoch)

    return total_loss


def evaluate_cls(y_true, y_score):
    """Compute binary classification metrics for one task."""
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float("nan")

    try:
        aupr = average_precision_score(y_true, y_score)
    except Exception:
        aupr = float("nan")

    probs = 1.0 / (1.0 + np.exp(-y_score))
    y_pred = (probs >= 0.5).astype(int)

    try:
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
    except Exception:
        f1 = recall = precision = float("nan")

    return auc, aupr, f1, recall, precision


def evaluate_reg(y_true, y_score):
    """Compute regression metrics for one task."""
    rmse = math.sqrt(mean_squared_error(y_true, y_score))
    mae = mean_absolute_error(y_true, y_score)
    return rmse, mae


def compute_cls_metrics(model, device, loader):
    """Compute per-task and averaged multi-task classification metrics."""
    model.eval()
    _set_return_attn(model, False)

    y_true_list = []
    y_scores_list = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            pred, _ = _parse_model_output(model(batch))
            y_true_list.append(batch.y.view(pred.shape))
            y_scores_list.append(pred)

    y_true = torch.cat(y_true_list, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores_list, dim=0).cpu().numpy()

    per_task_metrics = []
    for task_id in range(y_true.shape[1]):
        target = y_true[:, task_id]
        score = y_scores[:, task_id]
        labeled = ~np.isnan(target)
        target_valid = target[labeled]
        score_valid = score[labeled]

        if len(target_valid) > 0 and np.any(target_valid == 1) and np.any(target_valid == 0):
            per_task_metrics.append(evaluate_cls(target_valid, score_valid))
        else:
            per_task_metrics.append((float("nan"),) * 5)

    per_task_metrics = np.array(per_task_metrics)
    mean_metrics = np.nanmean(per_task_metrics, axis=0)
    return per_task_metrics, mean_metrics


def eval_cls(model, device, loader, writer=None):
    _, mean_metrics = compute_cls_metrics(model, device, loader)
    return mean_metrics


def eval_cls_per_task(model, device, loader):
    per_task_metrics, _ = compute_cls_metrics(model, device, loader)
    return per_task_metrics


def compute_reg_metrics(model, device, loader):
    """Compute per-task and averaged multi-task regression metrics."""
    model.eval()
    _set_return_attn(model, False)

    y_true_list = []
    y_scores_list = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            pred, _ = _parse_model_output(model(batch))
            y_true_list.append(batch.y)
            y_scores_list.append(pred)

    y_true = torch.cat(y_true_list, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores_list, dim=0).cpu().numpy()

    per_task_metrics = []
    for task_id in range(y_true.shape[1]):
        target = y_true[:, task_id]
        score = y_scores[:, task_id]
        labeled = ~np.isnan(target)
        target_valid = target[labeled]
        score_valid = score[labeled]

        if len(target_valid) > 0:
            per_task_metrics.append(evaluate_reg(target_valid, score_valid))
        else:
            per_task_metrics.append((float("nan"),) * 2)

    per_task_metrics = np.array(per_task_metrics)
    mean_metrics = np.nanmean(per_task_metrics, axis=0)
    return per_task_metrics, mean_metrics


def eval_reg(model, device, loader, writer=None):
    _, mean_metrics = compute_reg_metrics(model, device, loader)
    return mean_metrics


def eval_reg_per_task(model, device, loader):
    per_task_metrics, _ = compute_reg_metrics(model, device, loader)
    return per_task_metrics
