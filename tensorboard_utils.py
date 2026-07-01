import torch
import numpy as np


def _log_task_gradient_distribution(writer, total_task_grad, epoch, num_tasks):
    """Log per-task head gradients and their cross-task distribution."""
    if num_tasks <= 30:
        for task_id in range(num_tasks):
            writer.add_scalar(
                f"TaskHead_Grad/Task_{task_id + 1}",
                total_task_grad[task_id].item(),
                epoch,
            )

    if epoch % 20 == 0:
        writer.add_histogram(
            "TaskHead_Grad/Distribution_Across_Tasks",
            total_task_grad,
            epoch,
        )


def _collect_task_head_gradients(model, num_tasks):
    """Collect mean absolute gradients for each task-specific prediction head."""
    if hasattr(model, "task_heads"):
        device = next(model.parameters()).device
        total_task_grad = torch.zeros(num_tasks, device=device)

        for task_id, head in enumerate(model.task_heads):
            grads = [
                p.grad.detach().abs().mean()
                for p in head.parameters()
                if p.grad is not None
            ]
            if grads:
                total_task_grad[task_id] = torch.stack(grads).mean()

        return total_task_grad

    task_specific_grads = []

    for _, param in model.named_parameters():
        if param.grad is None:
            continue

        grad_abs = param.grad.detach().abs()

        if grad_abs.ndim >= 1 and grad_abs.shape[0] == num_tasks:
            per_task = grad_abs if grad_abs.ndim == 1 else grad_abs.mean(
                dim=tuple(range(1, grad_abs.ndim))
            )
        elif grad_abs.ndim >= 1 and grad_abs.shape[-1] == num_tasks:
            per_task = grad_abs if grad_abs.ndim == 1 else grad_abs.mean(
                dim=tuple(range(grad_abs.ndim - 1))
            )
        else:
            continue

        task_specific_grads.append(per_task)

    if not task_specific_grads:
        return None

    return torch.stack(task_specific_grads).mean(dim=0)


def log_task_gradients(model, writer, epoch, num_tasks, task_type="cls"):
    """Log task-head gradient statistics to TensorBoard."""
    if task_type.lower() == "cls":
        total_task_grad = _collect_task_head_gradients(model, num_tasks)
        if total_task_grad is not None:
            _log_task_gradient_distribution(writer, total_task_grad, epoch, num_tasks)
        return

    grad_norms = []
    prediction_layer_keywords = ("linear", "fc", "pred")

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if any(keyword in name.lower() for keyword in prediction_layer_keywords):
            grad_norms.append(param.grad.detach().norm(p=2))

    if grad_norms:
        total_norm = torch.stack(grad_norms).norm(p=2).item()
        writer.add_scalar("TaskHead_Grad/Overall_L2_Norm", total_norm, epoch)


def log_per_task_metrics(model, device, test_loader, writer, epoch, task_type="cls"):
    """Log per-task test metrics to TensorBoard."""
    if task_type.lower() != "cls":
        return

    from train_utils import eval_cls_per_task

    per_task_test = eval_cls_per_task(model, device, test_loader)

    if len(per_task_test) > 30:
        return

    for task_id, metrics in enumerate(per_task_test):
        auc = metrics[0]
        if not np.isnan(auc):
            writer.add_scalar(f"Test/Task_{task_id + 1}_AUC", auc, epoch)


def log_training_metrics(
    writer,
    epoch,
    train_loss,
    train_metric,
    val_metric,
    test_metric,
    task_type="cls",
):
    """Log aggregate training, validation, and test metrics to TensorBoard."""
    writer.add_scalar("Loss/Train-Total-Loss", train_loss, epoch)

    if task_type.lower() == "cls":
        metric_names = ("AUC", "AUPR", "F1", "Recall", "Precision")
        for idx, name in enumerate(metric_names):
            writer.add_scalar(f"Train/{name}", train_metric[idx], epoch)
            writer.add_scalar(f"Val/{name}", val_metric[idx], epoch)
            writer.add_scalar(f"Test/{name}", test_metric[idx], epoch)
        return

    writer.add_scalar("Train/RMSE", train_metric[0], epoch)
    writer.add_scalar("Train/MAE", train_metric[1], epoch)
    writer.add_scalar("Val/RMSE", val_metric[0], epoch)
    writer.add_scalar("Val/MAE", val_metric[1], epoch)
    writer.add_scalar("Test/RMSE", test_metric[0], epoch)
    writer.add_scalar("Test/MAE", test_metric[1], epoch)
