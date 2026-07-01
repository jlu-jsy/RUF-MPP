import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from cite.scheduler import build_lr_scheduler
from config import get_config
from loader import MoleculeDataset
from model import CSFA
from splitters import random_split, scaffold_split
from tensorboard_utils import (
    log_per_task_metrics,
    log_task_gradients,
    log_training_metrics,
)
from train_utils import (
    MultiTaskLossWrapper,
    eval_cls,
    eval_cls_per_task,
    eval_reg,
    train_cls,
    train_reg,
    compute_cls_metrics,
    compute_reg_metrics,

)

CLASSIFICATION_DATASETS = {
    "tox21",
    "hiv",
    "pcba",
    "muv",
    "bace",
    "bbbp",
    "toxcast",
    "sider",
    "clintox",
    "mutag",
}

DATASET_TASK_MAP = {
    "tox21": 12,
    "hiv": 1,
    "pcba": 128,
    "muv": 17,
    "bace": 1,
    "bbbp": 1,
    "toxcast": 617,
    "sider": 27,
    "clintox": 2,
    "esol": 1,
    "freesolv": 1,
    "lipo": 1,
    "mutag": 1,
    "qm7": 1,
    "qm8": 12,
    "qm9": 3,
}

def save_metrics_to_excel(metrics, log_dir, task_type, run_num, tag):
    if task_type == 'cls':
        task_names = [f'task_{i}' for i in range(len(metrics))]
        columns = ['Task', 'AUC', 'AUPR', 'F1', 'Recall', 'Precision']
        
        data = []
        for i, metric in enumerate(metrics):
            data.append([task_names[i], *metric])

        df = pd.DataFrame(data, columns=columns)

        metrics_array = np.array(metrics)
        means = np.nanmean(metrics_array, axis=0)
        stds = np.nanstd(metrics_array, axis=0)
        
        df.loc['mean'] = ['Mean', *means]
        df.loc['std'] = ['Std', *stds]
    else:
        task_names = [f'task_{i}' for i in range(len(metrics))]
        columns = ['Task', 'RMSE', 'MAE']
        
        data = []
        for i, metric in enumerate(metrics):
            data.append([task_names[i], *metric])

        df = pd.DataFrame(data, columns=columns)

        metrics_array = np.array(metrics)
        means = np.nanmean(metrics_array, axis=0)
        stds = np.nanstd(metrics_array, axis=0)
        
        df.loc['mean'] = ['Mean', *means]
        df.loc['std'] = ['Std', *stds]

    excel_file = f"{log_dir}/tb_run_{run_num}_{tag}_per_task_scatter_metrics.xlsx"
    df.to_excel(excel_file, index=False)
    print(f"[Metrics] Saved metrics to {excel_file}")

def plot_scatter_and_save_metrics(model, test_loader, device, task_type, log_dir, run_num, tag):
    try:
        print("[Scatter] Generating scatter plots and saving metrics...")
        model.eval()

        if task_type == 'cls':
            per_task_metrics, _ = compute_cls_metrics(model, device, test_loader)
            metrics = per_task_metrics.tolist()
        else:
            per_task_metrics, _ = compute_reg_metrics(model, device, test_loader)
            metrics = per_task_metrics.tolist()
        
        save_metrics_to_excel(metrics, log_dir, task_type, run_num, tag=tag)

        return True

    except Exception as e:
        print(f"[Scatter] Skip scatter plotting due to error: {e}")
        return False

def parse_rule_probs(rule_probs: str):
    probs = tuple(float(x) for x in rule_probs.split(","))
    if len(probs) != 3:
        raise ValueError("rule_probs must contain three values.")
    if abs(sum(probs) - 1.0) >= 1e-6:
        raise ValueError("rule_probs must sum to 1.0.")
    return probs


def get_task_type(dataset_name: str) -> str:
    return "cls" if dataset_name in CLASSIFICATION_DATASETS else "reg"


def get_num_tasks(dataset_name: str) -> int:
    if dataset_name not in DATASET_TASK_MAP:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    return DATASET_TASK_MAP[dataset_name]


def build_save_dir(args) -> tuple[str, str]:
    current_date = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    suffix = "Test" if args.test_if == "yes" else args.aug
    save_dir = f"run/{args.dataset}/{current_date}_{suffix}"
    os.makedirs(save_dir, exist_ok=True)
    return save_dir, current_date


def save_args(args, save_dir: str) -> None:
    args_file = os.path.join(save_dir, "args.txt")
    with open(args_file, "w") as f:
        f.write("=== Configuration ===\n")
        for key, value in sorted(vars(args).items()):
            f.write(f"{key:<20}: {value}\n")
        f.write("=" * 30)


def build_dataset_root(args, rule_mode: str, rule_probs) -> str:
    if args.aug == "llm" and args.aug_ratio > 0:
        root = f"data/{args.dataset}/aug_llm_{args.aug_ratio}"
    elif args.aug == "rule" and args.aug_ratio > 0:
        prob_tag = "_".join(str(p).replace(".", "p") for p in rule_probs)
        aug_tag = f"aug_rule_{rule_mode}_{args.aug_ratio}_prob_{prob_tag}"
        root = f"data/{args.dataset}/{aug_tag}"
    else:
        root = f"data/{args.dataset}/raw"

    os.makedirs(root, exist_ok=True)
    print(f"[Dataset Root] {root}")
    return root


def build_dataset(args, device, root: str, rule_mode: str, rule_probs):
    return MoleculeDataset(
        root=root,
        device=device,
        dataset_name=args.dataset,
        aug=args.aug,
        aug_ratio=args.aug_ratio,
        rule_mode=rule_mode,
        rule_probs=rule_probs,
    )


def get_split_seed(args, run_idx: int) -> int:
    if args.seed != -1:
        return args.seed
    return args.seed + run_idx + 1


def split_dataset(dataset, smiles_list, args, split_seed: int):
    if args.split == "scaffold":
        return scaffold_split(
            dataset,
            smiles_list,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=split_seed,
        )

    if args.split == "random":
        return random_split(
            dataset,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=split_seed,
        )

    if args.split == "balanced_scaffold":
        return scaffold_split(
            dataset,
            smiles_list,
            balanced=True,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=split_seed,
        )

    raise ValueError(f"Invalid split option: {args.split}")


def build_loaders(train_dataset, valid_dataset, test_dataset, args, device):
    use_cuda = device.type == "cuda"
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": use_cuda,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def build_model(args, num_tasks: int):
    return CSFA(
        args.num_layer,
        args.emb_dim,
        num_tasks,
        drop_ratio=args.dropout_ratio,
        JK=args.JK,
        graph_pooling=args.graph_pooling,
        gnn_type=args.gnn_type,
    )


def build_mtl_wrapper(args, task_type: str, num_tasks: int, device):
    if task_type != "cls" or args.mtl_loss == "none":
        if task_type == "cls":
            print("[MTL] Disabled: using simple average loss.")
        return None

    wrapper = MultiTaskLossWrapper(
        strategy=args.mtl_loss,
        num_tasks=num_tasks,
        temp=args.mtl_temp,
        alpha=args.mtl_alpha,
        entropy_lambda=0.01,
    ).to(device)

    print(f"[MTL] Enabled: {args.mtl_loss} (temp={args.mtl_temp}, alpha={args.mtl_alpha})")
    return wrapper


def build_optimizer(model, mtl_wrapper, args):
    optim_params = model.parameters()
    if mtl_wrapper is not None and args.mtl_loss in ["uncertainty", "uncertainty_v2", "gradnorm"]:
        optim_params = list(model.parameters()) + list(mtl_wrapper.parameters())

    return optim.Adam(optim_params, lr=args.lr, weight_decay=args.decay)


def compute_pos_weight(train_loader, num_tasks: int, device):
    pos = torch.zeros(num_tasks, dtype=torch.float64)
    neg = torch.zeros(num_tasks, dtype=torch.float64)

    for batch in train_loader:
        y = batch.y.detach().cpu().double()
        mask = ~torch.isnan(y)
        pos += ((y == 1) & mask).sum(dim=0).double()
        neg += ((y == 0) & mask).sum(dim=0).double()

    pos_weight = (neg / pos.clamp(min=1.0)).float()
    pos_weight = pos_weight.clamp(max=100.0)
    print("[pos_weight]", pos_weight.tolist())
    return pos_weight.to(device)


def init_logging_objects(task_type: str):
    if task_type == "cls":
        log_df = pd.DataFrame(
            columns=[
                "Epoch",
                "Train_AUC",
                "Train_AUPR",
                "Train_F1",
                "Train_Recall",
                "Train_Precision",
                "Val_AUC",
                "Val_AUPR",
                "Val_F1",
                "Val_Recall",
                "Val_Precision",
                "Test_AUC",
                "Test_AUPR",
                "Test_F1",
                "Test_Recall",
                "Test_Precision",
            ]
        )
        best_val = [0.0] * 5
        best_test = [0.0] * 5
        best_min_auc = -1.0
    else:
        log_df = pd.DataFrame(
            columns=[
                "Epoch",
                "Train_RMSE",
                "Train_MAE",
                "Val_RMSE",
                "Val_MAE",
                "Test_RMSE",
                "Test_MAE",
            ]
        )
        best_val = [float("inf")] * 2
        best_test = [float("inf")] * 2
        best_min_auc = None

    return log_df, best_val, best_test, best_min_auc


def append_epoch_log(log_df, epoch: int, task_type: str, train_metric, val_metric, test_metric):
    if task_type == "cls":
        metric_names = ["AUC", "AUPR", "F1", "Recall", "Precision"]
        new_row = {
            "Epoch": epoch,
            **{f"Train_{name}": value for name, value in zip(metric_names, train_metric)},
            **{f"Val_{name}": value for name, value in zip(metric_names, val_metric)},
            **{f"Test_{name}": value for name, value in zip(metric_names, test_metric)},
        }
    else:
        new_row = {
            "Epoch": epoch,
            "Train_RMSE": train_metric[0],
            "Train_MAE": train_metric[1],
            "Val_RMSE": val_metric[0],
            "Val_MAE": val_metric[1],
            "Test_RMSE": test_metric[0],
            "Test_MAE": test_metric[1],
        }

    log_df.loc[len(log_df)] = new_row


def save_run_excel(excelfile, log_df, task_type: str, best_val, best_test, best_val_epoch: int):
    with pd.ExcelWriter(excelfile, engine="openpyxl") as writer_excel:
        log_df.to_excel(writer_excel, sheet_name="Training Log", index=False)

        metric_names = ["AUC", "AUPR", "F1", "Recall", "Precision"] if task_type == "cls" else ["RMSE", "MAE"]
        summary_df = pd.DataFrame(
            {
                "Metric": metric_names,
                "Best_Val": best_val,
                "Best_Test": best_test,
            }
        )
        summary_df["Best_Val_Epoch"] = best_val_epoch
        summary_df.to_excel(writer_excel, sheet_name="Best Results", index=False)

def save_overall_summary(save_dir: str, task_type: str, all_best_metrics, all_last1_metrics):
    all_best_metrics = np.array(all_best_metrics)
    mean_metrics = np.mean(all_best_metrics, axis=0)
    std_metrics = np.std(all_best_metrics, axis=0)

    all_last1_metrics = np.array(all_last1_metrics)
    mean_last1 = np.mean(all_last1_metrics, axis=0)
    std_last1 = np.std(all_last1_metrics, axis=0)

    metric_names = ["AUC", "AUPR", "F1", "Recall", "Precision"] if task_type == "cls" else ["RMSE", "MAE"]

    overall_best_file = f"{save_dir}/ALL_RUNS_SUMMARY_best.xlsx"
    pd.DataFrame(
        {
            "Metric": metric_names,
            "Mean": mean_metrics,
            "Std": std_metrics,
        }
    ).to_excel(overall_best_file, index=False)

    overall_last1_file = f"{save_dir}/ALL_RUNS_SUMMARY_last1.xlsx"
    pd.DataFrame(
        {
            "Metric": metric_names,
            "Mean": mean_last1,
            "Std": std_last1,
        }
    ).to_excel(overall_last1_file, index=False)

    print(f"\nOverall BEST results saved to:  {overall_best_file}")
    print(f"Overall LAST1 results saved to: {overall_last1_file}")


def main():
    args = get_config()
    rule_probs = parse_rule_probs(args.rule_probs)

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    task_type = get_task_type(args.dataset)
    num_tasks = get_num_tasks(args.dataset)

    save_dir, current_date = build_save_dir(args)
    save_args(args, save_dir)

    all_best_metrics = []
    all_last1_metrics = []

    rule_mode = "prefer_non_aromatic"
    root = build_dataset_root(args, rule_mode, rule_probs)
    dataset = build_dataset(args, device, root, rule_mode, rule_probs)
    smiles_list = dataset.smiles()

    print(f"[Split] Final dataset size: {len(dataset)}, SMILES count: {len(smiles_list)}")

    for run_idx in range(args.num_runs):
        print(f"\n=== Run {run_idx + 1}/{args.num_runs} ===")

        split_seed = get_split_seed(args, run_idx)
        train_dataset, valid_dataset, test_dataset = split_dataset(
            dataset=dataset,
            smiles_list=smiles_list,
            args=args,
            split_seed=split_seed,
        )

        print(f"Train size: {len(train_dataset)}")
        print(f"Val size: {len(valid_dataset)}")
        print(f"Test size: {len(test_dataset)}")

        train_loader, val_loader, test_loader = build_loaders(
            train_dataset,
            valid_dataset,
            test_dataset,
            args,
            device,
        )

        model = build_model(args, num_tasks).to(device)
        mtl_wrapper = build_mtl_wrapper(args, task_type, num_tasks, device)

        optimizer = build_optimizer(model, mtl_wrapper, args)
        scheduler = build_lr_scheduler(optimizer, args, len(train_dataset))

        log_dir = f"{save_dir}/tb_run_{run_idx + 1}"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        output_prefix = f"{log_dir}/seed-{args.runseed}_split-{split_seed}"
        excelfile = output_prefix + ".xlsx"

        if task_type == "cls":
            pos_weight = compute_pos_weight(train_loader, num_tasks, device)
            criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        else:
            criterion = nn.MSELoss()

        log_df, best_val, best_test, best_min_auc = init_logging_objects(task_type)
        best_model_path = os.path.join(log_dir, f"best_model_run{run_idx + 1}.pt")
        best_val_epoch = 0

        for epoch in range(1, args.epochs + 1):
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{now_str}] Run {run_idx + 1}/{args.num_runs} | "
                f"Epoch {epoch}/{args.epochs} | Dataset {args.dataset} | Run ID {current_date}"
            )

            if task_type == "cls":
                train_loss = train_cls(
                    model,
                    device,
                    train_loader,
                    optimizer,
                    scheduler,
                    criterion,
                    writer,
                    epoch,
                    mtl_wrapper=mtl_wrapper,
                )
            else:
                train_loss = train_reg(
                    model,
                    device,
                    train_loader,
                    optimizer,
                    scheduler,
                    criterion,
                    writer,
                    epoch,
                )

            print(f"Train Loss: {train_loss:.4f}")

            log_task_gradients(model, writer, epoch, num_tasks, task_type)

            train_metric = eval_cls(model, device, train_loader, writer) if task_type == "cls" else eval_reg(model, device, train_loader, writer)
            val_metric = eval_cls(model, device, val_loader, writer) if task_type == "cls" else eval_reg(model, device, val_loader, writer)
            test_metric = eval_cls(model, device, test_loader, writer) if task_type == "cls" else eval_reg(model, device, test_loader, writer)

            if task_type == "cls":
                val_per_task = eval_cls_per_task(model, device, val_loader)
                val_min_auc = float(np.nanmin(val_per_task[:, 0]))

            log_training_metrics(writer, epoch, train_loss, train_metric, val_metric, test_metric, task_type)

            if task_type == "cls" and num_tasks <= 30:
                log_per_task_metrics(model, device, test_loader, writer, epoch, task_type)

            if task_type == "cls":
                if val_min_auc >= best_min_auc or (
                    val_min_auc == best_min_auc and val_metric[0] >= best_val[0]
                ):
                    best_min_auc = val_min_auc
                    best_val = val_metric.copy()
                    best_test = test_metric.copy()
                    best_val_epoch = epoch
                    torch.save(model.state_dict(), best_model_path)
            else:
                if val_metric[0] <= best_val[0]:
                    best_val = val_metric.copy()
                    best_test = test_metric.copy()
                    best_val_epoch = epoch
                    torch.save(model.state_dict(), best_model_path)

            append_epoch_log(log_df, epoch, task_type, train_metric, val_metric, test_metric)

        last1_model_path = os.path.join(log_dir, "last1_model.pt")
        torch.save(model.state_dict(), last1_model_path)

        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
            plot_scatter_and_save_metrics(
                model=model,
                test_loader=test_loader,
                device=device,
                task_type=task_type,
                log_dir=log_dir,
                run_num=run_idx + 1,
                tag="best",
            )

        if os.path.exists(last1_model_path):
            model.load_state_dict(torch.load(last1_model_path, map_location=device, weights_only=True))
            plot_scatter_and_save_metrics(
                model=model,
                test_loader=test_loader,
                device=device,
                task_type=task_type,
                log_dir=log_dir,
                run_num=run_idx + 1,
                tag="last1",
            )

            last1_metric = eval_cls(model, device, test_loader, writer=None) if task_type == "cls" else eval_reg(model, device, test_loader, writer=None)
            all_last1_metrics.append(last1_metric)

        save_run_excel(excelfile, log_df, task_type, best_val, best_test, best_val_epoch)
        all_best_metrics.append(best_test)
        writer.close()

        print(f"Run {run_idx + 1} finished. Best test metrics saved to: {excelfile}")

    save_overall_summary(save_dir, task_type, all_best_metrics, all_last1_metrics)
    print(f"\nAll {args.num_runs} runs completed!")


if __name__ == "__main__":
    main()