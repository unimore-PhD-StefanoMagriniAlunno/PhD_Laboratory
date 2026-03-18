import torch
from tqdm.notebook import tqdm
from typing import Any, Optional, Dict
from phd_laboratory.neuralnet.templates import (
    NNTemplate,
    Closure,
    Loss,
    Metrics,
    Scheduler,
)


# questa classe fa una diagnostica
def diagnostic(
    model: NNTemplate,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    closure: Closure | Loss,
    metrics_fn: Optional[Metrics] = None,
    scheduler: Scheduler = None,
    schedule_with_metrics: bool = False,
) -> dict[str, Any]:
    lr_history: list[list[float]] = [[] for _ in optimizer.param_groups]
    val_loss_history: list[Dict[str, float]] = []
    train_loss_history: list[list[float | dict[str, float]]] = []
    grad_norm_history: list[dict[str, dict[str, list[float]]]] = []
    model.to(device)
    model.train()
    # epoch loop
    for _ in tqdm(range(n_epochs), desc="Training Progress", leave=False):
        for i, param_group in enumerate(optimizer.param_groups):
            lr_history[i].append(param_group["lr"])
        epoch_grad_norms: dict[str, dict[str, list[float]]] = {
            name: {"L2": [], "L1": []} for name, _ in model.named_parameters()
        }
        epoch_losses: list[float | dict[str, float]] = []

        # batch loop
        batch: tuple[torch.Tensor, ...]
        for batch in tqdm(dataloader, desc="Training Batches", leave=False):
            # save the model state before the optimization step
            model_state = model.get_parameters()

            # set device for batch
            batch = tuple(x.to(device) for x in batch)  # OK con tuple immutabile
            # optimization
            optimizer.zero_grad()
            if isinstance(closure, Closure):
                closure.setting(model, batch)
                optimizer.step(closure)
            else:
                loss_tensor = closure(model, batch)
                loss_tensor.backward()
                optimizer.step()

            # loss in evaluation mode
            loss_dict: Dict[str, float]
            if isinstance(closure, Closure):
                loss_dict = closure.loss_fn.eval(model, batch)
            else:
                loss_dict = closure.leval(model, batch)
            epoch_losses.append(loss_dict)
            # record the stepsize
            current_model_state = model.get_parameters()
            for name, _ in model.named_parameters():
                current_param = current_model_state[name]
                previous_param = model_state[name]
                grad = current_param - previous_param
                L2_grad = torch.norm(grad, 2).item()
                L1_grad = torch.norm(grad, 1).item()
                if name not in epoch_grad_norms:
                    epoch_grad_norms[name] = {"L2": [], "L1": []}
                epoch_grad_norms[name]["L2"].append(L2_grad)
                epoch_grad_norms[name]["L1"].append(L1_grad)

        # compute metrics for the epoch
        eval_loss = metrics_fn.test(model) if metrics_fn is not None else {}
        if scheduler is not None:
            if schedule_with_metrics:
                scheduler.sstep(eval_loss)
            else:
                scheduler.sstep()
        val_loss_history.append(eval_loss)

        # save losses and gradient norms for the epoch
        train_loss_history.append(epoch_losses)
        grad_norm_history.append(epoch_grad_norms)

    return {
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "lr_history": lr_history,
        "grad_norm_history": grad_norm_history,
    }
