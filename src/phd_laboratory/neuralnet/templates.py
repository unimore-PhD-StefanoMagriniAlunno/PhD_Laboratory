from abc import ABC
from typing import Any, Optional, Tuple
import torch
from abc import ABC, abstractmethod


class NNTemplate(torch.nn.Module, ABC):
    net: torch.nn.Sequential

    @torch.no_grad()
    def get_covariances(self, x: torch.Tensor) -> dict[str, dict[str, Any]]:
        covariances: dict[str, dict[str, Any]] = {}
        for i in range(len(self.net)):
            layer = self.net[i]
            x = layer(x)
            name = f"layer_{i}_{layer.__class__.__name__}"
            C = x.T.cov()
            # compute eigenvalues of C
            if C.numel() == 1:
                eigv = C.clone()
                eigv = torch.clamp(eigv, min=0.0)
                trace = eigv.item()
            else:
                eigv = torch.linalg.eigvalsh(C)
                eigv = torch.clamp(eigv, min=0.0)
                trace = torch.trace(C).item()
            covariances[name] = {
                "layer_name": name,
                "covariance_matrix": C,
                "eigenvalues": eigv,
                "trace": trace,
            }
        return covariances

    @torch.no_grad()
    def get_parameters(self) -> dict[str, torch.Tensor]:
        parameters: dict[str, torch.Tensor] = {}
        for name, param in self.named_parameters():
            parameters[name] = param.data.clone()
        return parameters


class Loss(ABC, torch.nn.Module):

    def __init__(self):
        raise NotImplementedError("Loss function must implement the __init__ method")

    @abstractmethod
    def __call__(
        self, model: NNTemplate, batch: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        raise NotImplementedError("Loss function must implement the __call__ method")

    @abstractmethod
    def leval(
        self, model: NNTemplate, batch: tuple[torch.Tensor, ...]
    ) -> dict[str, float]:
        raise NotImplementedError("Loss function must implement the eval method")

    @property
    def loss_fn(self) -> "Loss":
        return self


class Closure(ABC):
    def __init__(self, loss_fn: Loss):
        self.loss_fn = loss_fn

    def setting(self, model: NNTemplate, batch: tuple[torch.Tensor]):
        self.model = model
        self.batch = batch

    # Closure definition
    def __call__(self):
        loss = self.loss_fn(self.model, self.batch)
        loss.backward()
        return loss


class Metrics(ABC):
    def __init__(
        self,
        metrics_fn: Loss,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ):
        self.metrics_fn = metrics_fn
        self.dataloader = dataloader
        self.device = device

    def test(self, model: NNTemplate) -> dict[str, float]:
        batch: Tuple[torch.Tensor, ...] = next(iter(self.dataloader))
        batch = tuple(x.to(self.device) for x in batch)
        return self.metrics_fn.leval(model, batch)


class Scheduler(ABC):
    def __init__(self, scheduler: torch.optim.lr_scheduler._LRScheduler):
        self.scheduler = scheduler

    @abstractmethod
    def step(self, metrics: Optional[Any] = None):
        raise NotImplementedError("Scheduler must implement the step method")
