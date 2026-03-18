import torch
from typing import Callable
from torch.func import vmap, jacrev


class Divergence:
    class DivF:
        def __init__(
            self,
            fun: Callable[[torch.Tensor], torch.Tensor],
            outer_dim: torch.Size,
            vectorized: bool = False,
        ):
            self.fun = fun
            self.outer_dim = outer_dim
            self.vectorized = vectorized

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            df_x: torch.Tensor
            if self.vectorized:
                in_dim = x.shape[1:]
                df_x = vmap(jacrev(self.fun))(x)  # (N,b1,...,a1,...,a1,...)
                df_x = df_x.reshape(
                    -1, self.outer_dim.numel(), in_dim.numel(), in_dim.numel()
                )  # (N,B,A,A)
                divf_x = torch.einsum("...jii->...j", df_x)  # (N,b1,...)
                divf_x = divf_x.reshape(-1, *self.outer_dim)  # (N,b1,...)
            else:
                in_dim = x.shape
                df_x = jacrev(self.fun)(x)  # (N,b1,...,a1,...,a1,...)
                df_x = df_x.reshape(
                    self.outer_dim.numel(), in_dim.numel(), in_dim.numel()
                )  # (N,B,A,A)
                divf_x = torch.einsum("jii->j", df_x)  # (N,b1,...)
                divf_x = divf_x.reshape(*self.outer_dim)  # (N,b1,...)
            return divf_x

    def __init__(
        self,
        order: int,
        fun: Callable[[torch.Tensor], torch.Tensor],
        outer_dim: torch.Size,
        vectorized: bool = False,
    ):
        self.order = order
        self.fun = fun  # take in_dim and return outer_dim + order x in_dim
        self.outer_dim = outer_dim
        self.vectorized = vectorized

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.order == 0:
            if self.vectorized:
                return vmap(self.fun)(x)
            else:
                return self.fun(x)
        mult_divf = self.fun
        if self.vectorized:
            in_dim = x.shape[1:]
        else:
            in_dim = x.shape
        for i in range(1, self.order):
            current_outer_dim = self.outer_dim + (self.order - i) * in_dim
            mult_divf = Divergence.DivF(mult_divf, current_outer_dim)
        return Divergence.DivF(mult_divf, self.outer_dim, self.vectorized)(x)


class Gradient:
    class GradF:
        def __init__(
            self, fun: Callable[[torch.Tensor], torch.Tensor], vectorized: bool = False
        ):
            self.fun = fun
            self.vectorized = vectorized

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            df_x: torch.Tensor
            if self.vectorized:
                df_x = vmap(jacrev(self.fun))(x)  # (N,b1,...,a1,...,a1,...)
            else:
                df_x = jacrev(self.fun)(x)  # (N,b1,...,a1,...,a1,...)
            return df_x

    def __init__(
        self,
        order: int,
        fun: Callable[[torch.Tensor], torch.Tensor],
        vectorized: bool = False,
    ):
        self.order = order
        self.fun = fun
        self.vectorized = vectorized

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.order == 0:
            if self.vectorized:
                return vmap(self.fun)(x)
            else:
                return self.fun(x)
        mult_divf = self.fun
        for _ in range(1, self.order):
            mult_divf = Gradient.GradF(mult_divf)
        return Gradient.GradF(mult_divf, self.vectorized)(x)
