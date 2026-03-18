from phd_laboratory.fokker_planck import Model
from phd_laboratory.differential import Gradient
import torch
from typing import Callable, Optional, Tuple, cast


class Gaussian:

    def __init__(
        self,
        potential: Callable[[torch.Tensor], torch.Tensor],
        rotational_matrix: torch.Tensor,
        sigma: float,
        d_U: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,  # to define drift
        d2_U: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,  # to define u_t
        L: Optional[
            Callable[[torch.Tensor, float], Tuple[torch.Tensor, torch.Tensor]]
        ] = None,  # to define u_t
    ) -> None:
        """This class implements the Fokker-Planck model with Gaussian steady-state distribution.

        In particular, the steady-state distribution is given by :math:`p(x) \\propto \\exp(-2 \\frac{U(\\left\\|x\\right\\|^2}{\\sigma^2}))` where :math:`U` is the potential function provided by the user.

        The steady-state distribution satisfies the Fokker-Planck equation with drift and diffusion defined as follows:
        - drift: :math:`\\text{drift}(x, t) = -2 d_U(\\left\\|x\\right\\|^2) x + R x` where :math:`d_U` is the derivative of the potential function and :math:`R` is a rotational matrix provided by the user.
        - diffusion: :math:`\\text{diffusion}(x, t) = \\sigma^2 I` where :math:`\\sigma` is the diffusion coefficient provided by the user and :math:`I` is the identity matrix.

        Parameters
        ----------
        potential : Callable[[torch.Tensor], torch.Tensor]
            It implements the potential function :math:`U(r^2)` where :math:`r^2 = \\left\\|x\\right\\|^2`.
        rotational_matrix : torch.Tensor
            It implements the rotational matrix :math:`R`.
        sigma : float
            It implements the diffusion coefficient.
        d_U : Optional[Callable[[torch.Tensor], torch.Tensor]]
            It implements the derivative of the potential function :math:`d_U(r^2)`.
            If not provided, it will be computed using autograd. Default is None.
        d2_U : Optional[Callable[[torch.Tensor], torch.Tensor]]
            It implements the second derivative of the potential function :math:`d2_U(r^2)`.
            If not provided, it will be computed using autograd. Default is None.
        L : Optional[Callable[[torch.Tensor, float], Tuple[torch.Tensor, torch.Tensor]]]
            It implements a custom function to compute the spectrum of the Fokker-Planck operator.
            If not provided, it will be computed using autograd. Default is None.
        """
        self.__potential = potential
        self.__rotational_matrix = rotational_matrix
        self.__diffusion = sigma**2
        if L is None:
            _d_U: Callable[[torch.Tensor], torch.Tensor]
            _d2_U: Callable[[torch.Tensor], torch.Tensor]
            if d2_U is None and d_U is None:

                def get_spectrum(r2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                    return (
                        Gradient(1, potential, True)(r2),
                        Gradient(2, potential, True)(r2),
                    )

            elif d2_U is not None and d_U is None:
                _d2_U = cast(Callable[[torch.Tensor], torch.Tensor], d2_U)

                def get_spectrum(r2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                    return Gradient(1, potential, True)(r2), _d2_U(r2)

            elif d2_U is None and d_U is not None:
                _d_U = cast(Callable[[torch.Tensor], torch.Tensor], d_U)

                def get_spectrum(r2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                    return _d_U(r2), Gradient(2, potential, True)(r2)

            else:
                _d_U = cast(Callable[[torch.Tensor], torch.Tensor], d_U)
                _d2_U = cast(Callable[[torch.Tensor], torch.Tensor], d2_U)

                def get_spectrum(r2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                    return _d_U(r2), _d2_U(r2)

            usefull_1 = lambda d_U, d2_U, x: (
                4 * d2_U * torch.as_tensor(torch.norm(x, dim=1) ** 2, device=x.device)
                + 2 * d_U * x.shape[1],
                2 * d_U[:, None] * x,
            )
            L = lambda x, _t: usefull_1(*get_spectrum(torch.norm(x, dim=1) ** 2), x)

        if d_U is None:
            d_U = lambda r2: Gradient(1, potential, True)(r2)

        def drift_fn(x: torch.Tensor, _t: float) -> torch.Tensor:
            return -2.0 * d_U(torch.norm(x, dim=1) ** 2)[:, None] * x + torch.einsum(
                "ij,sj->si", self.__rotational_matrix, x
            )

        def diffusion_fn(x: torch.Tensor, _t: float) -> torch.Tensor:
            return self.__diffusion * torch.tile(
                torch.eye(x.shape[1]), (x.shape[0], 1, 1)
            )

        def u_t(
            u: Callable[[torch.Tensor, float], torch.Tensor], x: torch.Tensor, t: float
        ) -> torch.Tensor:
            L0, L1 = L(x, t)
            _u, d_u, d2_u = (
                Gradient(0, lambda y: u(y, t), True)(x),
                Gradient(1, lambda y: u(y, t), True)(x),
                Gradient(2, lambda y: u(y, t), True)(x),
            )
            return (
                L0 * _u
                + torch.einsum("si,si->s", L1, d_u)
                + 0.5 * self.__diffusion * torch.einsum("sii -> s", d2_u)
            )

        self.FPModel = Model(
            drift=drift_fn,
            diffusion=diffusion_fn,
            u_t=u_t,
        )

    def u_t(
        self,
        u: Callable[[torch.Tensor, float], torch.Tensor],
        x: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        """This method returns :math:`u_t = \\mathcal{L}u` where :math:`\\mathcal{L}` is the Fokker-Planck operator associated with this model.

        Parameters
        ----------
            u : Callable[[torch.Tensor, float], torch.Tensor]
                A function that takes a tensor with shape (N, d) and a float (time) and returns a tensor with shape (N,).
            x : torch.Tensor
                A tensor with shape (N, d).
            t : float
                A float representing the time.

        Returns
        -------
            torch.Tensor:
                A tensor with shape (N,).
        """
        return self.FPModel.u_t(u, x, t)

    def steady_state(self, x: torch.Tensor):
        """This method returns the steady-state distribution evaluated at x.

        Parameters
        ----------
            x : torch.Tensor
                A tensor with shape (N, d).

        Returns
        -------
            torch.Tensor:
                A tensor with shape (N,).
        """
        r2 = torch.norm(x, dim=1) ** 2
        return torch.exp(-2.0 / self.__diffusion * self.__potential(r2))
