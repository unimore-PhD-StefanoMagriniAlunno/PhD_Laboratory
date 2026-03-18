from phd_laboratory.differential import Divergence
import torch
from typing import Callable


class Model:

    def __init__(
        self,
        drift: Callable[[torch.Tensor, float], torch.Tensor],
        diffusion: Callable[[torch.Tensor, float], torch.Tensor],
        **kwargs,
    ) -> None:
        """This class implements a Fokker Planck model.

        In particular, we have: :math:`u_t = \\sum_i \\left(\\text{drift}\\left(x, t\\right)_i u\\right)_{x_i} + \\sum_{ij} \\left(\\text{diffusion}\\left(x, t\\right)_{ij}u\\right)_{x_i x_j}`

        where :math:`u_t` is the partial derivative of the density with respect to time, :math:`\\text{drift}:\\mathbb{R}^d\\times \\mathbb{R} \\to \\mathbb{R}^d` and :math:`\\text{diffusion}:\\mathbb{R}^d\\times \\mathbb{R} \\to \\mathbb{R}^{d\\times d}`.

        This class provides a batch implementation of the drift and diffusion functions.

        Parameters
        ----------
        drift : Callable[[torch.Tensor, float], torch.Tensor]
            It implements the function :math:`\\text{drift}\\left(x, t\\right)` in the Fokker-Planck equation.
            This function takes as input a tensor of shape (N, d) and a float (time) and returns a tensor of shape (N, d).
        diffusion : Callable[[torch.Tensor, float], torch.Tensor]
            It implements the function :math:`\\text{diffusion}\\left(x, t\\right)` in the Fokker-Planck equation.
            This function takes as input a tensor of shape (N, d) and a float (time) and returns a tensor of shape (N, d, d).
        **kwargs : additional arguments
            Additional arguments to encrease the performance of this class.
            For a simpler notation, the type of the Fokker-Planck solution :math:`u_t`, defined as `Callable[[Callable[[torch.Tensor, float], torch.Tensor], torch.Tensor, float], torch.Tensor]`,is renamed as `T_u`
            In particular, the user can provide custom implementations of the following functions:
            - **div_drift** : `Callable[[T_u, torch.Tensor, float], torch.Tensor]`
            to compute the divergence of the drift term, it takes as input a solution, a tensor of shape (N, d) and a float (time) and returns a tensor of shape (N,).
            - **div2_diffusion** : `Callable[[T_u, torch.Tensor, float], torch.Tensor]`
            to compute the divergence of the diffusion term, it takes as input a solution, a tensor of shape (N, d) and a float (time) and returns a tensor of shape (N,).
            - **u_t** : `Callable[[T_u, torch.Tensor, float], torch.Tensor]`
            to compute the partial derivative of the density with respect to time, it takes as input a solution, a tensor of shape (N, d) and a float (time) and returns a tensor of shape (N,).
            If u_t is defined, it overrides the other two functions. Otherwise, if div_drift and/or div2_diffusion are defined, they override the default implementations.

        Raises
        ------
            TypeError: `div_drift` must be a callable function
            TypeError: `div2_diffusion` must be a callable function
            TypeError: `u_t` must be a callable function
        """
        self.__drift = drift
        self.__diffusion = diffusion
        # Set other attributes from kwargs
        # if defined u_t, set it, else use default implementation
        if "u_t" in kwargs:  # if user provided u_t function
            # chek type
            if not callable(kwargs["u_t"]):
                raise TypeError("u_t must be a callable function")
            self.__u_t = kwargs["u_t"]  # override u_t if provided
        else:
            if "div_drift" in kwargs:  # if user provided divergence of drift
                # chek type
                if not callable(kwargs["u_t"]):
                    raise TypeError("u_t must be a callable function")
                self.__div_drift = kwargs["div_drift"]  # override div_drift if provided
            if "div2_diffusion" in kwargs:  # if user provided divergence of diffusion
                # chek type
                if not callable(kwargs["u_t"]):
                    raise TypeError("u_t must be a callable function")
                self.__div2_diffusion = kwargs[
                    "div2_diffusion"
                ]  # override div2_diffusion if provided

    @property
    def drift(self) -> Callable[[torch.Tensor, float], torch.Tensor]:
        return self.__drift

    @property
    def diffusion(self) -> Callable[[torch.Tensor, float], torch.Tensor]:
        return self.__diffusion

    def div_drift(
        self,
        u: Callable[[torch.Tensor, float], torch.Tensor],
        x: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        """This method returns the divergence of the drift term. Methematically it returns :math:`\\nabla_x \\cdot (fu)`

        Parameters
        ----------
        u : Callable[[torch.Tensor, float], torch.Tensor]
            This function takes a tensor with shape (N, d) and a float (time) and returns a tensor with shape (N,).
        x : torch.Tensor
            A tensor with shape (N, d).
        t : float
            A float representing the time.

        Returns
        -------
            torch.Tensor:
                A tensor with shape (N,).
        """
        if hasattr(self, "_Model__div_drift"):
            return self.__div_drift(u, x, t)
        return Divergence(
            1, lambda x: self.drift(x, t) * u(x, t), torch.Size([1]), True
        )(x)

    def div2_diffusion(
        self,
        u: Callable[[torch.Tensor, float], torch.Tensor],
        x: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        """This method returns the double divergence of the diffusion term. Methematically it return :math:`\\nabla_x^2 \\cdot (\\Sigma u)`

        Parameters
        ----------
        u : Callable[[torch.Tensor, float], torch.Tensor]
            This function takes a tensor with shape (N, d) and a float (time) and returns a tensor with shape (N,).
        x : torch.Tensor
            A tensor with shape (N, d).
        t : float
            A float representing the time.

        Returns
        -------
            torch.Tensor:
                A tensor with shape (N,).
        """
        if hasattr(self, "_Model__div2_diffusion"):
            return self.__div2_diffusion(u, x, t)
        return Divergence(
            2, lambda x: self.diffusion(x, t) * u(x, t), torch.Size([1]), True
        )(x)

    def u_t(
        self,
        u: Callable[[torch.Tensor, float], torch.Tensor],
        x: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        """This method returns the derivative :math:`u_t = -\\sum_i\\left(f_iu\\right)_{x_i} + \\frac{1}{2}\\sum_{ij}(\\Sigma_{ij} u)_{x_ix_j}`

        Parameters
        ----------
        u : Callable[[torch.Tensor, float], torch.Tensor]
            This function takes a tensor with shape (N, d) and a float (time) and returns a tensor with shape (N,).
        x : torch.Tensor
            A tensor with shape (N, d).
        t : float
            A float representing the time.

        Returns
        -------
            torch.Tensor:
                A tensor with shape (N,).
        """
        if hasattr(self, "_Model__u_t"):
            return self.__u_t(u, x, t)
        return -self.div_drift(u, x, t) + 0.5 * self.div2_diffusion(u, x, t)
