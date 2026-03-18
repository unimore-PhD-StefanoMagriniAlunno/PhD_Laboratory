from phd_laboratory.sde.model import Model
import torch


class EulerMaruyama:
    """Implements the Euler-Maruyama scheme for simulating stochastic differential equations (SDEs).

    SDE Model:
        :math:`dX_t = dt\\text{drift}(X_t, t) + \\text{diffusion}(X_t, t) dW_t \\quad X_t\\in\\mathbb{R}^{d}`

        where :math:`W_t` is a standard Wiener process in :math:`\\mathbb{R}^{d}`, :math:`\\text{drift}\\in\\mathbb{R}^{d}`, :math:`\\text{diffusion}\\in\\mathbb{R}^{d\\times d}`.

    Euler-Maruyama Update:
        :math:`X_{t+dt} = X_t + dt\\text{drift}(X_t, t) + \\text{diffusion}(X_t, t) \\Delta W_t`

        where :math:`\\Delta W_t \\sim \\mathcal{N}(0, dt I_d)`.
    """

    def __init__(
        self,
        model: Model,
        dt_schedule: torch.Tensor,
    ) -> None:
        """Initialize the Euler-Maruyama scheme with a given SDE model and time step schedule.

        Parameters
        ----------
            model : Model
                An instance of the SDE model defining the drift and diffusion terms.
            dt_schedule : torch.Tensor
                A 1D array of positive floats representing the time step sizes for each iteration.

        Raises
        ------
            ValueError: If any element in dt_schedule is non-positive or if dt_schedule is not a 1D array.

        Examples
        --------

            >>> model = Model(...)
            >>> dt_schedule = torch.array([0.01, 0.01, 0.01])
            >>> euler_maruyama = EulerMaruyama(model, dt_schedule)
        """

        if torch.any(dt_schedule <= 0):
            raise ValueError("All elements in dt_schedule must be positive floats.")
        if dt_schedule.ndim != 1:
            raise ValueError("dt_schedule must be a 1D array.")

        self.model = model
        self.dt_schedule = dt_schedule

    @property
    def n_steps(self) -> int:
        """Number of steps in the dt_schedule.

        Returns
        -------
            int
                Number of steps.
        """
        return self.dt_schedule.shape[0]

    def get_trajectory(
        self: "EulerMaruyama",
        n_samples: int,
    ) -> torch.Tensor:
        """Generate a trajectory using the Euler-Maruyama scheme.

        Parameters
        ----------
            n_samples : int | None
                Number of samples to generate.

        Raises
        ------
            RuntimeError: there is an error computing `drift` or `diffusion` terms.
            ValueError: `drift` or `diffusion` terms have incorrect shapes.

        Returns
        -------
            torch.Tensor
                Trajectory of states. Shape: (n_steps + 1, n_samples, n_dim)

        Examples
        --------

            >>> euler_maruyama = EulerMaruyama(model, dt_schedule)
            >>> trajectory = euler_maruyama.get_trajectory(n_samples=100)
        """
        X0 = self.model.sample_initial_state(n_samples)
        Xtraj = torch.zeros(
            (self.n_steps + 1, n_samples, self.model.n_dim),
            dtype=X0.dtype,
            device=self.dt_schedule.device,
        )
        Xtraj[0] = X0
        del X0

        t_current = self.model.initial_time
        for k in range(1, self.n_steps + 1):
            dt = self.dt_schedule[k - 1]
            dW = torch.randn(
                n_samples,
                self.model.n_dim,
                dtype=Xtraj.dtype,
                device=self.dt_schedule.device,
            ) * torch.sqrt(dt)
            try:
                drift_term = self.model.drift(Xtraj[k - 1], t_current)
                diffusion_term = self.model.diffusion(Xtraj[k - 1], t_current)
            except Exception as e:
                raise RuntimeError(f"Error computing drift or diffusion terms: {e}")
            else:
                if drift_term.shape != (n_samples, self.model.n_dim):
                    raise ValueError("Drift term has incorrect shape.")
                if diffusion_term.shape != (
                    n_samples,
                    self.model.n_dim,
                    self.model.n_dim,
                ):
                    raise ValueError("Diffusion term has incorrect shape.")
                Xtraj[k] = (
                    Xtraj[k - 1]
                    + drift_term * dt
                    + torch.einsum("sij,sj->sj", diffusion_term, dW)
                )
                t_current += dt
        return Xtraj

    def get_end(
        self: "EulerMaruyama",
        n_samples: int,
    ) -> torch.Tensor:
        """Generate samples at the final time using the Euler-Maruyama scheme.

        Parameters
        ----------
            n_samples : int
                Number of samples to generate.

        Raises
        ------
            RuntimeError: there is an error computing `drift` or `diffusion` terms.
            ValueError: `drift` or `diffusion` terms have incorrect shapes.

        Returns
        -------
            torch.Tensor
                Updated state after applying all time steps in `dt_schedule`. Shape: (n_samples, n_dim)

        Examples
        --------

            >>> euler_maruyama = EulerMaruyama(model, dt_schedule)
            >>> end_states = euler_maruyama.get_end(n_samples=100)
        """
        X_current = self.model.sample_initial_state(n_samples)
        t_current = self.model.initial_time
        for k in range(self.n_steps):
            dt = self.dt_schedule[k]
            dW = torch.randn(
                n_samples,
                self.model.n_dim,
                dtype=X_current.dtype,
                device=self.dt_schedule.device,
            ) * torch.sqrt(dt)
            try:
                drift_term = self.model.drift(X_current, t_current)
                diffusion_term = self.model.diffusion(X_current, t_current)
            except Exception as e:
                raise RuntimeError(f"Error computing drift or diffusion terms: {e}")
            else:
                if drift_term.shape != (n_samples, self.model.n_dim):
                    raise ValueError("Drift term has incorrect shape.")
                if diffusion_term.shape != (
                    n_samples,
                    self.model.n_dim,
                    self.model.n_dim,
                ):
                    raise ValueError("Diffusion term has incorrect shape.")
                X_current = (
                    X_current
                    + drift_term * dt
                    + torch.einsum("sij,sj->sj", diffusion_term, dW)
                )
                t_current += dt
        return X_current

    @property
    def device(self) -> torch.device:
        """Device on which the dt_schedule tensor is stored.

        Returns
        -------
            torch.device
                Device of the dt_schedule tensor.
        """
        return self.dt_schedule.device
