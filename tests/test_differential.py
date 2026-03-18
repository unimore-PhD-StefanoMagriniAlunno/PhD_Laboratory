import pytest

__all__ = ["pytest"]

"""
from phd_laboratory.differential import Differential, div_at, nabla_at


def test_nabla_at():
    import torch

    # N,2,4 -> N,2,3
    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [  # x2
                torch.stack(
                    [
                        x[0, 0] + x[0, 1],
                        x[1, 0] * x[1, 1],
                        x[0, 2] ** 2 + x[1, 2] ** 2,
                    ],
                    dim=-1,
                ),  # x3
                torch.stack(
                    [
                        x[0, 2] - x[0, 3],
                        x[1, 2] * x[1, 3],
                        x[0, 3] ** 2 - x[1, 3] ** 2,
                    ],
                    dim=-1,
                ),  # x3
            ],
            dim=-2,
        )

    # N,2,4 -> N,2,3,2,4
    def expJ(x: torch.Tensor) -> torch.Tensor:
        # x has shape (N,2,4)
        # returns a tensor with shape (N,2,3,2,4)
        N = x.shape[0]
        return torch.stack(
            [  # x2
                torch.stack(
                    [  # x3
                        torch.stack(
                            [  # x2
                                torch.stack(
                                    [
                                        torch.ones(N),
                                        torch.ones(N),
                                        torch.zeros(N),
                                        torch.zeros(N),
                                    ],
                                    dim=-1,
                                ),  # x4  dx[0,:]
                                torch.stack(
                                    [
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        torch.zeros(N),
                                    ],
                                    dim=-1,
                                ),  # x4  dx[1,:]
                            ],
                            dim=-2,
                        ),
                        torch.stack(
                            [  # x2
                                torch.stack(
                                    [
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        torch.zeros(N),
                                    ],
                                    dim=-1,
                                ),  # x4
                                torch.stack(
                                    [
                                        x[:, 1, 1],
                                        x[:, 1, 0],
                                        torch.zeros(N),
                                        torch.zeros(N),
                                    ],
                                    dim=-1,
                                ),  # x4
                            ],
                            dim=-2,
                        ),
                        torch.stack(
                            [  # x2
                                torch.stack(
                                    [
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        2 * x[:, 0, 2],
                                        torch.zeros(N),
                                    ],
                                    dim=-1,
                                ),  # x4
                                torch.stack(
                                    [
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        2 * x[:, 1, 2],
                                        torch.zeros(N),
                                    ],
                                    dim=-1,
                                ),  # x4
                            ],
                            dim=-2,
                        ),
                    ],
                    dim=-3,
                ),
                torch.stack(
                    [  # x3
                        torch.stack(
                            [  # x2
                                torch.stack(
                                    [
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        torch.ones(N),
                                        -torch.ones(N),
                                    ],
                                    dim=-1,
                                ),  # x4
                                torch.stack(
                                    [
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        torch.zeros(N),
                                    ],
                                    dim=-1,
                                ),  # x4
                            ],
                            dim=-2,
                        ),
                        torch.stack(
                            [  # x2
                                torch.stack(
                                    [
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        torch.zeros(N),
                                    ],
                                    dim=-1,
                                ),  # x4
                                torch.stack(
                                    [
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        x[:, 1, 3],
                                        x[:, 1, 2],
                                    ],
                                    dim=-1,
                                ),  # x4
                            ],
                            dim=-2,
                        ),
                        torch.stack(
                            [  # x2
                                torch.stack(
                                    [
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        2 * x[:, 0, 3],
                                    ],
                                    dim=-1,
                                ),  # x4
                                torch.stack(
                                    [
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        -2 * x[:, 1, 3],
                                    ],
                                    dim=-1,
                                ),  # x4
                            ],
                            dim=-2,
                        ),
                    ],
                    dim=-3,
                ),
            ],
            dim=-4,
        )

    N = 10
    eps = 0.0001
    x = torch.rand(N, 2, 4) * 10 - 20
    expJ_x = expJ(x)

    J_x = nabla_at(f, x)
    expJ_x = expJ(x)
    assert J_x.shape == expJ_x.shape
    assert torch.max(J_x - expJ_x) < eps


def test_div_at():
    import torch

    # N,2,3 -> N,4,2,3
    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [  # x4
                torch.stack(
                    [  # x2
                        torch.stack(
                            [
                                x[0, 0] * x[0, 1],
                                x[0, 2] * x[1, 0],
                                x[1, 1] * x[1, 2],
                            ],
                            dim=-1,
                        ),  # x3
                        torch.stack(
                            [x[0, 0] ** 2, x[0, 1] ** 2, x[0, 2] ** 2], dim=-1
                        ),  # x3
                    ],
                    dim=-2,
                ),
                torch.stack(
                    [  # x2
                        torch.stack(
                            [
                                x[1, 0] - x[1, 1],
                                x[1, 2] - x[0, 0],
                                x[0, 1] - x[0, 2],
                            ],
                            dim=-1,
                        ),  # x3
                        torch.stack(
                            [
                                x[0, 2] + x[1, 0],
                                x[1, 1] + x[1, 2],
                                x[0, 0] + x[0, 1],
                            ],
                            dim=-1,
                        ),  # x3
                    ],
                    dim=-2,
                ),
                torch.stack(
                    [  # x2
                        torch.stack(
                            [
                                x[0, 2] ** 2 - x[1, 0],
                                x[1, 1] ** 2 - x[1, 2],
                                x[0, 0] ** 2 - x[0, 1],
                            ],
                            dim=-1,
                        ),  # x3
                        torch.stack(
                            [
                                x[0, 1] - x[0, 2] ** 2,
                                x[1, 0] - x[1, 1] ** 2,
                                x[1, 2] - x[0, 1] ** 2,
                            ],
                            dim=-1,
                        ),  # x3
                    ],
                    dim=-2,
                ),
                torch.stack(
                    [  # x2
                        torch.stack(
                            [
                                x[0, 1] ** 2 + x[0, 2],
                                x[1, 0] ** 2 + x[1, 1],
                                x[1, 2] ** 2 + x[0, 0],
                            ],
                            dim=-1,
                        ),  # x3
                        torch.stack(
                            [
                                x[0, 0] + x[0, 1] ** 2,
                                x[0, 2] + x[1, 0] ** 2,
                                x[1, 1] + x[1, 2] ** 2,
                            ],
                            dim=-1,
                        ),  # x3
                    ],
                    dim=-2,
                ),
            ],
            dim=-3,
        )

    # N,2,3 -> N,4
    def expDiv(x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        return torch.stack(
            [
                x[:, 0, 1],
                torch.ones(N),
                (-2.0) * x[:, 1, 1] + torch.ones(N),
                (2.0) * x[:, 1, 2],
            ],
            dim=-1,
        )

    N = 10
    eps = 0.0001
    x = torch.rand(N, 2, 3) * 10 - 20
    expDiv_x = expDiv(x)
    div_x = div_at(f, x)
    assert div_x.shape == expDiv_x.shape
    assert torch.max(div_x - expDiv_x) < eps


def test_differential():
    diff = Differential(2, create_graph=True)
    assert diff.order == 2
    assert diff.create_graph is True


def test_differentialNabla():
    import torch

    # N,2,4 -> N,2,3
    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [  # x2
                torch.stack(
                    [
                        x[0, 0] + x[0, 1],
                        x[1, 0] * x[1, 1],
                        x[0, 2] ** 2 + x[1, 2] ** 2,
                    ],
                    dim=-1,
                ),  # x3
                torch.stack(
                    [
                        x[0, 2] - x[0, 3],
                        x[1, 2] * x[1, 3],
                        x[0, 3] ** 2 - x[1, 3] ** 2,
                    ],
                    dim=-1,
                ),  # x3
            ],
            dim=-2,
        )

    # N,2,4 -> N,2,3,2,4
    def expJ(x: torch.Tensor) -> torch.Tensor:
        # x has shape (N,2,4)
        # returns a tensor with shape (N,2,3,2,4)
        N = x.shape[0]
        return torch.stack(
            [  # x2
                torch.stack(
                    [  # x3
                        torch.stack(
                            [  # x2
                                torch.stack(
                                    [
                                        torch.ones(N),
                                        torch.ones(N),
                                        torch.zeros(N),
                                        torch.zeros(N),
                                    ],
                                    dim=-1,
                                ),  # x4  dx[0,:]
                                torch.stack(
                                    [
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        torch.zeros(N),
                                    ],
                                    dim=-1,
                                ),  # x4  dx[1,:]
                            ],
                            dim=-2,
                        ),
                        torch.stack(
                            [  # x2
                                torch.stack(
                                    [
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        torch.zeros(N),
                                    ],
                                    dim=-1,
                                ),  # x4
                                torch.stack(
                                    [
                                        x[:, 1, 1],
                                        x[:, 1, 0],
                                        torch.zeros(N),
                                        torch.zeros(N),
                                    ],
                                    dim=-1,
                                ),  # x4
                            ],
                            dim=-2,
                        ),
                        torch.stack(
                            [  # x2
                                torch.stack(
                                    [
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        2 * x[:, 0, 2],
                                        torch.zeros(N),
                                    ],
                                    dim=-1,
                                ),  # x4
                                torch.stack(
                                    [
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        2 * x[:, 1, 2],
                                        torch.zeros(N),
                                    ],
                                    dim=-1,
                                ),  # x4
                            ],
                            dim=-2,
                        ),
                    ],
                    dim=-3,
                ),
                torch.stack(
                    [  # x3
                        torch.stack(
                            [  # x2
                                torch.stack(
                                    [
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        torch.ones(N),
                                        -torch.ones(N),
                                    ],
                                    dim=-1,
                                ),  # x4
                                torch.stack(
                                    [
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        torch.zeros(N),
                                    ],
                                    dim=-1,
                                ),  # x4
                            ],
                            dim=-2,
                        ),
                        torch.stack(
                            [  # x2
                                torch.stack(
                                    [
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        torch.zeros(N),
                                    ],
                                    dim=-1,
                                ),  # x4
                                torch.stack(
                                    [
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        x[:, 1, 3],
                                        x[:, 1, 2],
                                    ],
                                    dim=-1,
                                ),  # x4
                            ],
                            dim=-2,
                        ),
                        torch.stack(
                            [  # x2
                                torch.stack(
                                    [
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        2 * x[:, 0, 3],
                                    ],
                                    dim=-1,
                                ),  # x4
                                torch.stack(
                                    [
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        torch.zeros(N),
                                        -2 * x[:, 1, 3],
                                    ],
                                    dim=-1,
                                ),  # x4
                            ],
                            dim=-2,
                        ),
                    ],
                    dim=-3,
                ),
            ],
            dim=-4,
        )

    # N,2,4 -> N,2,3,2,4,2,4
    def expJ2(x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        ret = torch.zeros((N, 2, 3, 2, 4, 2, 4), dtype=torch.float64)
        ret[:, 0, 2, 0, 2, 0, 2] = 2.0 * torch.ones(N)
        ret[:, 1, 2, 0, 3, 0, 3] = 2.0 * torch.ones(N)
        ret[:, 0, 1, 1, 1, 1, 0] = 1.0 * torch.ones(N)
        ret[:, 0, 1, 1, 0, 1, 1] = 1.0 * torch.ones(N)
        ret[:, 0, 2, 1, 2, 1, 2] = 2.0 * torch.ones(N)
        ret[:, 1, 1, 1, 3, 1, 2] = 1.0 * torch.ones(N)
        ret[:, 1, 1, 1, 2, 1, 3] = 1.0 * torch.ones(N)
        ret[:, 1, 2, 1, 3, 1, 3] = -2.0 * torch.ones(N)
        return ret

    diff = Differential(2, create_graph=True)

    N = 10
    eps = 0.0001
    x = torch.rand(N, 2, 4) * 10 - 20
    expf_x = f(x)
    expJ1_x = expJ(x)
    expJ2_x = expJ2(x)

    f_x, J1_x, J2_x = diff.nabla(f)(x, True)

    assert f_x.shape == expf_x.shape
    assert torch.max(f_x - expf_x) < eps
    assert J1_x.shape == expJ1_x.shape
    assert torch.max(J1_x - expJ1_x) < eps
    assert expJ2_x.shape == J2_x.shape
    assert torch.max(J2_x - expJ2_x) < eps


def test_differentialDiv():
    import torch

    # N,2,3 -> N,2,3,2,3
    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [  # x2
                torch.stack(
                    [  # x3
                        torch.stack(
                            [  # x2
                                torch.stack(
                                    [
                                        x[:, 0, 0] * x[:, 0, 1],
                                        x[:, 0, 2] * x[:, 1, 0],
                                        x[:, 1, 1] * x[:, 1, 2],
                                    ],
                                    dim=-1,
                                ),  # x3
                                torch.stack(
                                    [x[:, 0, 0] ** 2, x[:, 0, 1] ** 2, x[:, 0, 2] ** 2],
                                    dim=-1,
                                ),  # x3
                            ],
                            dim=-2,
                        ),
                        torch.stack(
                            [  # x2
                                torch.stack(
                                    [
                                        x[:, 1, 0] - x[:, 1, 1],
                                        x[:, 1, 2] - x[:, 0, 0],
                                        x[:, 0, 1] - x[:, 0, 2],
                                    ],
                                    dim=-1,
                                ),  # x3
                                torch.stack(
                                    [
                                        x[:, 0, 2] + x[:, 1, 0],
                                        x[:, 1, 1] + x[:, 1, 2],
                                        x[:, 0, 0] + x[:, 0, 1],
                                    ],
                                    dim=-1,
                                ),  # x3
                            ],
                            dim=-2,
                        ),
                        torch.stack(
                            [  # x2
                                torch.stack(
                                    [
                                        x[:, 0, 2] ** 2 - x[:, 1, 0],
                                        x[:, 1, 1] ** 2 - x[:, 1, 2],
                                        x[:, 0, 0] ** 2 - x[:, 0, 1],
                                    ],
                                    dim=-1,
                                ),  # x3
                                torch.stack(
                                    [
                                        x[:, 0, 1] - x[:, 0, 2] ** 2,
                                        x[:, 1, 0] - x[:, 1, 1] ** 2,
                                        x[:, 1, 2] - x[:, 0, 1] ** 2,
                                    ],
                                    dim=-1,
                                ),  # x3
                            ],
                            dim=-2,
                        ),
                    ],
                    dim=-3,
                ),
                torch.stack(
                    [  # x3
                        torch.stack(
                            [  # x2
                                torch.stack(
                                    [x[:, 0, 1], x[:, 0, 2], x[:, 1, 0]], dim=-1
                                ),  # x3
                                torch.stack(
                                    [x[:, 1, 1] ** 2, x[:, 1, 2] ** 2, x[:, 0, 0] ** 2],
                                    dim=-1,
                                ),  # x3
                            ],
                            dim=-2,
                        ),
                        torch.stack(
                            [  # x2
                                torch.stack(
                                    [
                                        x[:, 0, 1] + x[:, 0, 1],
                                        x[:, 0, 2] + x[:, 1, 0],
                                        x[:, 1, 1] + x[:, 1, 2],
                                    ],
                                    dim=-1,
                                ),  # x3
                                torch.stack(
                                    [x[:, 0, 0] ** 2, x[:, 0, 1] ** 2, x[:, 0, 2] ** 2],
                                    dim=-1,
                                ),  # x3
                            ],
                            dim=-2,
                        ),
                        torch.stack(
                            [  # x2
                                torch.stack(
                                    [
                                        x[:, 0, 2] ** 2 - x[:, 1, 0],
                                        x[:, 1, 1] ** 2 - x[:, 1, 2],
                                        x[:, 0, 0] ** 2 - x[:, 0, 1],
                                    ],
                                    dim=-1,
                                ),  # x3
                                torch.stack(
                                    [
                                        x[:, 0, 1] + x[:, 0, 2] ** 2,
                                        x[:, 1, 0] + x[:, 1, 1] ** 2,
                                        x[:, 1, 2] + x[:, 0, 1] ** 2,
                                    ],
                                    dim=-1,
                                ),  # x3
                            ],
                            dim=-2,
                        ),
                    ],
                    dim=-3,
                ),
            ],
            dim=-4,
        )

    # N,2,3 -> N,2,3
    def expDiv(x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        return torch.stack(
            [
                torch.stack(
                    [x[:, 0, 1], torch.ones(N), (-2.0) * x[:, 1, 1] + torch.ones(N)],
                    dim=-1,
                ),
                torch.stack(
                    [
                        torch.zeros(N),
                        torch.zeros(N),
                        (2.0) * x[:, 1, 1] + torch.ones(N),
                    ],
                    dim=-1,
                ),
            ],
            dim=-2,
        )

    # N,2,3 -> N
    def expDiv2(x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        return torch.zeros(N)

    diff = Differential(2, create_graph=True)

    N = 10
    eps = 0.0001
    x = torch.rand(N, 2, 3) * 10 - 20
    expf_x = f(x)
    expDiv1_x = expDiv(x)
    expDiv2_x = expDiv2(x)

    f_x, Div1_x, Div2_x = diff.div(f)(x, True)

    assert f_x.shape == expf_x.shape
    assert torch.max(f_x - expf_x) < eps
    assert Div1_x.shape == expDiv1_x.shape
    assert torch.max(Div1_x - expDiv1_x) < eps
    assert expDiv2_x.shape == Div2_x.shape
    assert torch.max(Div2_x - expDiv2_x) < eps
"""
