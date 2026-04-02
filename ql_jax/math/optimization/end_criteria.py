"""End criteria for optimization: convergence tests."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EndCriteria:
    """Convergence criteria for optimization algorithms.

    Attributes:
        max_iterations: Maximum number of iterations.
        max_stationary_state_iterations: Max iterations with no improvement.
        root_epsilon: Absolute tolerance on function value.
        function_epsilon: Relative tolerance on function value change.
        gradient_epsilon: Tolerance on gradient norm.
    """
    max_iterations: int = 1000
    max_stationary_state_iterations: int = 100
    root_epsilon: float = 1e-8
    function_epsilon: float = 1e-8
    gradient_epsilon: float = 1e-8

    def check_max_iterations(self, iteration: int) -> bool:
        return iteration >= self.max_iterations

    def check_stationary_point(self, f_old: float, f_new: float) -> bool:
        return abs(f_new - f_old) < self.function_epsilon

    def check_root(self, f_value: float) -> bool:
        return abs(f_value) < self.root_epsilon

    def check_gradient(self, grad_norm: float) -> bool:
        return grad_norm < self.gradient_epsilon
