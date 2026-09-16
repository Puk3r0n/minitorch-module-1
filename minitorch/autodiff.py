from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # Symmetric difference: nudge only the arg we differentiate with respect to.
    # It is more accurate than the one-sided version, the error is O(epsilon^2).
    vals_plus = list(vals)
    vals_minus = list(vals)
    vals_plus[arg] = vals_plus[arg] + epsilon
    vals_minus[arg] = vals_minus[arg] - epsilon
    return (f(*vals_plus) - f(*vals_minus)) / (2.0 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    order: List[Variable] = []
    visited = set()

    def visit(var: Variable) -> None:
        if var.unique_id in visited or var.is_constant():
            return
        if not var.is_leaf():
            for parent in var.parents:
                visit(parent)
        visited.add(var.unique_id)
        # Parents are appended before the node itself, so prepending here
        # leaves the right-most variable first.
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # Walking in topological order guarantees a node is only processed once
    # every path leading into it has already contributed its derivative.
    derivatives = {variable.unique_id: deriv}

    for var in topological_sort(variable):
        d_output = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(d_output)
            continue
        for parent, d_parent in var.chain_rule(d_output):
            if parent.is_constant():
                continue
            derivatives.setdefault(parent.unique_id, 0.0)
            derivatives[parent.unique_id] += d_parent


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
