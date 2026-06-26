import random
from pathlib import Path

from cvrp_utils import (
    read_vrplib_cvrp,
    build_distance_matrix,
    greedy_initial_solution,
    total_cost,
    is_solution_feasible,
)

from cvrp_operators import (
    relocate_customer,
    swap_customers,
    two_opt_route,
    random_neighbor,
)


def test_operator(instance_path: Path, operator_name: str, operator_func):
    rng = random.Random(42)

    instance = read_vrplib_cvrp(str(instance_path))
    dist = build_distance_matrix(instance)

    solution = greedy_initial_solution(instance)
    initial_cost = total_cost(solution, dist)

    feasible, errors = is_solution_feasible(solution, instance)

    if not feasible:
        print(f"{operator_name}: initial solution infeasible")
        print(errors)
        return

    successful_moves = 0
    best_cost = initial_cost

    current_solution = solution

    for _ in range(100):
        result = operator_func(current_solution, instance, rng)

        if result is None:
            continue

        neighbor, move = result

        feasible, errors = is_solution_feasible(neighbor, instance)

        if not feasible:
            print(f"{operator_name}: produced infeasible solution!")
            print("Move:", move)
            print(errors)
            return

        successful_moves += 1

        neighbor_cost = total_cost(neighbor, dist)

        if neighbor_cost < best_cost:
            best_cost = neighbor_cost

        # For testing, continue from the new neighbor.
        current_solution = neighbor

    print(f"Operator: {operator_name}")
    print(f"Initial cost: {initial_cost:.2f}")
    print(f"Best cost after test moves: {best_cost:.2f}")
    print(f"Successful moves: {successful_moves}/100")
    print()


def main():
    data_folder = Path("data")

    # Start with a small instance first.
    instance_path = data_folder / "P-n16-k8.vrp"

    if not instance_path.exists():
        print("P-n16-k8.vrp not found in data folder.")
        return

    print("Testing operators on:", instance_path.name)
    print("=" * 70)

    test_operator(instance_path, "relocate", relocate_customer)
    test_operator(instance_path, "swap", swap_customers)
    test_operator(instance_path, "two_opt", two_opt_route)
    test_operator(instance_path, "random_neighbor", random_neighbor)


if __name__ == "__main__":
    main()