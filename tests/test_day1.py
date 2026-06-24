from cvrp_utils import (
    create_assignment_example_instance,
    build_distance_matrix,
    route_cost,
    total_cost,
    route_demand,
    is_solution_feasible,
    greedy_initial_solution,
    print_solution,
)


def main():
    instance = create_assignment_example_instance()
    dist = build_distance_matrix(instance)

    print("Instance name:", instance.name)
    print("Vehicle count:", instance.vehicle_count)
    print("Capacity:", instance.capacity)
    print("Coordinates:", instance.coordinates)
    print("Demands:", instance.demands)

    print("\nTesting assignment example solution:")

    assignment_solution = [
        [0, 1, 2, 3, 0],
        [0, 4, 0],
        [0, 0],
        [0, 0],
    ]

    print("Route 0 cost:", route_cost(assignment_solution[0], dist))
    print("Route 1 cost:", route_cost(assignment_solution[1], dist))
    print("Total cost:", total_cost(assignment_solution, dist))

    feasible, errors = is_solution_feasible(assignment_solution, instance)
    print("Feasible?", feasible)

    if errors:
        print("Errors:")
        for error in errors:
            print("-", error)

    print("\nAssignment-style output:")
    print_solution(assignment_solution, dist)

    print("\nTesting greedy initial solution:")

    greedy_solution = greedy_initial_solution(instance)

    print("Greedy solution:")
    print(greedy_solution)

    print("Greedy total cost:", total_cost(greedy_solution, dist))

    feasible, errors = is_solution_feasible(greedy_solution, instance)
    print("Greedy feasible?", feasible)

    if errors:
        print("Errors:")
        for error in errors:
            print("-", error)

    print("\nGreedy assignment-style output:")
    print_solution(greedy_solution, dist)


if __name__ == "__main__":
    main()