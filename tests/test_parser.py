from pathlib import Path

from cvrp_utils import (
    read_vrplib_cvrp,
    build_distance_matrix,
    greedy_initial_solution,
    total_cost,
    is_solution_feasible,
)


def test_instance(path: Path):
    print("=" * 70)
    print("File:", path.name)

    instance = read_vrplib_cvrp(str(path))
    dist = build_distance_matrix(instance)

    print("Instance name:", instance.name)
    print("Number of nodes including depot:", len(instance.coordinates))
    print("Number of customers:", len(instance.coordinates) - 1)
    print("Vehicle count:", instance.vehicle_count)
    print("Capacity:", instance.capacity)

    total_demand = sum(instance.demands.values())
    min_required_vehicles = (total_demand + instance.capacity - 1) // instance.capacity

    print("Total demand:", total_demand)
    print("Minimum vehicles by capacity only:", min_required_vehicles)

    solution = greedy_initial_solution(instance)
    cost = total_cost(solution, dist)

    feasible, errors = is_solution_feasible(solution, instance)

    print("Greedy cost:", round(cost, 2))
    print("Feasible?", feasible)
    print("Number of routes:", len(solution))

    used_routes = [route for route in solution if route != [0, 0]]
    print("Used vehicles:", len(used_routes))

    print("First routes preview:")
    for route in solution[:5]:
        print(route)

    if errors:
        print("Errors:")
        for error in errors:
            print("-", error)


def main():
    data_folder = Path("data")

    vrp_files = sorted(data_folder.glob("*.vrp"))

    if not vrp_files:
        print("No .vrp files found in data/ folder.")
        return

    for path in vrp_files:
        try:
            test_instance(path)
        except Exception as e:
            print("=" * 70)
            print("File:", path.name)
            print("FAILED:", e)


if __name__ == "__main__":
    main()