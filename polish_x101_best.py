from pathlib import Path

from cvrp_utils import (
    read_vrplib_cvrp,
    read_reference_cost,
    build_distance_matrix,
    total_cost,
    is_solution_feasible,
    write_solution_file,
    plot_solution,
)

from cvrp_local_search import (
    local_search_improvement,
    two_route_repair_local_search,
    exact_route_polish_solution,
    group_exchange_local_search,
    two_opt_star_local_search,
)

def gap_percent(cost, reference_cost):
    return ((cost - reference_cost) / reference_cost) * 100


def read_saved_solution(path):
    """
    Reads assignment-style solution file:
    first line: cost time
    next lines: routes
    """
    routes = []

    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in lines[1:]:
        clean = line.strip()

        if not clean:
            continue

        route = list(map(int, clean.split()))
        routes.append(route)

    return routes


def main():
    vrp_path = Path("data") / "X-n101-k25.vrp"
    solution_path = Path("outputs") / "X-n101-k25_alns_tuned_best.txt"

    instance = read_vrplib_cvrp(str(vrp_path))
    dist = build_distance_matrix(instance)
    reference_cost = read_reference_cost(str(vrp_path.with_suffix(".sol")))

    solution = read_saved_solution(solution_path)

    feasible, errors = is_solution_feasible(solution, instance)

    if not feasible:
        raise ValueError(f"Loaded solution infeasible: {errors}")

    current_solution = solution
    current_cost = total_cost(current_solution, dist)

    print("Loaded solution:")
    print(f"cost={current_cost:.2f}, gap={gap_percent(current_cost, reference_cost):.2f}%")

    configs = [
        {"passes": 15, "max_combined": 16},
        {"passes": 20, "max_combined": 16},
        {"passes": 12, "max_combined": 18},
        {"passes": 15, "max_combined": 18},
        {"passes": 8, "max_combined": 20},
    ]

    best_solution = current_solution
    best_cost = current_cost
    best_config = None

    for config in configs:
        print(
            f"\nTrying final polish: passes={config['passes']}, "
            f"max_combined={config['max_combined']}"
        )

        candidate = local_search_improvement(
            best_solution,
            instance,
            dist,
            max_passes=20,
        )

        candidate = two_route_repair_local_search(
            candidate,
            instance,
            dist,
            max_passes=config["passes"],
            max_combined_customers=config["max_combined"],
        )

        candidate = two_opt_star_local_search(
            candidate,
            instance,
            dist,
            max_passes=20,
        )

        candidate = group_exchange_local_search(
            candidate,
            instance,
            dist,
            max_passes=10,
            max_group_size=2,
            max_route_customers=10,
        )

        candidate = local_search_improvement(
            candidate,
            instance,
            dist,
            max_passes=20,
        )

        candidate = exact_route_polish_solution(
            candidate,
            dist,
            max_customers=8,
        )

        feasible, errors = is_solution_feasible(candidate, instance)

        if not feasible:
            print("Infeasible polish result:", errors)
            continue

        candidate_cost = total_cost(candidate, dist)

        print(
            f"result cost={candidate_cost:.2f}, "
            f"gap={gap_percent(candidate_cost, reference_cost):.2f}%"
        )

        if candidate_cost < best_cost:
            best_solution = candidate
            best_cost = candidate_cost
            best_config = config
            print("NEW BEST!")

    print("\n" + "=" * 80)
    print("BEST POLISHED RESULT")
    print("Cost:", best_cost)
    print("Reference:", reference_cost)
    print("Gap:", f"{gap_percent(best_cost, reference_cost):.2f}%")
    print("Best config:", best_config)

    output_folder = Path("outputs")
    plot_folder = Path("plots")

    output_folder.mkdir(exist_ok=True)
    plot_folder.mkdir(exist_ok=True)

    write_solution_file(
        best_solution,
        dist,
        str(output_folder / "X-n101-k25_alns_polished_best.txt"),
        elapsed_time=0.0,
    )

    plot_solution(
        instance,
        best_solution,
        dist,
        save_path=str(plot_folder / "X-n101-k25_alns_polished_best.png"),
        show=False,
    )

    print("Saved polished output and plot.")


if __name__ == "__main__":
    main()