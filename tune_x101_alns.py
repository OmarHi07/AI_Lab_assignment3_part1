from pathlib import Path

from cvrp_utils import (
    read_vrplib_cvrp,
    read_reference_cost,
    build_distance_matrix,
    initial_solution_candidates,
    total_cost,
    is_solution_feasible,
    write_solution_file,
    plot_solution,
)

from cvrp_capacity_starts import randomized_capacity_dp_candidates

from cvrp_local_search import local_search_improvement
from cvrp_alns import alns


def gap_percent(cost, reference_cost):
    return ((cost - reference_cost) / reference_cost) * 100


def main():
    vrp_path = Path("data") / "X-n101-k25.vrp"

    instance = read_vrplib_cvrp(str(vrp_path))
    dist = build_distance_matrix(instance)
    reference_cost = read_reference_cost(str(vrp_path.with_suffix(".sol")))

    raw_candidates = initial_solution_candidates(instance)

    # Add many capacity-feasible variants.
    # This is important for X-n101-k25 because only capacity-DP works,
    # but one deterministic capacity-DP start is too limiting.
    raw_candidates.extend(randomized_capacity_dp_candidates(instance))

    # X-n101-k25 usually only has capacity_dp as feasible.
    candidates = []

    for name, solution, cost in raw_candidates:
        polished_solution = local_search_improvement(
            solution,
            instance,
            dist,
            max_passes=10,
        )

        polished_cost = total_cost(polished_solution, dist)

        feasible, errors = is_solution_feasible(polished_solution, instance)

        if feasible:
            candidates.append((name, polished_solution, polished_cost))
        else:
            print("Candidate infeasible:", name, errors)

    candidates.sort(key=lambda item: item[2])

    # Keep only the best few starting solutions to avoid huge runtime.
    candidates = candidates[:5]

    print("Initial candidates:")
    for name, solution, cost in candidates:
        print(f"{name:<15} cost={cost:.2f}, gap={gap_percent(cost, reference_cost):.2f}%")

    configs = [
        {
            "name": "medium_destroy",
            "iterations": 20000,
            "temp_factor": 0.10,
            "cooling": 0.9996,
            "q_min_ratio": 0.03,
            "q_max_ratio": 0.30,
        },
        {
            "name": "medium_hotter",
            "iterations": 25000,
            "temp_factor": 0.15,
            "cooling": 0.9997,
            "q_min_ratio": 0.03,
            "q_max_ratio": 0.35,
        },
    ]

    seeds = [42, 99]

    best_solution = None
    best_cost = float("inf")
    best_info = None

    customer_count = len(instance.coordinates) - 1

    for start_name, start_solution, start_cost in candidates:
        for config in configs:
            for seed in seeds:
                q_min = max(2, int(config["q_min_ratio"] * customer_count))
                q_max = max(q_min, int(config["q_max_ratio"] * customer_count))

                print(
                    f"\nRunning config={config['name']}, seed={seed}, "
                    f"q_min={q_min}, q_max={q_max}, start={start_name}"
                )

                result = alns(
                    instance=instance,
                    initial_solution=start_solution,
                    max_iterations=config["iterations"],
                    seed=seed,
                    initial_temperature=config["temp_factor"] * start_cost,
                    cooling_rate=config["cooling"],
                    reaction_factor=0.10,
                    q_min=q_min,
                    q_max=q_max,
                )

                polished_solution = local_search_improvement(
                    result.best_solution,
                    instance,
                    dist,
                    max_passes=15,
                )

                polished_cost = total_cost(polished_solution, dist)

                feasible, errors = is_solution_feasible(polished_solution, instance)

                if not feasible:
                    print("Infeasible result!", errors)
                    continue

                gap = gap_percent(polished_cost, reference_cost)

                print(
                    f"Result cost={polished_cost:.2f}, "
                    f"gap={gap:.2f}%, "
                    f"time={result.elapsed_time:.2f}s"
                )

                if polished_cost < best_cost:
                    best_cost = polished_cost
                    best_solution = polished_solution
                    best_info = {
                        "config": config["name"],
                        "seed": seed,
                        "start": start_name,
                        "time": result.elapsed_time,
                        "gap": gap,
                    }

                    print("NEW BEST!")

    print("\n" + "=" * 80)
    print("BEST X-n101-k25 RESULT")
    print("Cost:", best_cost)
    print("Reference:", reference_cost)
    print("Gap:", f"{best_info['gap']:.2f}%")
    print("Config:", best_info)

    output_folder = Path("outputs")
    plot_folder = Path("plots")

    output_folder.mkdir(exist_ok=True)
    plot_folder.mkdir(exist_ok=True)

    write_solution_file(
        best_solution,
        dist,
        str(output_folder / "X-n101-k25_alns_tuned_best.txt"),
        elapsed_time=best_info["time"],
    )

    plot_solution(
        instance,
        best_solution,
        dist,
        save_path=str(plot_folder / "X-n101-k25_alns_tuned_best.png"),
        show=False,
    )

    print("Saved best tuned output and plot.")


if __name__ == "__main__":
    main()