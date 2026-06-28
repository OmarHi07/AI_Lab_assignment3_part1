import csv
import statistics
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

from cvrp_local_search import local_search_improvement

from cvrp_sa import simulated_annealing
from cvrp_tabu import tabu_search
from cvrp_alns import alns


def gap_percent(cost, reference_cost):
    if reference_cost is None:
        return None
    return ((cost - reference_cost) / reference_cost) * 100


def get_params(instance_name, customer_count):
    """
    Different instance sizes need different search budgets.
    """
    if instance_name == "X-n101-k25":
        return {
            "seeds": [1, 42, 99],
            "candidate_limit": 1,

            "sa_iterations": 60000,
            "sa_cooling": 0.9996,
            "sa_temp_factor": 0.12,

            "tabu_iterations": 2000,
            "tabu_sample": 80,
            "tabu_tenure": 30,

            # X-n101-k25 has very tight capacity.
            # Large destroy sizes hurt repair quality, so we use smaller destroy.
            "alns_iterations": 15000,
            "alns_cooling": 0.9996,
            "alns_q_min_ratio": 0.03,
            "alns_q_max_ratio": 0.30,
        }

    if customer_count <= 35:
        return {
            "seeds": [1, 7, 42, 99],
            "candidate_limit": 4,

            "sa_iterations": 30000,
            "sa_cooling": 0.9993,
            "sa_temp_factor": 0.10,

            "tabu_iterations": 1500,
            "tabu_sample": 100,
            "tabu_tenure": 20,

            "alns_iterations": 5000,
            "alns_cooling": 0.9993,
            "alns_q_min_ratio": 0.05,
            "alns_q_max_ratio": 0.30,
        }

    if customer_count <= 100:
        return {
            "seeds": [1, 42, 99],
            "candidate_limit": 3,

            "sa_iterations": 60000,
            "sa_cooling": 0.9996,
            "sa_temp_factor": 0.12,

            "tabu_iterations": 2500,
            "tabu_sample": 120,
            "tabu_tenure": 30,

            "alns_iterations": 20000,
            "alns_cooling": 0.9996,
            "alns_q_min_ratio": 0.08,
            "alns_q_max_ratio": 0.55,
        }

    return {
        "seeds": [42, 99],
        "candidate_limit": 2,

        "sa_iterations": 80000,
        "sa_cooling": 0.9997,
        "sa_temp_factor": 0.15,

        "tabu_iterations": 2000,
        "tabu_sample": 100,
        "tabu_tenure": 40,

        "alns_iterations": 20000,
        "alns_cooling": 0.9997,
        "alns_q_min_ratio": 0.05,
        "alns_q_max_ratio": 0.45,
    }


def summarize_runs(instance, dist, algorithm_name, runs, reference_cost):
    best_run = min(runs, key=lambda run: run["cost"])

    costs = [run["cost"] for run in runs]
    times = [run["time"] for run in runs]

    avg_cost = statistics.mean(costs)
    std_cost = statistics.stdev(costs) if len(costs) > 1 else 0.0
    avg_time = statistics.mean(times)

    best_gap = gap_percent(best_run["cost"], reference_cost)
    avg_gap = gap_percent(avg_cost, reference_cost)

    feasible, errors = is_solution_feasible(best_run["solution"], instance)

    if not feasible:
        raise ValueError(f"{algorithm_name} best solution infeasible: {errors}")

    return {
        "instance": instance.name,
        "algorithm": algorithm_name,
        "best_cost": best_run["cost"],
        "avg_cost": avg_cost,
        "std_cost": std_cost,
        "reference_cost": reference_cost,
        "best_gap_percent": best_gap,
        "avg_gap_percent": avg_gap,
        "avg_time": avg_time,
        "best_time": best_run["time"],
        "runs": len(runs),
        "best_seed": best_run["seed"],
        "best_start": best_run["start_name"],
        "feasible": feasible,
        "best_solution": best_run["solution"],
    }


def run_instance(vrp_path: Path):
    print("=" * 100)
    print("Instance:", vrp_path.name)

    instance = read_vrplib_cvrp(str(vrp_path))
    dist = build_distance_matrix(instance)
    customer_count = len(instance.coordinates) - 1

    params = get_params(instance.name, customer_count)

    reference_cost = read_reference_cost(str(vrp_path.with_suffix(".sol")))

    raw_candidates = initial_solution_candidates(instance)

    polished_candidates = []
    seen = set()

    for name, solution, cost in raw_candidates:
        polished_solution = local_search_improvement(
            solution,
            instance,
            dist,
            max_passes=20 if customer_count <= 100 else 10,
        )

        polished_cost = total_cost(polished_solution, dist)

        signature = tuple(tuple(route) for route in polished_solution)

        if signature in seen:
            continue

        seen.add(signature)

        feasible, errors = is_solution_feasible(polished_solution, instance)

        if not feasible:
            print(f"Polished initial method {name} infeasible:", errors)
            continue

        polished_candidates.append((name, polished_solution, polished_cost))

    polished_candidates.sort(key=lambda item: item[2])

    candidates = polished_candidates[:params["candidate_limit"]]

    print("\nInitial solution candidates:")
    for name, solution, cost in candidates:
        gap = gap_percent(cost, reference_cost)
        gap_text = "N/A" if gap is None else f"{gap:.2f}%"
        print(f"  {name:<15} cost={cost:.2f} gap={gap_text}")

    summaries = []

    # ------------------------------------------------------------
    # Greedy / initial baseline
    # ------------------------------------------------------------
    best_initial_name, best_initial_solution, best_initial_cost = candidates[0]

    summaries.append({
        "instance": instance.name,
        "algorithm": "Best Initial",
        "best_cost": best_initial_cost,
        "avg_cost": best_initial_cost,
        "std_cost": 0.0,
        "reference_cost": reference_cost,
        "best_gap_percent": gap_percent(best_initial_cost, reference_cost),
        "avg_gap_percent": gap_percent(best_initial_cost, reference_cost),
        "avg_time": 0.0,
        "best_time": 0.0,
        "runs": len(candidates),
        "best_seed": "-",
        "best_start": best_initial_name,
        "feasible": True,
        "best_solution": best_initial_solution,
    })

    # ------------------------------------------------------------
    # Simulated Annealing
    # ------------------------------------------------------------
    sa_runs = []

    for start_name, start_solution, start_cost in candidates:
        for seed in params["seeds"]:
            result = simulated_annealing(
                instance=instance,
                initial_solution=start_solution,
                max_iterations=params["sa_iterations"],
                initial_temperature=params["sa_temp_factor"] * start_cost,
                cooling_rate=params["sa_cooling"],
                final_temperature=1e-4,
                seed=seed,
                operator_weights={
                    "relocate": 2.0,
                    "swap": 2.0,
                    "two_opt": 1.0,
                },
            )

            polished_solution = local_search_improvement(
                result.best_solution,
                instance,
                dist,
                max_passes=30 if customer_count <= 100 else 15,
            )

            polished_cost = total_cost(polished_solution, dist)

            sa_runs.append({
                "cost": polished_cost,
                "time": result.elapsed_time,
                "seed": seed,
                "start_name": start_name,
                "solution": polished_solution,
            })

    summaries.append(
        summarize_runs(instance, dist, "Simulated Annealing", sa_runs, reference_cost)
    )

    # ------------------------------------------------------------
    # Tabu Search
    # ------------------------------------------------------------
    tabu_runs = []

    for start_name, start_solution, start_cost in candidates:
        for seed in params["seeds"]:
            result = tabu_search(
                instance=instance,
                initial_solution=start_solution,
                max_iterations=params["tabu_iterations"],
                neighborhood_sample_size=params["tabu_sample"],
                tabu_tenure=params["tabu_tenure"],
                seed=seed,
                operator_weights={
                    "relocate": 2.0,
                    "swap": 2.0,
                    "two_opt": 1.0,
                },
            )

            polished_solution = local_search_improvement(
                result.best_solution,
                instance,
                dist,
                max_passes=30 if customer_count <= 100 else 15,
            )

            polished_cost = total_cost(polished_solution, dist)

            tabu_runs.append({
                "cost": polished_cost,
                "time": result.elapsed_time,
                "seed": seed,
                "start_name": start_name,
                "solution": polished_solution,
            })

    summaries.append(
        summarize_runs(instance, dist, "Tabu Search", tabu_runs, reference_cost)
    )

    # ------------------------------------------------------------
    # ALNS
    # ------------------------------------------------------------
    alns_runs = []

    q_min = max(2, int(params["alns_q_min_ratio"] * customer_count))
    q_max = max(q_min, int(params["alns_q_max_ratio"] * customer_count))

    for start_name, start_solution, start_cost in candidates:
        for seed in params["seeds"]:
            result = alns(
                instance=instance,
                initial_solution=start_solution,
                max_iterations=params["alns_iterations"],
                seed=seed,
                initial_temperature=0.10 * start_cost,
                cooling_rate=params["alns_cooling"],
                reaction_factor=0.10,
                q_min=q_min,
                q_max=q_max,
            )

            polished_solution = local_search_improvement(
                result.best_solution,
                instance,
                dist,
                max_passes=30 if customer_count <= 100 else 15,
            )

            polished_cost = total_cost(polished_solution, dist)

            alns_runs.append({
                "cost": polished_cost,
                "time": result.elapsed_time,
                "seed": seed,
                "start_name": start_name,
                "solution": polished_solution,
            })

    summaries.append(
        summarize_runs(instance, dist, "ALNS", alns_runs, reference_cost)
    )

    # ------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------
    print("\nOptimized summary:")
    print(
        f"{'Algorithm':<22} "
        f"{'Best':>12} "
        f"{'Avg':>12} "
        f"{'Std':>10} "
        f"{'Best Gap':>10} "
        f"{'Avg Time':>10} "
        f"{'Runs':>6} "
        f"{'Best Start':>15} "
        f"{'Seed':>6}"
    )
    print("-" * 115)

    for row in summaries:
        best_gap = row["best_gap_percent"]
        best_gap_text = "N/A" if best_gap is None else f"{best_gap:.2f}%"

        print(
            f"{row['algorithm']:<22} "
            f"{row['best_cost']:>12.2f} "
            f"{row['avg_cost']:>12.2f} "
            f"{row['std_cost']:>10.2f} "
            f"{best_gap_text:>10} "
            f"{row['avg_time']:>10.2f} "
            f"{row['runs']:>6} "
            f"{row['best_start']:>15} "
            f"{str(row['best_seed']):>6}"
        )

    # ------------------------------------------------------------
    # Save best outputs
    # ------------------------------------------------------------
    output_folder = Path("outputs")
    plot_folder = Path("plots")

    output_folder.mkdir(exist_ok=True)
    plot_folder.mkdir(exist_ok=True)

    for row in summaries:
        safe_algorithm_name = row["algorithm"].lower().replace(" ", "_")

        output_path = output_folder / f"{instance.name}_{safe_algorithm_name}_optimized.txt"
        plot_path = plot_folder / f"{instance.name}_{safe_algorithm_name}_optimized.png"

        write_solution_file(
            row["best_solution"],
            dist,
            str(output_path),
            elapsed_time=row["best_time"],
        )

        plot_solution(
            instance,
            row["best_solution"],
            dist,
            save_path=str(plot_path),
            show=False,
        )

    # Remove full solution object before writing CSV
    csv_ready = []

    for row in summaries:
        row_copy = dict(row)
        row_copy.pop("best_solution")
        csv_ready.append(row_copy)

    return csv_ready


def main():
    data_folder = Path("data")
    results_folder = Path("results")
    results_folder.mkdir(exist_ok=True)

    test_files = [
        ##"P-n16-k8.vrp",
        ##"E-n22-k4.vrp",
        ##"A-n32-k5.vrp",
        ##"A-n80-k10.vrp",
        "X-n101-k25.vrp",
        ##"M-n200-k17.vrp",
    ]

    all_rows = []

    print("Optimized comparison: multi-start + multi-seed\n")

    for file_name in test_files:
        vrp_path = data_folder / file_name

        if not vrp_path.exists():
            print("Missing file:", file_name)
            continue

        rows = run_instance(vrp_path)
        all_rows.extend(rows)

    csv_path = results_folder / "optimized_my_algorithms_comparison.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "instance",
            "algorithm",
            "best_cost",
            "avg_cost",
            "std_cost",
            "reference_cost",
            "best_gap_percent",
            "avg_gap_percent",
            "avg_time",
            "best_time",
            "runs",
            "best_seed",
            "best_start",
            "feasible",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nSaved optimized comparison CSV:", csv_path)


if __name__ == "__main__":
    main()