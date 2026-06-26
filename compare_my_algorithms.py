import csv
from pathlib import Path

from cvrp_utils import (
    read_vrplib_cvrp,
    read_reference_cost,
    build_distance_matrix,
    greedy_initial_solution,
    total_cost,
    is_solution_feasible,
    write_solution_file,
    plot_solution,
)

from cvrp_sa import simulated_annealing
from cvrp_tabu import tabu_search
from cvrp_alns import alns


def gap_percent(cost, reference_cost):
    if reference_cost is None:
        return None
    return ((cost - reference_cost) / reference_cost) * 100


def run_comparison_on_instance(vrp_path: Path):
    print("=" * 90)
    print("Instance:", vrp_path.name)

    instance = read_vrplib_cvrp(str(vrp_path))
    dist = build_distance_matrix(instance)

    reference_cost = read_reference_cost(str(vrp_path.with_suffix(".sol")))

    output_folder = Path("outputs")
    plot_folder = Path("plots")

    output_folder.mkdir(exist_ok=True)
    plot_folder.mkdir(exist_ok=True)

    rows = []

    # -------------------------
    # Greedy baseline
    # -------------------------
    greedy_solution = greedy_initial_solution(instance)
    greedy_cost = total_cost(greedy_solution, dist)

    feasible, errors = is_solution_feasible(greedy_solution, instance)

    if not feasible:
        raise ValueError(f"Greedy solution infeasible: {errors}")

    rows.append({
        "instance": instance.name,
        "algorithm": "Greedy",
        "cost": greedy_cost,
        "reference_cost": reference_cost,
        "gap_percent": gap_percent(greedy_cost, reference_cost),
        "improvement_from_greedy": 0.0,
        "improvement_percent": 0.0,
        "elapsed_time": 0.0,
        "feasible": feasible,
    })

    write_solution_file(
        greedy_solution,
        dist,
        str(output_folder / f"{instance.name}_greedy_compare.txt"),
        elapsed_time=0.0,
    )

    # -------------------------
    # Simulated Annealing
    # -------------------------
    sa_result = simulated_annealing(
        instance=instance,
        initial_solution=greedy_solution,
        max_iterations=10000,
        cooling_rate=0.995,
        seed=42,
    )

    feasible, errors = is_solution_feasible(sa_result.best_solution, instance)

    if not feasible:
        raise ValueError(f"SA solution infeasible: {errors}")

    rows.append({
        "instance": instance.name,
        "algorithm": "Simulated Annealing",
        "cost": sa_result.best_cost,
        "reference_cost": reference_cost,
        "gap_percent": gap_percent(sa_result.best_cost, reference_cost),
        "improvement_from_greedy": greedy_cost - sa_result.best_cost,
        "improvement_percent": ((greedy_cost - sa_result.best_cost) / greedy_cost) * 100,
        "elapsed_time": sa_result.elapsed_time,
        "feasible": feasible,
    })

    write_solution_file(
        sa_result.best_solution,
        dist,
        str(output_folder / f"{instance.name}_sa_compare.txt"),
        elapsed_time=sa_result.elapsed_time,
    )

    plot_solution(
        instance,
        sa_result.best_solution,
        dist,
        save_path=str(plot_folder / f"{instance.name}_sa_compare.png"),
        show=False,
    )

    # -------------------------
    # Tabu Search
    # -------------------------
    tabu_result = tabu_search(
        instance=instance,
        initial_solution=greedy_solution,
        max_iterations=1000,
        neighborhood_sample_size=80,
        tabu_tenure=20,
        seed=42,
    )

    feasible, errors = is_solution_feasible(tabu_result.best_solution, instance)

    if not feasible:
        raise ValueError(f"Tabu solution infeasible: {errors}")

    rows.append({
        "instance": instance.name,
        "algorithm": "Tabu Search",
        "cost": tabu_result.best_cost,
        "reference_cost": reference_cost,
        "gap_percent": gap_percent(tabu_result.best_cost, reference_cost),
        "improvement_from_greedy": greedy_cost - tabu_result.best_cost,
        "improvement_percent": ((greedy_cost - tabu_result.best_cost) / greedy_cost) * 100,
        "elapsed_time": tabu_result.elapsed_time,
        "feasible": feasible,
    })

    write_solution_file(
        tabu_result.best_solution,
        dist,
        str(output_folder / f"{instance.name}_tabu_compare.txt"),
        elapsed_time=tabu_result.elapsed_time,
    )

    plot_solution(
        instance,
        tabu_result.best_solution,
        dist,
        save_path=str(plot_folder / f"{instance.name}_tabu_compare.png"),
        show=False,
    )

    # -------------------------
    # ALNS
    # -------------------------
    alns_result = alns(
        instance=instance,
        initial_solution=greedy_solution,
        max_iterations=2000,
        seed=42,
        cooling_rate=0.999,
        reaction_factor=0.15,
    )

    feasible, errors = is_solution_feasible(alns_result.best_solution, instance)

    if not feasible:
        raise ValueError(f"ALNS solution infeasible: {errors}")

    rows.append({
        "instance": instance.name,
        "algorithm": "ALNS",
        "cost": alns_result.best_cost,
        "reference_cost": reference_cost,
        "gap_percent": gap_percent(alns_result.best_cost, reference_cost),
        "improvement_from_greedy": greedy_cost - alns_result.best_cost,
        "improvement_percent": ((greedy_cost - alns_result.best_cost) / greedy_cost) * 100,
        "elapsed_time": alns_result.elapsed_time,
        "feasible": feasible,
    })

    write_solution_file(
        alns_result.best_solution,
        dist,
        str(output_folder / f"{instance.name}_alns_compare.txt"),
        elapsed_time=alns_result.elapsed_time,
    )

    plot_solution(
        instance,
        alns_result.best_solution,
        dist,
        save_path=str(plot_folder / f"{instance.name}_alns_compare.png"),
        show=False,
    )

    # -------------------------
    # Print instance summary
    # -------------------------
    print(f"{'Algorithm':<22} {'Cost':>12} {'Gap %':>12} {'Improve %':>12} {'Time':>10}")
    print("-" * 75)

    for row in rows:
        gap = row["gap_percent"]

        gap_text = "N/A" if gap is None else f"{gap:.2f}"

        print(
            f"{row['algorithm']:<22} "
            f"{row['cost']:>12.2f} "
            f"{gap_text:>12} "
            f"{row['improvement_percent']:>12.2f} "
            f"{row['elapsed_time']:>10.2f}"
        )

    return rows


def main():
    data_folder = Path("data")
    results_folder = Path("results")
    results_folder.mkdir(exist_ok=True)

    test_files = [
        "P-n16-k8.vrp",
        "E-n22-k4.vrp",
        "A-n32-k5.vrp",
        "A-n80-k10.vrp",
        "X-n101-k25.vrp",
        "M-n200-k17.vrp",
    ]

    all_rows = []

    print("Comparison: Greedy vs SA vs Tabu vs ALNS\n")

    for file_name in test_files:
        vrp_path = data_folder / file_name

        if not vrp_path.exists():
            print("Missing file:", file_name)
            continue

        rows = run_comparison_on_instance(vrp_path)
        all_rows.extend(rows)

    csv_path = results_folder / "my_algorithms_comparison.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "instance",
            "algorithm",
            "cost",
            "reference_cost",
            "gap_percent",
            "improvement_from_greedy",
            "improvement_percent",
            "elapsed_time",
            "feasible",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nSaved comparison CSV:", csv_path)


if __name__ == "__main__":
    main()