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

from cvrp_alns import alns


def run_alns_on_instance(vrp_path: Path):
    print("=" * 80)
    print("Instance:", vrp_path.name)

    instance = read_vrplib_cvrp(str(vrp_path))
    dist = build_distance_matrix(instance)

    initial_solution = greedy_initial_solution(instance)
    initial_cost = total_cost(initial_solution, dist)

    feasible, errors = is_solution_feasible(initial_solution, instance)

    if not feasible:
        print("Initial solution infeasible!")
        print(errors)
        return

    reference_cost = read_reference_cost(str(vrp_path.with_suffix(".sol")))

    result = alns(
        instance=instance,
        initial_solution=initial_solution,
        max_iterations=2000,
        seed=42,
        cooling_rate=0.999,
        reaction_factor=0.15,
    )

    feasible, errors = is_solution_feasible(result.best_solution, instance)

    print("Initial greedy cost:", round(initial_cost, 2))
    print("ALNS best cost:", round(result.best_cost, 2))
    print("Improvement:", f"{initial_cost - result.best_cost:.2f}")
    print("Improvement %:", f"{((initial_cost - result.best_cost) / initial_cost) * 100:.2f}%")
    print("Feasible?", feasible)
    print("Iterations:", result.iterations)
    print("Accepted moves:", result.accepted_moves)
    print("Improved moves:", result.improved_moves)
    print("Global best updates:", result.global_best_updates)
    print("Elapsed time:", f"{result.elapsed_time:.2f} sec")

    if reference_cost is not None:
        gap = ((result.best_cost - reference_cost) / reference_cost) * 100
        print("Reference cost:", reference_cost)
        print("Gap from reference:", f"{gap:.2f}%")
    else:
        print("Reference cost: not found")

    print("Final destroy weights:", result.destroy_weights)
    print("Final repair weights:", result.repair_weights)
    print("Destroy counts:", result.destroy_counts)
    print("Repair counts:", result.repair_counts)

    if errors:
        print("Errors:")
        for error in errors:
            print("-", error)

    output_folder = Path("outputs")
    plot_folder = Path("plots")

    output_folder.mkdir(exist_ok=True)
    plot_folder.mkdir(exist_ok=True)

    output_path = output_folder / f"{instance.name}_alns.txt"
    plot_path = plot_folder / f"{instance.name}_alns.png"

    write_solution_file(
        solution=result.best_solution,
        dist=dist,
        output_path=str(output_path),
        elapsed_time=result.elapsed_time,
    )

    plot_solution(
        instance=instance,
        solution=result.best_solution,
        dist=dist,
        save_path=str(plot_path),
        show=False,
    )

    print("Saved output:", output_path)
    print("Saved plot:", plot_path)


def main():
    data_folder = Path("data")

    test_files = [
        "P-n16-k8.vrp",
        "E-n22-k4.vrp",
        "A-n32-k5.vrp",
    ]

    print("Day 6-7: ALNS test\n")

    for file_name in test_files:
        vrp_path = data_folder / file_name

        if not vrp_path.exists():
            print("Missing file:", file_name)
            continue

        run_alns_on_instance(vrp_path)


if __name__ == "__main__":
    main()