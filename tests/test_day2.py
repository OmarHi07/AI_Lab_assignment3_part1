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


def main():
    data_folder = Path("data")
    output_folder = Path("outputs")
    plot_folder = Path("plots")

    output_folder.mkdir(exist_ok=True)
    plot_folder.mkdir(exist_ok=True)

    vrp_files = sorted(data_folder.glob("*.vrp"))

    if not vrp_files:
        print("No .vrp files found in data/ folder.")
        return

    print("Day 2 test: output files + plots + reference comparison\n")

    for vrp_path in vrp_files:
        print("=" * 80)
        print("Instance:", vrp_path.name)

        instance = read_vrplib_cvrp(str(vrp_path))
        dist = build_distance_matrix(instance)

        solution = greedy_initial_solution(instance)
        cost = total_cost(solution, dist)

        feasible, errors = is_solution_feasible(solution, instance)

        sol_path = vrp_path.with_suffix(".sol")
        reference_cost = read_reference_cost(str(sol_path))

        print("Greedy initial cost:", round(cost, 2))
        print("Feasible?", feasible)

        if reference_cost is not None:
            gap = ((cost - reference_cost) / reference_cost) * 100
            print("Reference cost:", reference_cost)
            print("Gap from reference:", f"{gap:.2f}%")
        else:
            print("Reference cost: not found")

        if errors:
            print("Errors:")
            for error in errors:
                print("-", error)

        output_path = output_folder / f"{instance.name}_greedy.txt"
        plot_path = plot_folder / f"{instance.name}_greedy.png"

        write_solution_file(
            solution=solution,
            dist=dist,
            output_path=str(output_path),
            elapsed_time=0.0
        )

        plot_solution(
            instance=instance,
            solution=solution,
            dist=dist,
            save_path=str(plot_path),
            show=False
        )

        print("Saved output:", output_path)
        print("Saved plot:", plot_path)


if __name__ == "__main__":
    main()