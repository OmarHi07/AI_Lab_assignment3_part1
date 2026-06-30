import random
from pathlib import Path

from cvrp_utils import (
    read_vrplib_cvrp,
    read_reference_cost,
    build_distance_matrix,
    total_cost,
    is_solution_feasible,
    read_existing_solution_cost,
    write_solution_file,
    plot_solution,
)

from cvrp_local_search import (
    local_search_improvement,
    two_route_repair_local_search,
    two_opt_star_local_search,
    group_exchange_local_search,
    exact_route_polish_solution,
)

from cvrp_alns import (
    alns,
    destroy_random,
    destroy_related,
    remove_customers_from_solution,
    repair_regret_3_randomized,
)

# Stable best result for X-n101-k25 achieved so far. A run is only allowed to
# overwrite the stable output files if it beats this (or the cost already on
# disk, whichever is lower) strictly.
STABLE_BEST_COST = 31512.0


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


def full_polish(solution, instance, dist):
    """
    Same final polish chain used by tune_x101_alns.py / polish_x101_best.py.
    """
    polished = local_search_improvement(solution, instance, dist, max_passes=20)

    polished = two_route_repair_local_search(
        polished, instance, dist, max_passes=15, max_combined_customers=16,
    )

    polished = two_opt_star_local_search(polished, instance, dist, max_passes=20)

    polished = group_exchange_local_search(
        polished, instance, dist, max_passes=10, max_group_size=2, max_route_customers=10,
    )

    polished = local_search_improvement(polished, instance, dist, max_passes=20)
    polished = exact_route_polish_solution(polished, dist, max_customers=8)

    return polished


def ils_round(
    best_solution,
    best_cost,
    instance,
    dist,
    rng,
    q_min,
    q_max,
    burst_iterations,
    remove_count_range,
):
    """
    One Iterated Local Search round:

    1. Perturb the current best with a moderate random/related destroy.
    2. Repair with randomized regret-3 insertion.
    3. Re-settle with a short ALNS burst.
    4. Run the full polish chain.

    Returns (candidate_solution, candidate_cost, feasible) or None if the
    destroy/repair step itself failed (common on this tight-capacity
    instance with larger removals).
    """
    destroy_fn = rng.choice([destroy_random, destroy_related])
    remove_count = rng.randint(*remove_count_range)

    removed = destroy_fn(best_solution, instance, dist, rng, remove_count)

    if not removed:
        return None

    partial = remove_customers_from_solution(best_solution, removed)
    perturbed = repair_regret_3_randomized(partial, removed, instance, dist, rng)

    if perturbed is None:
        return None

    burst_seed = rng.randint(0, 10**6)

    burst_result = alns(
        instance=instance,
        initial_solution=perturbed,
        max_iterations=burst_iterations,
        seed=burst_seed,
        initial_temperature=0.10 * best_cost,
        cooling_rate=0.999,
        reaction_factor=0.10,
        q_min=q_min,
        q_max=q_max,
    )

    candidate = full_polish(burst_result.best_solution, instance, dist)
    candidate_cost = total_cost(candidate, dist)
    feasible, errors = is_solution_feasible(candidate, instance)

    return candidate, candidate_cost, feasible, errors, destroy_fn.__name__, len(removed)


def main():
    vrp_path = Path("data") / "X-n101-k25.vrp"
    solution_path = Path("outputs") / "X-n101-k25_alns_tuned_best.txt"

    instance = read_vrplib_cvrp(str(vrp_path))
    dist = build_distance_matrix(instance)
    reference_cost = read_reference_cost(str(vrp_path.with_suffix(".sol")))

    if not solution_path.exists():
        raise FileNotFoundError(
            f"{solution_path} not found. Run tune_x101_alns.py first to "
            "produce a stable best solution to perturb."
        )

    solution = read_saved_solution(solution_path)

    feasible, errors = is_solution_feasible(solution, instance)

    if not feasible:
        raise ValueError(f"Loaded solution infeasible: {errors}")

    best_solution = solution
    best_cost = total_cost(best_solution, dist)

    print("Loaded solution:")
    print(f"cost={best_cost:.2f}, gap={gap_percent(best_cost, reference_cost):.2f}%")

    customer_count = len(instance.coordinates) - 1
    q_min = max(2, int(0.03 * customer_count))
    q_max = max(q_min, int(0.35 * customer_count))

    ROUNDS = 50
    BURST_ITERS = 4000
    REMOVE_COUNT_RANGE = (8, 15)
    SEED = 2026

    rng = random.Random(SEED)

    print(f"\nStarting {ROUNDS} ILS rounds from cost={best_cost:.2f}")
    print(f"(burst_iterations={BURST_ITERS}, remove_count={REMOVE_COUNT_RANGE}, seed={SEED})\n")

    succeeded = 0

    for round_idx in range(1, ROUNDS + 1):
        result = ils_round(
            best_solution,
            best_cost,
            instance,
            dist,
            rng,
            q_min,
            q_max,
            BURST_ITERS,
            REMOVE_COUNT_RANGE,
        )

        if result is None:
            continue

        candidate, candidate_cost, feasible, errors, destroy_name, removed_count = result
        succeeded += 1

        if not feasible:
            print(f"Round {round_idx}: infeasible result, discarding. {errors}")
            continue

        gap = gap_percent(candidate_cost, reference_cost)
        improved = candidate_cost < best_cost - 1e-6

        print(
            f"Round {round_idx}: destroy={destroy_name}, removed={removed_count}, "
            f"cost={candidate_cost:.2f}, gap={gap:.2f}%"
            + (" NEW BEST!" if improved else "")
        )

        if improved:
            best_solution = candidate
            best_cost = candidate_cost

    print(f"\nRounds where destroy/repair succeeded: {succeeded}/{ROUNDS}")

    print("\n" + "=" * 80)
    print("ILS RESULT FOR X-n101-k25")
    print("Cost:", best_cost)
    print("Reference:", reference_cost)
    print("Gap:", f"{gap_percent(best_cost, reference_cost):.2f}%")

    feasible, errors = is_solution_feasible(best_solution, instance)

    if not feasible:
        print("Final solution infeasible, not writing:", errors)
        return

    output_folder = Path("outputs")
    plot_folder = Path("plots")

    output_folder.mkdir(exist_ok=True)
    plot_folder.mkdir(exist_ok=True)

    output_path = output_folder / "X-n101-k25_alns_tuned_best.txt"
    plot_path = plot_folder / "X-n101-k25_alns_tuned_best.png"

    existing_cost = read_existing_solution_cost(str(output_path))
    protected_cost = STABLE_BEST_COST if existing_cost is None else min(existing_cost, STABLE_BEST_COST)

    if best_cost < protected_cost - 1e-6:
        write_solution_file(best_solution, dist, str(output_path), elapsed_time=0.0)

        plot_solution(
            instance,
            best_solution,
            dist,
            save_path=str(plot_path),
            show=False,
        )

        print(f"NEW STABLE BEST! Saved ILS output and plot (cost={best_cost:.2f}).")
    else:
        print(
            f"No improvement over stable best ({protected_cost:.2f}). "
            "Stable output files left untouched."
        )


if __name__ == "__main__":
    main()
