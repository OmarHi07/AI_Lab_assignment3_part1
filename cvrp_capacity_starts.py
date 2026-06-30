import math
import random
from typing import List, Tuple

from cvrp_utils import (
    CVRPInstance,
    build_distance_matrix,
    order_route_nearest_neighbor,
    improve_solution_routes_2opt,
    is_solution_feasible,
    total_cost,
)


def capacity_dp_initial_solution_with_order(
    instance: CVRPInstance,
    customer_order: List[int],
) -> List[List[int]]:
    """
    Capacity-DP initializer, but with a chosen customer order.

    The order changes which subsets are found by the DP.
    This gives us multiple feasible starting solutions for hard tight-capacity
    instances like X-n101-k25.
    """
    dist = build_distance_matrix(instance)

    remaining_customers = set(instance.coordinates.keys()) - {0}
    vehicle_customer_groups = []

    for vehicle_index in range(instance.vehicle_count):
        vehicles_left_after_this = instance.vehicle_count - vehicle_index - 1
        remaining_demand = sum(instance.demands[c] for c in remaining_customers)

        if not remaining_customers:
            vehicle_customer_groups.append([])
            continue

        min_required_load = max(
            0,
            remaining_demand - vehicles_left_after_this * instance.capacity
        )

        max_allowed_load = min(instance.capacity, remaining_demand)

        customers_list = [
            c for c in customer_order
            if c in remaining_customers
        ]

        reachable = {0}
        parent = {}

        for customer in customers_list:
            demand = instance.demands[customer]

            for current_sum in sorted(list(reachable), reverse=True):
                new_sum = current_sum + demand

                if new_sum > max_allowed_load:
                    continue

                if new_sum not in reachable:
                    reachable.add(new_sum)
                    parent[new_sum] = (current_sum, customer)

        valid_sums = [
            s for s in reachable
            if min_required_load <= s <= max_allowed_load
        ]

        if not valid_sums:
            raise ValueError(
                "Ordered capacity-DP failed. "
                f"vehicle={vehicle_index}, "
                f"remaining demand={remaining_demand}, "
                f"min required={min_required_load}, "
                f"max allowed={max_allowed_load}"
            )

        chosen_sum = max(valid_sums)

        chosen_customers = []
        current_sum = chosen_sum

        while current_sum > 0:
            previous_sum, customer = parent[current_sum]
            chosen_customers.append(customer)
            current_sum = previous_sum

        for customer in chosen_customers:
            remaining_customers.remove(customer)

        vehicle_customer_groups.append(chosen_customers)

    if remaining_customers:
        raise ValueError(
            "Ordered capacity-DP failed: unassigned customers "
            f"{sorted(remaining_customers)}"
        )

    solution = [
        order_route_nearest_neighbor(customers, dist)
        for customers in vehicle_customer_groups
    ]

    while len(solution) < instance.vehicle_count:
        solution.append([0, 0])

    solution = improve_solution_routes_2opt(solution, dist)

    feasible, errors = is_solution_feasible(solution, instance)

    if not feasible:
        raise ValueError(f"Ordered capacity-DP infeasible: {errors}")

    return solution


def build_angle_order(
    instance: CVRPInstance,
    offset: float = 0.0,
    reverse: bool = False,
) -> List[int]:
    depot_x, depot_y = instance.coordinates[0]

    customers = [node for node in instance.coordinates if node != 0]

    customers.sort(
        key=lambda node: (
            math.atan2(
                instance.coordinates[node][1] - depot_y,
                instance.coordinates[node][0] - depot_x,
            ) - offset
        ) % (2.0 * math.pi)
    )

    if reverse:
        customers.reverse()

    return customers


def build_distance_from_depot_order(
    instance: CVRPInstance,
    reverse: bool = False,
) -> List[int]:
    depot_x, depot_y = instance.coordinates[0]

    customers = [node for node in instance.coordinates if node != 0]

    customers.sort(
        key=lambda node: math.hypot(
            instance.coordinates[node][0] - depot_x,
            instance.coordinates[node][1] - depot_y,
        ),
        reverse=reverse,
    )

    return customers


def randomized_capacity_dp_candidates(
    instance: CVRPInstance,
    random_seeds: List[int] = None,
) -> List[Tuple[str, List[List[int]], float]]:
    """
    Generate many feasible capacity-DP starts using different customer orders.

    This is designed for tight-capacity instances where normal greedy,
    Clarke-Wright, and bin-packing may fail.
    """
    if random_seeds is None:
        random_seeds = [1, 7, 42, 99, 123, 2024]

    dist = build_distance_matrix(instance)

    orders = []

    # Angle-based orders. Sampled every pi/16 (11.25 deg) instead of pi/8:
    # capacity_dp_angle_2.75_rev (~7pi/8) has consistently been the strongest
    # partition found so far, so a denser sweep gives the top-10 selection in
    # tune_x101_alns.py a better chance at finding an even better one nearby.
    offsets = [k * math.pi / 16 for k in range(17)]

    for offset in offsets:
        orders.append((f"angle_{offset:.2f}_fwd", build_angle_order(instance, offset, False)))
        orders.append((f"angle_{offset:.2f}_rev", build_angle_order(instance, offset, True)))

    # Distance-from-depot orders.
    orders.append(("near_to_far", build_distance_from_depot_order(instance, reverse=False)))
    orders.append(("far_to_near", build_distance_from_depot_order(instance, reverse=True)))

    # Demand orders.
    customers = [node for node in instance.coordinates if node != 0]

    demand_desc = sorted(customers, key=lambda c: instance.demands[c], reverse=True)
    demand_asc = sorted(customers, key=lambda c: instance.demands[c])

    orders.append(("demand_desc", demand_desc))
    orders.append(("demand_asc", demand_asc))

    # Random orders.
    for seed in random_seeds:
        rng = random.Random(seed)
        random_order = customers[:]
        rng.shuffle(random_order)
        orders.append((f"random_{seed}", random_order))

    candidates = []
    seen = set()

    for name, order in orders:
        try:
            solution = capacity_dp_initial_solution_with_order(instance, order)

            signature = tuple(tuple(route) for route in solution)

            if signature in seen:
                continue

            seen.add(signature)

            cost = total_cost(solution, dist)
            candidates.append((f"capacity_dp_{name}", solution, cost))

        except Exception as e:
            print(f"Capacity-DP candidate failed {name}: {e}")

    candidates.sort(key=lambda item: item[2])

    return candidates