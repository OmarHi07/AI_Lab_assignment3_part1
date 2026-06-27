import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from cvrp_utils import (
    CVRPInstance,
    build_distance_matrix,
    greedy_initial_solution,
    is_solution_feasible,
    route_cost,
    route_demand,
    total_cost,
    improve_solution_routes_2opt,
)

from cvrp_operators import copy_solution, normalize_empty_route


@dataclass
class ALNSResult:
    best_solution: List[List[int]]
    best_cost: float
    initial_cost: float
    elapsed_time: float
    iterations: int
    accepted_moves: int
    improved_moves: int
    global_best_updates: int
    destroy_weights: Dict[str, float]
    repair_weights: Dict[str, float]
    destroy_counts: Dict[str, int]
    repair_counts: Dict[str, int]
    history: List[float]


def all_customers_in_solution(solution: List[List[int]]) -> List[int]:
    customers = []

    for route in solution:
        for node in route:
            if node != 0:
                customers.append(node)

    return customers


def remove_customers_from_solution(
    solution: List[List[int]],
    customers_to_remove: List[int]
) -> List[List[int]]:
    remove_set = set(customers_to_remove)
    new_solution = []

    for route in solution:
        remaining_customers = [
            node for node in route
            if node != 0 and node not in remove_set
        ]

        new_solution.append(normalize_empty_route([0] + remaining_customers + [0]))

    return new_solution


def find_feasible_insertions(
    solution: List[List[int]],
    customer: int,
    instance: CVRPInstance,
    dist: List[List[float]],
) -> List[Tuple[float, int, int]]:
    """
    Return all feasible insertions for a customer.

    Each insertion is:
        (delta_cost, route_index, insert_position)

    insert_position is the index where the customer should be inserted.
    """
    insertions = []
    customer_demand = instance.demands[customer]

    for route_index, route in enumerate(solution):
        current_demand = route_demand(route, instance.demands)

        if current_demand + customer_demand > instance.capacity:
            continue

        for insert_pos in range(1, len(route)):
            previous_node = route[insert_pos - 1]
            next_node = route[insert_pos]

            delta = (
                dist[previous_node][customer]
                + dist[customer][next_node]
                - dist[previous_node][next_node]
            )

            insertions.append((delta, route_index, insert_pos))

    return insertions


def apply_insertion(
    solution: List[List[int]],
    customer: int,
    route_index: int,
    insert_pos: int,
) -> List[List[int]]:
    new_solution = copy_solution(solution)
    new_solution[route_index].insert(insert_pos, customer)
    new_solution[route_index] = normalize_empty_route(new_solution[route_index])
    return new_solution


# ============================================================
# Destroy operators
# ============================================================

def destroy_random(
    solution: List[List[int]],
    instance: CVRPInstance,
    dist: List[List[float]],
    rng: random.Random,
    remove_count: int,
) -> List[int]:
    customers = all_customers_in_solution(solution)

    remove_count = min(remove_count, len(customers))

    return rng.sample(customers, remove_count)


def destroy_worst(
    solution: List[List[int]],
    instance: CVRPInstance,
    dist: List[List[float]],
    rng: random.Random,
    remove_count: int,
) -> List[int]:
    """
    Remove customers that contribute a lot to route cost.

    Contribution is estimated by:
        d(prev, customer) + d(customer, next) - d(prev, next)
    """
    contributions = []

    for route in solution:
        for pos in range(1, len(route) - 1):
            customer = route[pos]
            previous_node = route[pos - 1]
            next_node = route[pos + 1]

            saving_if_removed = (
                dist[previous_node][customer]
                + dist[customer][next_node]
                - dist[previous_node][next_node]
            )

            contributions.append((saving_if_removed, customer))

    contributions.sort(reverse=True)

    selected = [customer for _, customer in contributions[:remove_count]]

    return selected


def destroy_related(
    solution: List[List[int]],
    instance: CVRPInstance,
    dist: List[List[float]],
    rng: random.Random,
    remove_count: int,
) -> List[int]:
    """
    Remove customers that are geographically related.

    Pick one seed customer, then remove nearby customers.
    """
    customers = all_customers_in_solution(solution)

    if not customers:
        return []

    seed_customer = rng.choice(customers)

    related = sorted(
        customers,
        key=lambda customer: dist[seed_customer][customer]
    )

    return related[:remove_count]


def destroy_longest_route(
    solution: List[List[int]],
    instance: CVRPInstance,
    dist: List[List[float]],
    rng: random.Random,
    remove_count: int,
) -> List[int]:
    """
    Remove customers from the most expensive routes.
    """
    route_infos = []

    for route_index, route in enumerate(solution):
        customers = [node for node in route if node != 0]

        if not customers:
            continue

        cost = route_cost(route, dist)
        route_infos.append((cost, route_index, route))

    route_infos.sort(reverse=True)

    selected = []

    for _, _, route in route_infos:
        customer_contributions = []

        for pos in range(1, len(route) - 1):
            customer = route[pos]
            previous_node = route[pos - 1]
            next_node = route[pos + 1]

            saving_if_removed = (
                dist[previous_node][customer]
                + dist[customer][next_node]
                - dist[previous_node][next_node]
            )

            customer_contributions.append((saving_if_removed, customer))

        customer_contributions.sort(reverse=True)

        for _, customer in customer_contributions:
            selected.append(customer)

            if len(selected) >= remove_count:
                return selected

    return selected

def destroy_random_routes(
    solution: List[List[int]],
    instance: CVRPInstance,
    dist: List[List[float]],
    rng: random.Random,
    remove_count: int,
) -> List[int]:
    """
    Remove complete routes until approximately remove_count customers are removed.

    This is useful when the current route grouping is bad.
    It allows ALNS to rebuild entire vehicle routes.
    """
    non_empty_routes = [
        route for route in solution
        if any(node != 0 for node in route)
    ]

    if not non_empty_routes:
        return []

    rng.shuffle(non_empty_routes)

    removed = []

    for route in non_empty_routes:
        customers = [node for node in route if node != 0]
        removed.extend(customers)

        if len(removed) >= remove_count:
            break

    return removed


# ============================================================
# Repair operators
# ============================================================

def repair_greedy_order(
    partial_solution: List[List[int]],
    removed_customers: List[int],
    instance: CVRPInstance,
    dist: List[List[float]],
    rng: random.Random,
) -> Optional[List[List[int]]]:
    """
    Insert removed customers one by one in random order using cheapest position.
    """
    solution = copy_solution(partial_solution)
    customers = removed_customers[:]
    rng.shuffle(customers)

    for customer in customers:
        insertions = find_feasible_insertions(solution, customer, instance, dist)

        if not insertions:
            return None

        delta, route_index, insert_pos = min(insertions, key=lambda item: item[0])
        solution = apply_insertion(solution, customer, route_index, insert_pos)

    return solution


def repair_cheapest_global(
    partial_solution: List[List[int]],
    removed_customers: List[int],
    instance: CVRPInstance,
    dist: List[List[float]],
    rng: random.Random,
) -> Optional[List[List[int]]]:
    """
    At each step, insert the customer that has the cheapest possible insertion.
    """
    solution = copy_solution(partial_solution)
    remaining = set(removed_customers)

    while remaining:
        best_choice = None

        for customer in remaining:
            insertions = find_feasible_insertions(solution, customer, instance, dist)

            if not insertions:
                continue

            delta, route_index, insert_pos = min(insertions, key=lambda item: item[0])

            if best_choice is None or delta < best_choice[0]:
                best_choice = (delta, customer, route_index, insert_pos)

        if best_choice is None:
            return None

        _, customer, route_index, insert_pos = best_choice

        solution = apply_insertion(solution, customer, route_index, insert_pos)
        remaining.remove(customer)

    return solution


def repair_regret_2(
    partial_solution: List[List[int]],
    removed_customers: List[int],
    instance: CVRPInstance,
    dist: List[List[float]],
    rng: random.Random,
) -> Optional[List[List[int]]]:
    """
    Regret-2 insertion.

    For each customer:
    - find best insertion cost
    - find second-best insertion cost
    - regret = second_best - best

    Insert the customer with the largest regret first.
    """
    solution = copy_solution(partial_solution)
    remaining = set(removed_customers)

    while remaining:
        best_customer_choice = None

        for customer in remaining:
            insertions = find_feasible_insertions(solution, customer, instance, dist)

            if not insertions:
                continue

            insertions.sort(key=lambda item: item[0])

            best_delta, best_route_index, best_insert_pos = insertions[0]

            if len(insertions) >= 2:
                second_best_delta = insertions[1][0]
                regret = second_best_delta - best_delta
            else:
                # If there is only one feasible insertion, prioritize it.
                regret = 10**9

            candidate = (
                regret,
                best_delta,
                customer,
                best_route_index,
                best_insert_pos,
            )

            if best_customer_choice is None or candidate > best_customer_choice:
                best_customer_choice = candidate

        if best_customer_choice is None:
            return None

        _, _, customer, route_index, insert_pos = best_customer_choice

        solution = apply_insertion(solution, customer, route_index, insert_pos)
        remaining.remove(customer)

    return solution

def repair_regret_3(
    partial_solution: List[List[int]],
    removed_customers: List[int],
    instance: CVRPInstance,
    dist: List[List[float]],
    rng: random.Random,
) -> Optional[List[List[int]]]:
    """
    Regret-3 insertion.

    Similar to regret-2, but compares the best insertion with the third-best.
    This often works better for difficult CVRP instances because it inserts
    constrained customers earlier.
    """
    solution = copy_solution(partial_solution)
    remaining = set(removed_customers)

    while remaining:
        best_customer_choice = None

        for customer in remaining:
            insertions = find_feasible_insertions(solution, customer, instance, dist)

            if not insertions:
                continue

            insertions.sort(key=lambda item: item[0])

            best_delta, best_route_index, best_insert_pos = insertions[0]

            if len(insertions) >= 3:
                third_best_delta = insertions[2][0]
                regret = third_best_delta - best_delta
            elif len(insertions) == 2:
                regret = insertions[1][0] - best_delta
            else:
                regret = 10**9

            candidate = (
                regret,
                best_delta,
                customer,
                best_route_index,
                best_insert_pos,
            )

            if best_customer_choice is None or candidate > best_customer_choice:
                best_customer_choice = candidate

        if best_customer_choice is None:
            return None

        _, _, customer, route_index, insert_pos = best_customer_choice

        solution = apply_insertion(solution, customer, route_index, insert_pos)
        remaining.remove(customer)

    return solution




# ============================================================
# ALNS core
# ============================================================

def weighted_choice(
    weights: Dict[str, float],
    rng: random.Random,
) -> str:
    names = list(weights.keys())
    values = list(weights.values())

    return rng.choices(names, weights=values, k=1)[0]


def update_weight(
    weights: Dict[str, float],
    name: str,
    reward: float,
    reaction_factor: float,
    minimum_weight: float = 0.1,
) -> None:
    """
    Simple adaptive update.

    Higher reward increases operator weight.
    Low reward slowly decreases it.
    """
    weights[name] = max(
        minimum_weight,
        (1.0 - reaction_factor) * weights[name] + reaction_factor * reward
    )


def alns(
    instance: CVRPInstance,
    initial_solution: Optional[List[List[int]]] = None,
    max_iterations: int = 2000,
    seed: int = 42,
    initial_temperature: Optional[float] = None,
    cooling_rate: float = 0.999,
    reaction_factor: float = 0.15,
    q_min: Optional[int] = None,
    q_max: Optional[int] = None,
) -> ALNSResult:
    """
    Adaptive Large Neighborhood Search for CVRP.

    Main loop:
    1. Select destroy operator.
    2. Select repair operator.
    3. Remove q customers.
    4. Reinsert customers.
    5. Accept/reject using SA-style acceptance.
    6. Reward useful operators.
    """
    start_time = time.time()
    rng = random.Random(seed)

    dist = build_distance_matrix(instance)

    if initial_solution is None:
        current_solution = greedy_initial_solution(instance)
    else:
        current_solution = copy_solution(initial_solution)

    feasible, errors = is_solution_feasible(current_solution, instance)

    if not feasible:
        raise ValueError(f"Initial solution is infeasible: {errors}")

    current_cost = total_cost(current_solution, dist)

    best_solution = copy_solution(current_solution)
    best_cost = current_cost
    initial_cost = current_cost

    customer_count = len(instance.coordinates) - 1

    if q_min is None:
        q_min = max(2, int(0.05 * customer_count))

    if q_max is None:
        q_max = max(q_min, int(0.20 * customer_count))

    q_max = min(q_max, customer_count)

    if initial_temperature is None:
        temperature = max(1.0, 0.05 * current_cost)
    else:
        temperature = initial_temperature

    destroy_operators: Dict[str, Callable] = {
        "random": destroy_random,
        "worst": destroy_worst,
        "related": destroy_related,
        "longest_route": destroy_longest_route,
        "random_routes": destroy_random_routes,
    }

    repair_operators: Dict[str, Callable] = {
        "greedy_order": repair_greedy_order,
        "cheapest_global": repair_cheapest_global,
        "regret_2": repair_regret_2,
        "regret_3": repair_regret_3,
    }

    destroy_weights = {name: 1.0 for name in destroy_operators}
    repair_weights = {name: 1.0 for name in repair_operators}

    destroy_counts = {name: 0 for name in destroy_operators}
    repair_counts = {name: 0 for name in repair_operators}

    accepted_moves = 0
    improved_moves = 0
    global_best_updates = 0

    history = [best_cost]

    for iteration in range(1, max_iterations + 1):
        destroy_name = weighted_choice(destroy_weights, rng)
        repair_name = weighted_choice(repair_weights, rng)

        destroy_counts[destroy_name] += 1
        repair_counts[repair_name] += 1

        destroy_func = destroy_operators[destroy_name]
        repair_func = repair_operators[repair_name]

        remove_count = rng.randint(q_min, q_max)

        removed_customers = destroy_func(
            current_solution,
            instance,
            dist,
            rng,
            remove_count,
        )

        if not removed_customers:
            update_weight(destroy_weights, destroy_name, 0.0, reaction_factor)
            update_weight(repair_weights, repair_name, 0.0, reaction_factor)
            history.append(best_cost)
            continue

        partial_solution = remove_customers_from_solution(
            current_solution,
            removed_customers,
        )

        candidate_solution = repair_func(
            partial_solution,
            removed_customers,
            instance,
            dist,
            rng,
        )

        if candidate_solution is None:
            update_weight(destroy_weights, destroy_name, 0.0, reaction_factor)
            update_weight(repair_weights, repair_name, 0.0, reaction_factor)
            history.append(best_cost)
            temperature *= cooling_rate
            continue

        if candidate_solution is not None:
            candidate_solution = improve_solution_routes_2opt(candidate_solution, dist)


        feasible, _ = is_solution_feasible(candidate_solution, instance)

        if not feasible:
            update_weight(destroy_weights, destroy_name, 0.0, reaction_factor)
            update_weight(repair_weights, repair_name, 0.0, reaction_factor)
            history.append(best_cost)
            temperature *= cooling_rate
            continue

        candidate_cost = total_cost(candidate_solution, dist)
        delta = candidate_cost - current_cost

        accepted = False
        reward = 0.0

        if candidate_cost < best_cost:
            accepted = True
            reward = 5.0
            global_best_updates += 1

        elif candidate_cost < current_cost:
            accepted = True
            reward = 2.0
            improved_moves += 1

        else:
            probability = math.exp(-delta / max(temperature, 1e-12))

            if rng.random() < probability:
                accepted = True
                reward = 1.0

        if accepted:
            current_solution = candidate_solution
            current_cost = candidate_cost
            accepted_moves += 1

            if current_cost < best_cost:
                best_solution = copy_solution(current_solution)
                best_cost = current_cost

        update_weight(destroy_weights, destroy_name, reward, reaction_factor)
        update_weight(repair_weights, repair_name, reward, reaction_factor)

        temperature *= cooling_rate
        history.append(best_cost)

    elapsed_time = time.time() - start_time

    return ALNSResult(
        best_solution=best_solution,
        best_cost=best_cost,
        initial_cost=initial_cost,
        elapsed_time=elapsed_time,
        iterations=max_iterations,
        accepted_moves=accepted_moves,
        improved_moves=improved_moves,
        global_best_updates=global_best_updates,
        destroy_weights=destroy_weights,
        repair_weights=repair_weights,
        destroy_counts=destroy_counts,
        repair_counts=repair_counts,
        history=history,
    )