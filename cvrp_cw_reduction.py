from itertools import combinations
from typing import List, Optional, Tuple

from cvrp_utils import (
    CVRPInstance,
    build_distance_matrix,
    route_cost,
    route_demand,
    total_cost,
    is_solution_feasible,
    order_route_nearest_neighbor,
    two_opt_improve_route,
    improve_solution_routes_2opt,
)


def build_route_from_customers(
    customers: List[int],
    dist: List[List[float]],
) -> List[int]:
    if not customers:
        return [0, 0]

    route = order_route_nearest_neighbor(customers, dist)
    route = two_opt_improve_route(route, dist)
    return route


def clarke_wright_raw_routes(instance: CVRPInstance) -> List[List[int]]:
    """
    Clarke-Wright Savings heuristic, but it does NOT fail if it produces
    more routes than vehicles.

    This is useful for X-n101-k25 because normal Clarke-Wright gives good
    geographic routes but produces 28 routes instead of 25.
    """
    dist = build_distance_matrix(instance)

    customers = [node for node in instance.coordinates.keys() if node != 0]

    routes = {
        customer: [0, customer, 0]
        for customer in customers
    }

    loads = {
        customer: instance.demands[customer]
        for customer in customers
    }

    customer_to_route = {
        customer: customer
        for customer in customers
    }

    savings = []

    for i in customers:
        for j in customers:
            if i >= j:
                continue

            saving = dist[0][i] + dist[0][j] - dist[i][j]
            savings.append((saving, i, j))

    savings.sort(reverse=True)

    for saving, i, j in savings:
        route_i_id = customer_to_route[i]
        route_j_id = customer_to_route[j]

        if route_i_id == route_j_id:
            continue

        route_i = routes[route_i_id]
        route_j = routes[route_j_id]

        combined_load = loads[route_i_id] + loads[route_j_id]

        if combined_load > instance.capacity:
            continue

        new_route = None

        if route_i[-2] == i and route_j[1] == j:
            new_route = route_i[:-1] + route_j[1:]

        elif route_j[-2] == j and route_i[1] == i:
            new_route = route_j[:-1] + route_i[1:]

        elif route_i[1] == i and route_j[1] == j:
            reversed_i = [0] + list(reversed(route_i[1:-1])) + [0]
            new_route = reversed_i[:-1] + route_j[1:]

        elif route_i[-2] == i and route_j[-2] == j:
            reversed_j = [0] + list(reversed(route_j[1:-1])) + [0]
            new_route = route_i[:-1] + reversed_j[1:]

        if new_route is None:
            continue

        routes[route_i_id] = new_route
        loads[route_i_id] = combined_load

        del routes[route_j_id]
        del loads[route_j_id]

        for customer in new_route:
            if customer != 0:
                customer_to_route[customer] = route_i_id

    final_routes = list(routes.values())
    final_routes = improve_solution_routes_2opt(final_routes, dist)

    return final_routes


def best_pair_merge(
    routes: List[List[int]],
    instance: CVRPInstance,
    dist: List[List[float]],
) -> Optional[Tuple[int, int, List[int], float]]:
    """
    Find the best feasible merge of two routes into one route.
    """
    best = None

    for i, j in combinations(range(len(routes)), 2):
        route_i = routes[i]
        route_j = routes[j]

        customers_i = [node for node in route_i if node != 0]
        customers_j = [node for node in route_j if node != 0]

        combined_customers = customers_i + customers_j
        combined_demand = sum(instance.demands[c] for c in combined_customers)

        if combined_demand > instance.capacity:
            continue

        old_cost = route_cost(route_i, dist) + route_cost(route_j, dist)

        new_route = build_route_from_customers(combined_customers, dist)
        new_cost = route_cost(new_route, dist)

        delta = new_cost - old_cost

        if best is None or delta < best[3]:
            best = (i, j, new_route, delta)

    return best


def best_three_to_two_reduction(
    routes: List[List[int]],
    instance: CVRPInstance,
    dist: List[List[float]],
    max_customers: int = 16,
) -> Optional[Tuple[int, int, int, List[int], List[int], float]]:
    """
    Try replacing 3 routes with 2 feasible routes.

    This is useful when no direct pair merge is possible.
    """
    best = None

    for i, j, k in combinations(range(len(routes)), 3):
        route_i = routes[i]
        route_j = routes[j]
        route_k = routes[k]

        customers = (
            [node for node in route_i if node != 0]
            + [node for node in route_j if node != 0]
            + [node for node in route_k if node != 0]
        )

        if len(customers) > max_customers:
            continue

        total_demand = sum(instance.demands[c] for c in customers)

        if total_demand > 2 * instance.capacity:
            continue

        old_cost = (
            route_cost(route_i, dist)
            + route_cost(route_j, dist)
            + route_cost(route_k, dist)
        )

        first_customer = customers[0]
        remaining = customers[1:]

        for mask in range(1 << len(remaining)):
            group_a = [first_customer]
            group_b = []

            for bit_index, customer in enumerate(remaining):
                if mask & (1 << bit_index):
                    group_a.append(customer)
                else:
                    group_b.append(customer)

            if not group_b:
                continue

            demand_a = sum(instance.demands[c] for c in group_a)
            demand_b = sum(instance.demands[c] for c in group_b)

            if demand_a > instance.capacity or demand_b > instance.capacity:
                continue

            new_route_a = build_route_from_customers(group_a, dist)
            new_route_b = build_route_from_customers(group_b, dist)

            new_cost = route_cost(new_route_a, dist) + route_cost(new_route_b, dist)
            delta = new_cost - old_cost

            if best is None or delta < best[5]:
                best = (i, j, k, new_route_a, new_route_b, delta)

    return best


def reduce_routes_to_vehicle_count(
    routes: List[List[int]],
    instance: CVRPInstance,
) -> List[List[int]]:
    """
    Reduce Clarke-Wright routes until the number of routes equals vehicle_count.
    """
    dist = build_distance_matrix(instance)
    routes = [route for route in routes if route != [0, 0]]

    while len(routes) > instance.vehicle_count:
        pair_merge = best_pair_merge(routes, instance, dist)

        if pair_merge is not None:
            i, j, new_route, delta = pair_merge

            new_routes = []
            for index, route in enumerate(routes):
                if index not in {i, j}:
                    new_routes.append(route)

            new_routes.append(new_route)
            routes = improve_solution_routes_2opt(new_routes, dist)
            continue

        three_to_two = best_three_to_two_reduction(routes, instance, dist)

        if three_to_two is not None:
            i, j, k, new_route_a, new_route_b, delta = three_to_two

            new_routes = []
            for index, route in enumerate(routes):
                if index not in {i, j, k}:
                    new_routes.append(route)

            new_routes.append(new_route_a)
            new_routes.append(new_route_b)

            routes = improve_solution_routes_2opt(new_routes, dist)
            continue

        raise ValueError(
            f"Could not reduce routes further. Current route count={len(routes)}"
        )

    while len(routes) < instance.vehicle_count:
        routes.append([0, 0])

    feasible, errors = is_solution_feasible(routes, instance)

    if not feasible:
        raise ValueError(f"Reduced Clarke-Wright solution infeasible: {errors}")

    return routes


def clarke_wright_reduction_candidates(
    instance: CVRPInstance,
) -> List[Tuple[str, List[List[int]], float]]:
    """
    Generate a candidate by taking overfull Clarke-Wright and reducing
    the route count.
    """
    dist = build_distance_matrix(instance)

    raw_routes = clarke_wright_raw_routes(instance)

    print(
        f"Raw Clarke-Wright route count before reduction: {len(raw_routes)}"
    )

    reduced_solution = reduce_routes_to_vehicle_count(raw_routes, instance)

    cost = total_cost(reduced_solution, dist)

    return [("clarke_wright_reduced", reduced_solution, cost)]