import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os
import re
import matplotlib.pyplot as plt


@dataclass
class CVRPInstance:
    """
    Stores one CVRP problem instance.

    We use normalized indices:
    - depot is always 0
    - customers are 1, 2, ..., n-1

    coordinates[i] = (x, y)
    demands[i] = demand of node i
    """
    name: str
    capacity: int
    vehicle_count: int
    coordinates: Dict[int, Tuple[float, float]]
    demands: Dict[int, int]
    depot: int = 0


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Return Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def build_distance_matrix(instance: CVRPInstance) -> List[List[float]]:
    """
    Build full distance matrix.

    dist[i][j] = Euclidean distance between node i and node j.
    """
    n = len(instance.coordinates)
    dist = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            dist[i][j] = euclidean_distance(
                instance.coordinates[i],
                instance.coordinates[j]
            )

    return dist


def route_cost(route: List[int], dist: List[List[float]]) -> float:
    """
    Calculate cost of one route.

    Example route:
    [0, 1, 2, 3, 0]
    """
    if len(route) < 2:
        return 0.0

    cost = 0.0
    for i in range(len(route) - 1):
        cost += dist[route[i]][route[i + 1]]

    return cost


def total_cost(solution: List[List[int]], dist: List[List[float]]) -> float:
    """
    Calculate total cost of all vehicle routes.
    """
    return sum(route_cost(route, dist) for route in solution)


def route_demand(route: List[int], demands: Dict[int, int]) -> int:
    """
    Calculate total demand of customers in a route.

    Depot 0 has demand 0 and is ignored.
    """
    return sum(demands[node] for node in route if node != 0)


def is_route_feasible(
    route: List[int],
    demands: Dict[int, int],
    capacity: int
) -> bool:
    """
    Check if one route starts/ends at depot and respects capacity.
    """
    if len(route) < 2:
        return False

    if route[0] != 0 or route[-1] != 0:
        return False

    return route_demand(route, demands) <= capacity


def is_solution_feasible(
    solution: List[List[int]],
    instance: CVRPInstance
) -> Tuple[bool, List[str]]:
    """
    Check full CVRP feasibility.

    Conditions:
    1. Every route starts and ends at 0.
    2. Every route demand <= vehicle capacity.
    3. Every customer appears exactly once.
    4. No missing customers.
    5. No duplicate customers.
    """
    errors = []

    if len(solution) > instance.vehicle_count:
        errors.append(
            f"Solution uses {len(solution)} routes, but only "
            f"{instance.vehicle_count} vehicles are available."
        )

    visited_customers = []

    for route_index, route in enumerate(solution):
        if not is_route_feasible(route, instance.demands, instance.capacity):
            errors.append(
                f"Route {route_index} is infeasible. "
                f"Route={route}, demand={route_demand(route, instance.demands)}"
            )

        for node in route:
            if node != 0:
                visited_customers.append(node)

    expected_customers = set(instance.coordinates.keys()) - {0}
    visited_set = set(visited_customers)

    missing = expected_customers - visited_set
    duplicates = {
        customer
        for customer in visited_customers
        if visited_customers.count(customer) > 1
    }
    unknown = visited_set - expected_customers

    if missing:
        errors.append(f"Missing customers: {sorted(missing)}")

    if duplicates:
        errors.append(f"Duplicated customers: {sorted(duplicates)}")

    if unknown:
        errors.append(f"Unknown customers: {sorted(unknown)}")

    return len(errors) == 0, errors

def greedy_initial_solution(instance: CVRPInstance) -> List[List[int]]:
    """
    Main greedy initial solution.

    1. First try nearest-neighbor greedy because it gives nicer routes.
    2. If that fails, use DP capacity-first construction.
    3. If that fails, use bin-packing fallback.
    """
    try:
        return greedy_nearest_initial_solution(instance)
    except ValueError:
        pass

    try:
        return greedy_capacity_dp_initial_solution(instance)
    except ValueError:
        pass

    return greedy_bin_packing_initial_solution(instance)

def greedy_nearest_initial_solution(instance: CVRPInstance) -> List[List[int]]:
    """
    Simple greedy initial CVRP solution.

    Strategy:
    - Start each vehicle at depot.
    - Repeatedly choose the nearest unvisited customer that still fits.
    - Return to depot.
    - Continue with next vehicle.

    This is not guaranteed to be optimal.
    It is only a legal starting point for SA, TS, and ALNS.
    """
    dist = build_distance_matrix(instance)

    unvisited = set(instance.coordinates.keys()) - {0}
    solution = []

    for _ in range(instance.vehicle_count):
        route = [0]
        current_node = 0
        remaining_capacity = instance.capacity

        while True:
            feasible_customers = [
                customer for customer in unvisited
                if instance.demands[customer] <= remaining_capacity
            ]

            if not feasible_customers:
                break

            next_customer = min(
                feasible_customers,
                key=lambda customer: dist[current_node][customer]
            )

            route.append(next_customer)
            unvisited.remove(next_customer)
            remaining_capacity -= instance.demands[next_customer]
            current_node = next_customer

        route.append(0)
        solution.append(route)

        if not unvisited:
            break

    # Add unused vehicles as [0, 0], matching the assignment example format.
    while len(solution) < instance.vehicle_count:
        solution.append([0, 0])

    if unvisited:
        raise ValueError(
            "Greedy solution failed: not enough vehicle capacity "
            f"to serve all customers. Unvisited: {sorted(unvisited)}"
        )

    return solution

def greedy_capacity_dp_initial_solution(instance: CVRPInstance) -> List[List[int]]:
    """
    Capacity-first initial solution using repeated subset-sum DP.

    This is more robust for tight-capacity instances like X-n101-k25.

    Idea:
    For each vehicle, choose a subset of remaining customers whose total demand
    is as close as possible to the needed load, while still leaving enough
    total capacity for the remaining vehicles.

    Then order each vehicle route using nearest-neighbor.
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

        # This vehicle must carry at least this much, otherwise the remaining
        # vehicles cannot carry the remaining total demand.
        min_required_load = max(
            0,
            remaining_demand - vehicles_left_after_this * instance.capacity
        )

        max_allowed_load = min(instance.capacity, remaining_demand)

        customers_list = list(remaining_customers)

        # Subset-sum DP up to capacity.
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
                "DP capacity initializer failed. "
                f"Vehicle={vehicle_index}, "
                f"remaining demand={remaining_demand}, "
                f"min required load={min_required_load}, "
                f"max allowed load={max_allowed_load}, "
                f"remaining customers={len(remaining_customers)}"
            )

        # Prefer the largest valid load, because tight CVRP instances need
        # almost full trucks.
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
            "DP capacity initializer failed: customers still unassigned: "
            f"{sorted(remaining_customers)}"
        )

    solution = [
        order_route_nearest_neighbor(customers, dist)
        for customers in vehicle_customer_groups
    ]

    while len(solution) < instance.vehicle_count:
        solution.append([0, 0])

    return solution

def order_route_nearest_neighbor(
    customers: List[int],
    dist: List[List[float]]
) -> List[int]:
    """
    Given a set/list of customers assigned to one vehicle,
    order them using nearest-neighbor routing.

    Returns a full route:
    [0, ..., 0]
    """
    if not customers:
        return [0, 0]

    unvisited = set(customers)
    route = [0]
    current = 0

    while unvisited:
        next_customer = min(
            unvisited,
            key=lambda customer: dist[current][customer]
        )
        route.append(next_customer)
        unvisited.remove(next_customer)
        current = next_customer

    route.append(0)
    return route

def greedy_bin_packing_initial_solution(instance: CVRPInstance) -> List[List[int]]:
    """
    Robust greedy initial solution.

    Phase 1:
    Assign customers to vehicles using First-Fit Decreasing by demand.
    This focuses on capacity feasibility.

    Phase 2:
    For each vehicle, order the assigned customers using nearest neighbor.
    This focuses on route distance.

    This is usually more robust than nearest-neighbor greedy when total capacity
    is very tight, like X-n101-k25.
    """
    dist = build_distance_matrix(instance)

    customers = sorted(
        [node for node in instance.coordinates.keys() if node != 0],
        key=lambda node: instance.demands[node],
        reverse=True
    )

    vehicle_customers = [[] for _ in range(instance.vehicle_count)]
    remaining_capacity = [instance.capacity for _ in range(instance.vehicle_count)]

    for customer in customers:
        demand = instance.demands[customer]

        best_vehicle = None
        best_remaining_after = None

        for vehicle_index in range(instance.vehicle_count):
            if remaining_capacity[vehicle_index] >= demand:
                remaining_after = remaining_capacity[vehicle_index] - demand

                # Best-fit: choose the vehicle that will have the least remaining capacity
                # after inserting this customer.
                if best_remaining_after is None or remaining_after < best_remaining_after:
                    best_remaining_after = remaining_after
                    best_vehicle = vehicle_index

        if best_vehicle is None:
            raise ValueError(
                "Bin-packing greedy failed: customer cannot fit in any vehicle. "
                f"Customer={customer}, demand={demand}, "
                f"remaining capacities={remaining_capacity}"
            )

        vehicle_customers[best_vehicle].append(customer)
        remaining_capacity[best_vehicle] -= demand

    solution = [
        order_route_nearest_neighbor(customers_for_vehicle, dist)
        for customers_for_vehicle in vehicle_customers
    ]

    return solution


def print_solution(
    solution: List[List[int]],
    dist: List[List[float]],
    elapsed_time: float = 0.0
) -> None:
    """
    Print solution in a format similar to the assignment example.

    First line:
    total_cost elapsed_time

    Then one route per vehicle.
    """
    cost = total_cost(solution, dist)
    print(f"{cost:.2f} {elapsed_time:.4f}")

    for route in solution:
        print(" ".join(map(str, route)))


def create_assignment_example_instance() -> CVRPInstance:
    """
    Create the 4-customer example from the assignment PDF.

    Depot:
    0 -> (0, 0)

    Customers:
    1 -> (0, 10), demand 3
    2 -> (-10, 10), demand 3
    3 -> (0, -10), demand 3
    4 -> (10, -10), demand 3

    vehicles = 4
    capacity = 10
    """
    coordinates = {
        0: (0.0, 0.0),
        1: (0.0, 10.0),
        2: (-10.0, 10.0),
        3: (0.0, -10.0),
        4: (10.0, -10.0),
    }

    demands = {
        0: 0,
        1: 3,
        2: 3,
        3: 3,
        4: 3,
    }

    return CVRPInstance(
        name="assignment_example",
        capacity=10,
        vehicle_count=4,
        coordinates=coordinates,
        demands=demands,
        depot=0
    )


def read_vrplib_cvrp(path: str, vehicle_count: Optional[int] = None) -> CVRPInstance:
    """
    Basic parser for common CVRPLIB-style files.

    Many benchmark files use:
    - NODE_COORD_SECTION
    - DEMAND_SECTION
    - DEPOT_SECTION

    Usually depot is node 1 in the file.
    We normalize it to node 0 internally.

    If vehicle_count is not written in the file name or file body,
    pass it manually, for example:
        read_vrplib_cvrp("P-n16-k8.vrp", vehicle_count=8)
    """
    name = "unknown"
    capacity = None
    coordinates_original = {}
    demands_original = {}
    depot_original = None

    section = None

    with open(path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            upper = line.upper()

            if upper.startswith("NAME"):
                name = line.split(":")[-1].strip()

            elif upper.startswith("CAPACITY"):
                capacity = int(line.split(":")[-1].strip())

            elif upper.startswith("NODE_COORD_SECTION"):
                section = "coords"

            elif upper.startswith("DEMAND_SECTION"):
                section = "demands"

            elif upper.startswith("DEPOT_SECTION"):
                section = "depot"

            elif upper.startswith("EOF"):
                break

            else:
                parts = line.split()

                if section == "coords" and len(parts) >= 3:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coordinates_original[node_id] = (x, y)

                elif section == "demands" and len(parts) >= 2:
                    node_id = int(parts[0])
                    demand = int(float(parts[1]))
                    demands_original[node_id] = demand

                elif section == "depot":
                    node_id = int(parts[0])
                    if node_id != -1:
                        depot_original = node_id

    if capacity is None:
        raise ValueError("Could not find CAPACITY in file.")

    if depot_original is None:
        # Many files use 1 as depot if not clearly parsed.
        depot_original = 1

    if vehicle_count is None:
        # Try to extract k from NAME or filename, e.g. P-n16-k8
        text_to_search = name + " " + os.path.basename(path)

        match = re.search(r"-k(\d+)", text_to_search, re.IGNORECASE)

        if match:
            vehicle_count = int(match.group(1))

    if vehicle_count is None:
        raise ValueError(
            "vehicle_count was not found automatically. "
            "Pass it manually, e.g. vehicle_count=8."
        )

    # Normalize IDs:
    # depot_original -> 0
    # all other nodes -> 1, 2, 3, ...
    old_to_new = {depot_original: 0}
    next_id = 1

    for old_id in sorted(coordinates_original.keys()):
        if old_id == depot_original:
            continue
        old_to_new[old_id] = next_id
        next_id += 1

    coordinates = {}
    demands = {}

    for old_id, new_id in old_to_new.items():
        coordinates[new_id] = coordinates_original[old_id]
        demands[new_id] = demands_original.get(old_id, 0)

    return CVRPInstance(
        name=name,
        capacity=capacity,
        vehicle_count=vehicle_count,
        coordinates=coordinates,
        demands=demands,
        depot=0
    )

def read_reference_cost(sol_path: str) -> Optional[float]:
    """
    Read the known/best reference cost from a .sol file.

    Common CVRPLIB .sol files contain a line like:
    Cost 450

    We only read the cost for now.
    We do not parse the routes yet because route node IDs may use the original
    CVRPLIB numbering, while our internal representation normalizes depot to 0.
    """
    try:
        with open(sol_path, "r", encoding="utf-8") as file:
            for line in file:
                clean = line.strip()

                if not clean:
                    continue

                lower = clean.lower()

                if lower.startswith("cost"):
                    parts = clean.replace(":", " ").split()

                    for part in parts:
                        try:
                            return float(part)
                        except ValueError:
                            continue

    except FileNotFoundError:
        return None

    return None


def write_solution_file(
    solution: List[List[int]],
    dist: List[List[float]],
    output_path: str,
    elapsed_time: float = 0.0
) -> None:
    """
    Save solution in the assignment-style output format.

    First line:
    total_cost elapsed_time

    Then one line per vehicle route.
    """
    cost = total_cost(solution, dist)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(f"{cost:.2f} {elapsed_time:.4f}\n")

        for route in solution:
            file.write(" ".join(map(str, route)) + "\n")




def plot_solution(
    instance: CVRPInstance,
    solution: List[List[int]],
    dist: List[List[float]],
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot a CVRP solution.

    - Depot is marked separately.
    - Each route is drawn as a connected path.
    - Customer indices are shown on the plot.

    If save_path is given, saves the figure.
    If show=True, displays the figure.
    """


    cost = total_cost(solution, dist)

    plt.figure(figsize=(8, 6))

    # Plot all customers and depot labels
    for node, (x, y) in instance.coordinates.items():
        if node == 0:
            plt.scatter(x, y, marker="s", s=120, label="Depot")
            plt.text(x, y, f" {node}", fontsize=10, fontweight="bold")
        else:
            plt.scatter(x, y, s=40)
            plt.text(x, y, f" {node}", fontsize=8)

    # Plot routes
    for route_index, route in enumerate(solution):
        if route == [0, 0]:
            continue

        xs = [instance.coordinates[node][0] for node in route]
        ys = [instance.coordinates[node][1] for node in route]

        plt.plot(xs, ys, marker="o", linewidth=1, label=f"Vehicle {route_index}")

    plt.title(f"{instance.name} | cost = {cost:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis("equal")
    plt.legend(fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()

    plt.close()