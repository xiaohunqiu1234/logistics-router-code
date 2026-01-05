import heapq
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Set, Optional
import random
import math
import itertools
import csv
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.patches as mpatches
import time  # Imported for timing
from dateutil import parser

from scipy.optimize import differential_evolution
import matplotlib


matplotlib.use('Agg')  # Use non-interactive backend for server environments

class LogisticsRouter:
    """物流路径规划核心引擎"""

    def __init__(self):
        self.graph = {}  # 路网图
        self.nodes = set()  # 节点集合
        self.node_coordinates = {}  # Added to store coordinates for networkx plotting

    def add_edge(self, from_node: str, to_node: str,
                 distance: float, time: float, cost: float):
        """添加道路边（支持多维权重）"""
        if from_node not in self.graph:
            self.graph[from_node] = {}  # Changed to dict for easier neighbor access by ID
        self.graph[from_node][to_node] = {  # Changed to dict for easier neighbor access by ID
            'distance': distance,
            'time': time,
            'cost': cost
        }
        self.nodes.update([from_node, to_node])

    def add_node_coordinates(self, node: str, lat: float, lon: float):
        """Add node coordinates for visualization purposes."""
        self.node_coordinates[node] = (lat, lon)
        self.nodes.add(node)  # Ensure node is added

    def dijkstra_shortest_path(self, start: str, end: str,
                               weight_type: str = 'distance') -> Tuple[List[str], float]:
        """
        Dijkstra最短路径算法

        Args:
            start: 起点
            end: 终点
            weight_type: 权重类型 ('distance', 'time', 'cost')

        Returns:
            路径列表, 总权重
        """
        # Initialize distances with infinity, 0 for start node
        distances = {node: float('inf') for node in self.nodes}
        distances[start] = 0

        # Keep track of the previous node in the shortest path
        previous = {node: None for node in self.nodes}

        # Priority queue stores (distance, node)
        pq = [(0, start)]
        visited = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current in visited:
                continue
            visited.add(current)

            if current == end:
                break

            # Explore neighbors
            if current in self.graph:
                for neighbor, edge_data in self.graph[current].items():
                    weight = edge_data[weight_type]
                    distance = current_dist + weight

                    # If a shorter path to neighbor is found
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current
                        heapq.heappush(pq, (distance, neighbor))

        # Reconstruct the path from end to start
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()

        # Return path and total distance if path exists, otherwise empty list and infinity
        if path[0] == start:
            return path, distances[end]
        else:
            return [], float('inf')


class AStarRouter(LogisticsRouter):
    """A*算法路径规划（启发式搜索）"""

    def __init__(self):
        super().__init__()
        # self.coordinates is already inherited from LogisticsRouter but is public.
        # Using self.node_coordinates which is also public and more descriptive.

    def add_node_coordinates(self, node: str, lat: float, lon: float):
        """添加节点GPS坐标"""
        super().add_node_coordinates(node, lat, lon)  # Ensure node is added to self.nodes too

    def _get_node_id(self, location: Tuple[float, float]) -> Optional[str]:
        """根据坐标查找节点ID"""
        # This method is more for internal use if you need to map coordinates to node IDs.
        # For route calculation, we usually pass node IDs directly.
        for node_id, coords in self.node_coordinates.items():
            if coords == location:
                return node_id
        return None

    def haversine_distance(self, node1: str, node2: str) -> float:
        """计算两点间球面距离（启发函数）"""
        from math import radians, sin, cos, sqrt, atan2

        if node1 not in self.node_coordinates or node2 not in self.node_coordinates:
            # Fallback for cases where coordinates might be missing
            # This could happen if a node is in graph but not in coordinates
            # In a real scenario, you'd want to handle this more robustly
            # print(f"Warning: Coordinates missing for {node1} or {node2}. Using Euclidean distance.")
            if node1 in self.node_coordinates and node2 in self.node_coordinates:
                lat1, lon1 = self.node_coordinates[node1]
                lat2, lon2 = self.node_coordinates[node2]
                # Approximate conversion to km, assuming 1 degree latitude/longitude is ~111km
                return np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) * 111
            else:
                return 0.0  # Cannot calculate, return 0 or raise error

        lat1, lon1 = self.node_coordinates[node1]
        lat2, lon2 = self.node_coordinates[node2]

        R = 6371  # Earth radius(km)
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)

        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c

    def _calculate_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """Helper method to calculate distance between two coordinates (used by GA for its internal calculations)."""
        from math import radians, sin, cos, sqrt, atan2
        lat1, lon1 = loc1
        lat2, lon2 = loc2

        R = 6371  # Earth radius in kilometers
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)

        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c

    def a_star_search(self, start: str, end: str,
                      weight_type: str = 'distance') -> Tuple[List[str], float]:
        """
        A*算法（结合启发函数的最优搜索）

        f(n) = g(n) + h(n)
        g(n): 起点到n的实际代价
        h(n): n到终点的启发式估计
        """
        if start not in self.nodes or end not in self.nodes:
            print(f"Error: Start node '{start}' or end node '{end}' not in the graph.")
            return [], float('inf')

        # g值：起点到当前节点的实际代价
        g_score = {node: float('inf') for node in self.nodes}
        g_score[start] = 0

        # f值：g + h
        f_score = {node: float('inf') for node in self.nodes}
        # Ensure start and end nodes have coordinates for heuristic calculation
        if start in self.node_coordinates and end in self.node_coordinates:
            f_score[start] = self.haversine_distance(start, end)
        else:
            print(
                f"Warning: Coordinates missing for start ({start}) or end ({end}) node. Heuristic might be inaccurate.")
            # Fallback if coordinates are missing, maybe use a default heuristic or 0
            f_score[start] = 0

        # 优先队列
        open_set = [(f_score[start], start)]
        came_from = {}
        closed_set = set()

        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            if current == end:
                # 重构路径
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path, g_score[end]

            closed_set.add(current)

            # 探索邻居
            if current in self.graph:
                for neighbor, edge_data in self.graph[current].items():
                    if neighbor in closed_set:
                        continue

                    weight = edge_data[weight_type]
                    tentative_g = g_score[current] + weight

                    if tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        # Calculate heuristic for neighbor if coordinates are available
                        if neighbor in self.node_coordinates and end in self.node_coordinates:
                            f_score[neighbor] = tentative_g + self.haversine_distance(neighbor, end)
                        else:
                            # Fallback if coordinates are missing for neighbor or end node
                            f_score[neighbor] = tentative_g  # Use g_score as fallback for f_score

                        # Check if neighbor is already in open_set with a higher f_score
                        # This is a simplified approach, a priority queue implementation might
                        # handle updates more efficiently. For this example, we just push.
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return [], float('inf')


class MultiSourceRouter:
    """多源最短路径规划"""

    def floyd_warshall(self, graph_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Floyd-Warshall算法 - 适用于配送中心网络优化

        Args:
            graph_matrix: 邻接矩阵 (n x n)

        Returns:
            距离矩阵, 路径矩阵
        """
        n = len(graph_matrix)
        dist = graph_matrix.copy()
        next_node = np.full((n, n), -1, dtype=int)

        # Initialize path matrix
        for i in range(n):
            for j in range(n):
                if graph_matrix[i][j] != float('inf') and i != j:
                    next_node[i][j] = j

        # Dynamic programming
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_node[i][j] = next_node[i][k]

        return dist, next_node

    def reconstruct_path(self, next_node: np.ndarray, start: int, end: int) -> List[int]:
        """根据路径矩阵重构路径"""
        if next_node[start][end] == -1:
            return []

        path = [start]
        while start != end:
            start = next_node[start][end]
            path.append(start)

        return path


@dataclass
class DeliveryPoint:
    """配送点"""
    id: str
    location: Tuple[float, float]  # (lat, lon)
    demand: float  # 需求量
    time_window: Tuple[datetime, datetime]  # 时间窗
    service_time: int  # 服务时长(分钟)
    priority: int = 1  # 优先级

    def __hash__(self):
        """使对象可哈希，用于set和dict"""
        return hash(self.id)

    def __eq__(self, other):
        """对象相等性比较"""
        if not isinstance(other, DeliveryPoint):
            return False
        return self.id == other.id


@dataclass
class Vehicle:
    """车辆"""
    id: str
    capacity: float  # 载重
    start_location: Tuple[float, float]
    max_distance: float  # 最大行驶距离
    speed: float = 60  # 平均速度 km/h
    start_location_id: Optional[str] = None  # Added to store start location ID


class VRPTWSolver:
    """带时间窗的车辆路径问题求解器"""

    def __init__(self, vehicles: List[Vehicle], points: List[DeliveryPoint]):
        self.vehicles = vehicles
        self.points = points
        self.distance_matrix = self._build_distance_matrix()

    def _build_distance_matrix(self) -> np.ndarray:
        """构建距离矩阵"""
        n = len(self.points)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = self._calculate_distance(
                        self.points[i].location,
                        self.points[j].location
                    )
        return matrix

    def _calculate_distance(self, loc1: Tuple[float, float],
                            loc2: Tuple[float, float]) -> float:
        """计算两点距离"""
        from math import radians, sin, cos, sqrt, atan2
        lat1, lon1 = loc1
        lat2, lon2 = loc2

        R = 6371
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def nearest_neighbor_heuristic(self, vehicle: Vehicle) -> List[DeliveryPoint]:
        """最近邻启发式算法"""

        # ===== 预检查阶段 =====
        print("\n🔍 最近邻算法诊断信息:")
        print(f"  配送点总数: {len(self.points)}")
        print(f"  车辆容量: {vehicle.capacity} kg")
        print(f"  车辆最大距离: {vehicle.max_distance} km")
        print(f"  车辆速度: {vehicle.speed} km/h")
        print(f"  起始位置: {vehicle.start_location}")

        if not self.points:
            print("❌ 错误：配送点列表为空！")
            return []

        # 检查配送点需求
        total_demand = sum(p.demand for p in self.points)
        print(f"  总需求量: {total_demand} kg")

        if total_demand > vehicle.capacity:
            print(f"⚠️  警告：总需求 ({total_demand}kg) 超过车辆容量 ({vehicle.capacity}kg)")
            print("      算法将尽可能配送，但无法完成所有点")

        # 检查各配送点到起点的距离
        print("\n  各配送点初始状态:")
        for i, point in enumerate(self.points, 1):
            dist = self._calculate_distance(vehicle.start_location, point.location)
            print(f"    {i}. {point.id}:")
            print(f"       - 需求: {point.demand}kg")
            print(f"       - 距离起点: {dist:.2f}km")
            print(
                f"       - 时间窗: {point.time_window[0].strftime('%H:%M')} - {point.time_window[1].strftime('%H:%M')}")
            print(f"       - 服务时间: {point.service_time}分钟")

        # ===== 执行最近邻算法 =====
        route = []
        unvisited = self.points.copy()
        current_load = 0
        current_location = vehicle.start_location
        current_time = datetime.now()
        total_distance = 0

        iteration = 0
        rejection_reasons = {
            'capacity': 0,
            'distance': 0,
            'time_window': 0
        }

        print("\n🚚 开始路径规划...")

        while unvisited:
            iteration += 1
            print(f"\n  --- 迭代 {iteration} ---")
            print(f"  当前位置: {current_location}")
            print(f"  当前载重: {current_load}/{vehicle.capacity} kg")
            print(f"  已行驶: {total_distance:.2f}/{vehicle.max_distance} km")
            print(f"  当前时间: {current_time.strftime('%H:%M')}")
            print(f"  剩余点数: {len(unvisited)}")

            # 找到最近的可行点
            best_point = None
            best_distance = float('inf')
            candidates_checked = 0

            for point in unvisited:
                candidates_checked += 1
                distance = self._calculate_distance(current_location, point.location)

                # 检查约束条件
                feasible = True
                rejection_reason = None

                # 1. 容量约束
                if current_load + point.demand > vehicle.capacity:
                    feasible = False
                    rejection_reason = 'capacity'
                    rejection_reasons['capacity'] += 1

                # 2. 距离约束
                elif total_distance + distance > vehicle.max_distance:
                    feasible = False
                    rejection_reason = 'distance'
                    rejection_reasons['distance'] += 1

                # 3. 时间窗约束
                else:
                    travel_time = timedelta(hours=distance / vehicle.speed)
                    arrival_time = current_time + travel_time

                    if arrival_time > point.time_window[1]:
                        feasible = False
                        rejection_reason = 'time_window'
                        rejection_reasons['time_window'] += 1

                # 记录候选点信息（仅在详细模式下）
                if not feasible and candidates_checked <= 3:  # 只显示前3个被拒绝的
                    print(f"    ✗ {point.id}: 距离{distance:.2f}km - 被拒绝({rejection_reason})")

                # 选择最近的可行点
                if feasible and distance < best_distance:
                    best_distance = distance
                    best_point = point

            # 如果没有找到可行点
            if best_point is None:
                print(f"\n  ⚠️  无法继续配送！原因统计:")
                print(f"     - 容量限制: {rejection_reasons['capacity']} 次")
                print(f"     - 距离限制: {rejection_reasons['distance']} 次")
                print(f"     - 时间窗限制: {rejection_reasons['time_window']} 次")
                print(f"  ✅ 已完成 {len(route)}/{len(self.points)} 个配送点")
                break

            # 添加到路径
            print(f"  ✓ 选择: {best_point.id} (距离 {best_distance:.2f}km)")
            route.append(best_point)
            unvisited.remove(best_point)
            current_load += best_point.demand
            current_location = best_point.location
            total_distance += best_distance

            # 更新时间
            travel_time = timedelta(hours=best_distance / vehicle.speed)
            arrival_time = current_time + travel_time
            service_start = max(arrival_time, best_point.time_window[0])
            current_time = service_start + timedelta(minutes=best_point.service_time)

            # 显示等待时间
            if arrival_time < best_point.time_window[0]:
                wait_time = (best_point.time_window[0] - arrival_time).total_seconds() / 60
                print(f"    (等待 {wait_time:.0f} 分钟)")

        # ===== 结果汇总 =====
        print("\n" + "=" * 50)
        if route:
            return_dist = self._calculate_distance(current_location, vehicle.start_location)
            print(f"✅ 最近邻算法完成!")
            print(f"  配送点数: {len(route)}/{len(self.points)}")
            print(f"  总距离: {total_distance:.2f}km (返程+{return_dist:.2f}km)")
            print(f"  最终载重: {current_load}/{vehicle.capacity}kg")
            print(f"  路径: {' → '.join([p.id for p in route])}")
        else:
            print("❌ 最近邻算法未能生成任何路径!")
            print("\n可能原因:")
            print("  1. 第一个配送点的需求就超过了车辆容量")
            print("  2. 到任何配送点的距离都超过了最大行驶距离")
            print("  3. 当前时间已经超过了所有配送点的时间窗")
            print("\n建议:")
            print("  • 增加车辆容量 (当前: {}kg)".format(vehicle.capacity))
            print("  • 增加最大行驶距离 (当前: {}km)".format(vehicle.max_distance))
            print("  • 调整配送点的时间窗设置")
            print("  • 减少配送点的需求量")
        print("=" * 50 + "\n")

        return route


def diagnose_vrp_constraints(delivery_points: List[DeliveryPoint], vehicle: Vehicle):
    """诊断VRP约束是否合理"""
    print("\n" + "=" * 60)
    print("📋 VRP配置诊断报告")
    print("=" * 60)

    # 1. 需求分析
    demands = [p.demand for p in delivery_points]
    total_demand = sum(demands)
    max_demand = max(demands) if demands else 0

    print(f"\n1️⃣  需求分析:")
    print(f"  总需求: {total_demand:.1f} kg")
    print(f"  平均需求: {total_demand / len(demands):.1f} kg" if demands else "  N/A")
    print(f"  最大单点需求: {max_demand:.1f} kg")
    print(f"  车辆容量: {vehicle.capacity:.1f} kg")

    if max_demand > vehicle.capacity:
        print(f"  ❌ 有配送点需求超过车辆容量！")
    elif total_demand > vehicle.capacity:
        print(f"  ⚠️  总需求超过容量，需要多趟配送")
    else:
        print(f"  ✅ 容量充足")

    # 2. 距离分析
    print(f"\n2️⃣  距离分析:")
    solver = VRPTWSolver([vehicle], delivery_points)

    distances_from_start = []
    for point in delivery_points:
        dist = solver._calculate_distance(vehicle.start_location, point.location)
        distances_from_start.append(dist)

    if distances_from_start:
        print(f"  最近点距离: {min(distances_from_start):.2f} km")
        print(f"  最远点距离: {max(distances_from_start):.2f} km")
        print(f"  平均距离: {sum(distances_from_start) / len(distances_from_start):.2f} km")
        print(f"  车辆最大距离: {vehicle.max_distance:.2f} km")

        if max(distances_from_start) * 2 > vehicle.max_distance:
            print(f"  ❌ 有配送点往返距离就超过最大行驶距离！")
        else:
            print(f"  ✅ 距离约束合理")

    # 3. 时间窗分析
    print(f"\n3️⃣  时间窗分析:")
    now = datetime.now()
    time_windows = [(p.time_window[0], p.time_window[1]) for p in delivery_points]

    earliest_start = min([tw[0] for tw in time_windows]) if time_windows else now
    latest_end = max([tw[1] for tw in time_windows]) if time_windows else now

    print(f"  当前时间: {now.strftime('%H:%M')}")
    print(f"  最早时间窗: {earliest_start.strftime('%H:%M')}")
    print(f"  最晚时间窗: {latest_end.strftime('%H:%M')}")

    expired_count = sum(1 for tw in time_windows if now > tw[1])
    if expired_count > 0:
        print(f"  ❌ {expired_count} 个配送点的时间窗已过期！")
    else:
        print(f"  ✅ 时间窗设置合理")

    # 4. 给出建议
    print(f"\n4️⃣  优化建议:")
    suggestions = []

    if max_demand > vehicle.capacity:
        suggestions.append(f"  • 将车辆容量提升至至少 {max_demand * 1.2:.0f} kg")

    if total_demand > vehicle.capacity * 1.5:
        suggestions.append(f"  • 考虑使用多辆车或分批配送")

    if distances_from_start and max(distances_from_start) * 2 > vehicle.max_distance * 0.8:
        suggested_distance = max(distances_from_start) * 3
        suggestions.append(f"  • 将最大行驶距离提升至至少 {suggested_distance:.0f} km")

    if expired_count > 0:
        suggestions.append(f"  • 调整时间窗，确保在当前时间之后")

    if not suggestions:
        print("  ✅ 当前配置合理，可以正常运行")
    else:
        for s in suggestions:
            print(s)

    print("=" * 60 + "\n")

class GeneticAlgorithm:  # Renamed from GeneticVRPSolver for clarity in this context
    """遗传算法求解VRP"""

    def __init__(self, points: List[DeliveryPoint], router,
                 population_size: int = 50, generations: int = 100,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        self.points = points
        self.router = router
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.search_steps = []  # Used to store progress for animation

    def evolve(self, vehicle) -> List[DeliveryPoint]:  # Added vehicle parameter
        """遗传算法优化路径"""
        if not self.points:
            return []

        self.search_steps = []

        # Initialize population
        population = self._initialize_population()

        best_route = None
        best_distance = float('inf')  # Minimize distance

        for generation in range(self.generations):
            # Calculate fitness (total route distance with penalties)
            fitness_scores = []
            for individual in population:
                # Use _calculate_route_distance which takes vehicle
                distance = self._calculate_route_distance(individual, vehicle)
                fitness_scores.append(distance)

            # Record best solution from current generation
            # Find the index of the minimum fitness score (shortest distance)
            min_idx = np.argmin(fitness_scores)
            current_best_route = population[min_idx].copy()
            current_best_distance = fitness_scores[min_idx]

            # Update overall best solution found so far
            if current_best_distance < best_distance:
                best_distance = current_best_distance
                best_route = current_best_route.copy()

            # 每5代记录一次搜索步骤 (for animation)
            if generation % 5 == 0 or generation == 0:
                self.search_steps.append({
                    'iteration': generation,
                    'route': [p.id for p in best_route] if best_route else [],
                    'distance': best_distance
                })

            # Selection (using tournament selection)
            parents = self._tournament_selection(population, fitness_scores)

            # Crossover and Mutation to create next generation
            next_population = []
            for i in range(0, len(parents), 2):
                parent1 = parents[i]
                # Ensure parent2 exists, if not, cycle back to parent1
                parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]

                if random.random() < self.crossover_rate:
                    # Using Order Crossover (OX) as implemented
                    child1, child2 = self._order_crossover(parent1, parent2)
                else:
                    # If no crossover, children are copies of parents
                    child1, child2 = parent1[:], parent2[:]

                # Apply mutation
                if random.random() < self.mutation_rate:
                    child1 = self._swap_mutation(child1)  # Using swap mutation
                if random.random() < self.mutation_rate:
                    child2 = self._swap_mutation(child2)  # Using swap mutation

                next_population.extend([child1, child2])

            # Replace old population with new offspring, ensuring population size
            population = next_population[:self.population_size]

        # 添加最终结果记录
        if best_route:
            final_distance = self._calculate_route_distance(best_route, vehicle)
            self.search_steps.append({
                'iteration': self.generations,
                'route': [p.id for p in best_route],
                'distance': final_distance
            })
            # Ensure the returned route is valid in some basic sense, or rely on fitness
            return self._ensure_valid_route(best_route)
        else:
            # If no valid route found, return original points or empty list
            return self.points if self.points else []

    def _initialize_population(self) -> List[List[DeliveryPoint]]:
        """Initialize population with valid random permutations."""
        population = []
        for _ in range(self.population_size):
            # Create a valid permutation. _ensure_valid_route likely handles basic structure.
            # Then shuffle to create diversity.
            individual = self._ensure_valid_route(self.points.copy())
            random.shuffle(individual)  # Shuffle to create diverse initial routes
            population.append(individual)
        return population

    def _ensure_valid_route(self, route: List[DeliveryPoint]) -> List[DeliveryPoint]:
        """
        Ensures a route is valid based on vehicle capacity and distance.
        This is a simplified approach. A more robust VRP GA would handle depot returns
        and time windows more explicitly during initialization and mutation/crossover.
        For now, it just checks overall constraints.
        """
        # The current implementation seems to just return the route, possibly intended for
        # structural validation or as a placeholder. The actual constraint checks are in
        # _calculate_route_distance (fitness function).
        # For this merge, we'll keep it minimal as per original structure.
        return route

    def _calculate_route_distance(self, route: List[DeliveryPoint], vehicle: Vehicle) -> float:
        """Calculate total route distance including penalties for constraints."""
        total_distance = 0.0
        penalty = 0.0
        current_load = 0.0
        current_location = vehicle.start_location  # Start from vehicle's start location

        # If the route is empty, return a high distance (penalty)
        if not route:
            return float('inf')

        for point in route:
            # Calculate distance to the next point using the router's method
            # Ensure `self.router` is valid and has `_calculate_distance`
            dist_to_point = self.router._calculate_distance(current_location, point.location)
            total_distance += dist_to_point
            current_load += point.demand
            current_location = point.location  # Move to the current point's location

            # Capacity constraint penalty: if current load exceeds vehicle capacity
            if current_load > vehicle.capacity:
                # Add a penalty proportional to the excess load
                penalty += 1000 * (current_load - vehicle.capacity)

                # Max distance constraint penalty: if total distance exceeds vehicle's max distance
            if total_distance > vehicle.max_distance:
                # Add a penalty proportional to the excess distance
                penalty += 500 * (total_distance - vehicle.max_distance)

            # Note: Time window constraints are not directly penalized here in _calculate_route_distance.
            # A full VRP GA would also need to incorporate time window feasibility checks,
            # possibly as part of the route validation or in a more complex fitness function.
            # For this implementation, we focus on distance and capacity.

        # Add distance to return to the warehouse/start location
        dist_to_warehouse = self.router._calculate_distance(current_location, vehicle.start_location)
        total_distance += dist_to_warehouse

        # Total fitness is the sum of total distance and penalties
        return total_distance + penalty

    def _tournament_selection(self, population: List, fitness_scores: List[float], k: int = 3) -> List:
        """Tournament selection."""
        selected = []
        for _ in range(len(population)):
            # Randomly select k individuals for the tournament
            tournament_indices = np.random.choice(len(population), k, replace=False)
            # Find the winner (individual with the lowest fitness score, i.e., shortest distance)
            winner_index = tournament_indices[np.argmin([fitness_scores[i] for i in tournament_indices])]
            selected.append(population[winner_index].copy())
        return selected

    def _order_crossover(self, parent1: List, parent2: List) -> Tuple[List, List]:
        """Order Crossover (OX)."""
        size = len(parent1)
        if size == 0: return [], []

        # Select two random crossover points
        start, end = sorted(np.random.choice(size, 2, replace=False))

        # Initialize children with None values
        child1 = [None] * size
        child2 = [None] * size

        # Copy the segment from parent1 to child1 and parent2 to child2
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]

        # Fill the remaining genes for child1 from parent2, maintaining order
        self._fill_offspring(child1, parent2, end)
        # Fill the remaining genes for child2 from parent1, maintaining order
        self._fill_offspring(child2, parent1, end)

        return child1, child2

    def _fill_offspring(self, child: List, parent: List, start_pos: int):
        """Helper to fill offspring after crossover."""
        # Get the set of genes already present in the child
        child_genes_set = set(gene for gene in child if gene is not None)

        # Create a list of genes from the parent, starting from after the crossover segment and wrapping around
        parent_genes = parent[start_pos:] + parent[:start_pos]

        # Determine the starting position for filling the child
        child_pos = start_pos % len(child)

        # Iterate through the parent's genes and fill them into the child if not already present
        for gene in parent_genes:
            if gene not in child_genes_set:
                # Find the next available slot in the child
                while child[child_pos] is not None:
                    child_pos = (child_pos + 1) % len(child)
                # Place the gene in the child
                child[child_pos] = gene
                # Move to the next position in the child
                child_pos = (child_pos + 1) % len(child)

    def _swap_mutation(self, individual: List):
        """Swap mutation: randomly swaps two genes in the individual."""
        if len(individual) < 2:
            return  # Cannot swap if less than 2 elements
        # Select two distinct random indices
        idx1, idx2 = np.random.choice(len(individual), 2, replace=False)
        # Swap the elements at these indices
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual


class DynamicRoutePlanner:
    """动态路径重规划系统"""

    def __init__(self):
        self.traffic_conditions = {}  # 实时路况
        self.router = AStarRouter()

    def update_traffic(self, edge: Tuple[str, str], congestion_factor: float):
        """更新路况信息"""
        self.traffic_conditions[edge] = congestion_factor

    def adaptive_reroute(self, current_position: str,
                         remaining_points: List[DeliveryPoint],
                         vehicle: Vehicle) -> List[str]:
        """自适应重路由"""
        # 考虑实时路况的动态规划
        best_route = []
        min_time = float('inf')

        # 尝试不同的访问顺序
        # Limit permutations to avoid excessive computation for large remaining_points
        # Consider a smaller subset for permutation, or use a metaheuristic like GA/SA for re-optimization
        # For now, limiting to min(6, len(remaining_points)) to keep it computationally feasible.
        for perm in itertools.permutations(remaining_points[:min(6, len(remaining_points))]):
            route_time = self._estimate_route_time(current_position, list(perm), vehicle)
            if route_time < min_time:
                min_time = route_time
                best_route = list(perm)

        return best_route

    def _estimate_route_time(self, start: str, points: List, vehicle: Vehicle) -> float:
        """估算路径总时间"""
        total_time = 0
        current = start

        for point in points:
            # 考虑路况的时间计算
            # Use A* to find the path and distance
            path, distance = self.router.a_star_search(current, point.id, 'distance')  # Use 'distance' for pathfinding

            if not path:  # If no path found, consider it impossible or very long time
                return float('inf')

            # Calculate travel time along the path, applying congestion factor
            travel_time_segment = 0
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                # Retrieve edge data from router.graph for accurate speed/time
                edge_data = self.router.graph.get(path[i], {}).get(path[i + 1])
                if edge_data:
                    base_time = edge_data['time']  # Use pre-defined time if available
                    congestion_factor = self.traffic_conditions.get(edge, 1.0)  # Default to 1.0 if no traffic data
                    travel_time_segment += base_time * congestion_factor
                else:
                    # Fallback if edge data is missing (should not happen if router is built correctly)
                    # Approximate time based on distance and vehicle speed
                    dist_segment = self.router._calculate_distance(self.router.node_coordinates[path[i]],
                                                                   self.router.node_coordinates[path[i + 1]])
                    base_speed = vehicle.speed if vehicle.speed > 0 else 1.0  # Avoid division by zero
                    travel_time_segment += (dist_segment / base_speed) * self.traffic_conditions.get(edge, 1.0)

            total_time += travel_time_segment + (point.service_time / 60.0)  # Add service time in hours
            current = point.id

        return total_time


class MultiObjectiveOptimizer:
    """多目标路径优化"""

    def __init__(self, weight_cost: float = 0.4,
                 weight_time: float = 0.4,
                 weight_emission: float = 0.2):
        self.weights = {
            'cost': weight_cost,
            'time': weight_time,
            'emission': weight_emission
        }

    def optimize_route(self, points: List[DeliveryPoint],
                       vehicle: Vehicle) -> List[DeliveryPoint]:
        """多目标优化"""
        n = len(points)

        def objective(x):
            # x is a permutation encoded by values, we need to sort them to get the order
            # np.argsort returns indices that would sort an array. So argsort(x) gives the permutation.
            route_indices = np.argsort(x).astype(int)
            # Convert indices to the actual DeliveryPoint objects
            route = [points[i] for i in route_indices]

            # Calculate the three objectives for the given route
            cost = self._calculate_cost(route, vehicle)
            time = self._calculate_time(route, vehicle)
            emission = self._calculate_emission(route, vehicle)

            # Weighted sum of objectives
            return (self.weights['cost'] * cost +
                    self.weights['time'] * time +
                    self.weights['emission'] * emission)

        # Use differential evolution for optimization
        # Bounds for each variable (index in the permutation) are 0 to n-1.
        bounds = [(0, n - 1) for _ in range(n)]
        # Run differential evolution. `maxiter` controls the number of iterations.
        # `popsize` (default 15) might need tuning. `tol` (default 0.01) is tolerance for convergence.
        result = differential_evolution(objective, bounds, maxiter=100, popsize=20)

        # The `result.x` contains the optimized values. Use `np.argsort` again to get the permutation.
        route_indices = np.argsort(result.x).astype(int)
        # Reconstruct the optimized route from the sorted indices
        optimized_route = [points[i] for i in route_indices]

        return optimized_route

    def _calculate_cost(self, route: List[DeliveryPoint], vehicle: Vehicle) -> float:
        """Calculate total cost for the route."""
        fuel_cost_per_km = 0.8  # Fuel cost per kilometer
        driver_cost_per_hour = 50  # Driver cost per hour

        total_distance = 0.0
        total_time = 0.0
        current_loc = vehicle.start_location  # Start from warehouse

        for point in route:
            # Calculate distance from current location to the next point
            dist = self._calc_distance(current_loc, point.location)
            total_distance += dist
            # Calculate travel time and add service time
            # Speed is in km/h, service_time is in minutes. Convert service_time to hours.
            travel_time = dist / vehicle.speed if vehicle.speed > 0 else float('inf')
            total_time += travel_time + point.service_time / 60.0
            current_loc = point.location  # Move to the current point

        # Add return trip to warehouse
        dist_to_warehouse = self._calc_distance(current_loc, vehicle.start_location)
        total_distance += dist_to_warehouse
        total_time += dist_to_warehouse / vehicle.speed if vehicle.speed > 0 else float('inf')

        # Total cost = (distance * fuel_cost) + (time * driver_cost)
        return total_distance * fuel_cost_per_km + total_time * driver_cost_per_hour

    def _calculate_time(self, route: List[DeliveryPoint], vehicle: Vehicle) -> float:
        """Calculate total travel and service time for the route."""
        total_time = 0.0
        current_loc = vehicle.start_location

        for point in route:
            dist = self._calc_distance(current_loc, point.location)
            # Travel time + service time (in hours)
            travel_time = dist / vehicle.speed if vehicle.speed > 0 else float('inf')
            total_time += travel_time + point.service_time / 60.0
            current_loc = point.location

        # Add return trip time
        dist_to_warehouse = self._calc_distance(current_loc, vehicle.start_location)
        total_time += dist_to_warehouse / vehicle.speed if vehicle.speed > 0 else float('inf')

        return total_time

    def _calculate_emission(self, route: List[DeliveryPoint], vehicle: Vehicle) -> float:
        """Calculate total carbon emission for the route."""
        emission_per_km = 0.15  # kg CO2 per km (example value)

        total_distance = 0.0
        current_loc = vehicle.start_location

        for point in route:
            dist = self._calc_distance(current_loc, point.location)
            total_distance += dist
            current_loc = point.location

        # Add return trip distance
        dist_to_warehouse = self._calc_distance(current_loc, vehicle.start_location)
        total_distance += dist_to_warehouse

        # Total emission = total distance * emission per km
        return total_distance * emission_per_km

    def _calc_distance(self, loc1: Tuple[float, float],
                       loc2: Tuple[float, float]) -> float:
        """Calculate distance between two locations using Euclidean distance approximation
           and converting degrees to kilometers.
           This is a simplification for demonstration purposes.
           A real-world application would use the Haversine formula or the router's method.
        """
        from math import sqrt
        lat1, lon1 = loc1
        lat2, lon2 = loc2
        return sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) * 111  # Convert degree difference to km


class PerformanceAnalyzer:
    """性能分析器 - 评估算法性能并生成可视化报告"""

    def __init__(self):
        self.results: Dict[str, Dict] = {}
        self.route_data: Dict[str, Dict] = {}
        self.search_history: Dict[str, List] = {}

    def _validate_coordinates(self, coords):
        """验证坐标是否有效"""
        if coords is None:
            return False
        if not isinstance(coords, (tuple, list)) or len(coords) != 2:
            return False
        lat, lon = coords
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            return False
        if np.isnan(lat) or np.isnan(lon) or np.isinf(lat) or np.isinf(lon):
            return False
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return False
        return True

    def add_result(self, algorithm_name: str, route: List, vehicle,
                   router=None, execution_time: float = 0.0, search_steps: List = None):
        """添加算法结果（修复版）"""
        if not route:
            print(f"⚠️  算法 '{algorithm_name}' 返回的路径为空，跳过结果分析。")
            return

        # 过滤掉坐标无效的配送点
        valid_route = [p for p in route if self._validate_coordinates(p.location)]
        if len(valid_route) < len(route):
            print(f"⚠️  {algorithm_name}: {len(route) - len(valid_route)} 个配送点坐标无效，已过滤")

        if not valid_route:
            print(f"⚠️  算法 '{algorithm_name}' 没有有效的配送点，跳过")
            return

        # 验证车辆起点坐标
        if not self._validate_coordinates(vehicle.start_location):
            print(f"⚠️  算法 '{algorithm_name}' 车辆起点坐标无效，跳过")
            return

        # 基础指标计算（使用验证后的路径）
        total_distance = 0.0
        total_time = 0.0
        current_location = vehicle.start_location
        total_demand = sum(p.demand for p in valid_route)

        for point in valid_route:
            dist = 0.0
            if router and hasattr(router, 'a_star_search'):
                try:
                    current_node_id = router._get_node_id(current_location)
                    if not current_node_id and current_location == vehicle.start_location:
                        current_node_id = vehicle.start_location_id if hasattr(vehicle, 'start_location_id') else None

                    if current_node_id and point.id and current_node_id in router.nodes and point.id in router.nodes:
                        path, segment_dist = router.a_star_search(current_node_id, point.id, 'distance')
                        if path and np.isfinite(segment_dist):
                            dist = segment_dist
                        else:
                            dist = self._haversine_distance(current_location, point.location)
                    else:
                        dist = self._haversine_distance(current_location, point.location)
                except Exception:
                    dist = self._haversine_distance(current_location, point.location)
            else:
                dist = self._haversine_distance(current_location, point.location)

            total_distance += dist
            total_time += dist / vehicle.speed if vehicle.speed > 0 else 0
            total_time += point.service_time / 60.0
            current_location = point.location

        # 返程
        return_dist = self._haversine_distance(current_location, vehicle.start_location)
        total_distance += return_dist
        total_time += return_dist / vehicle.speed if vehicle.speed > 0 else 0

        # 成本计算
        fuel_cost_per_km = 0.8
        driver_cost_per_hour = 50
        total_cost = total_distance * fuel_cost_per_km + total_time * driver_cost_per_hour

        # 碳排放
        emission_per_km = 0.15
        carbon_emission = total_distance * emission_per_km

        # 效率指标
        capacity_utilization = (total_demand / vehicle.capacity * 100) if vehicle.capacity > 0 else 0

        route_efficiency = 0
        if len(valid_route) > 0 and total_distance > 0:
            direct_distance = self._haversine_distance(vehicle.start_location, valid_route[-1].location)
            route_efficiency = (direct_distance / total_distance * 100)

        result = {
            'algorithm': algorithm_name,
            'total_distance': round(total_distance, 2),
            'total_time': round(total_time, 2),
            'total_cost': round(total_cost, 2),
            'carbon_emission': round(carbon_emission, 2),
            'num_deliveries': len(valid_route),
            'total_demand': round(total_demand, 2),
            'capacity_utilization': round(capacity_utilization, 2),
            'route_efficiency': round(route_efficiency, 2),
            'avg_distance_per_stop': round(total_distance / len(valid_route), 2) if len(valid_route) > 0 else 0,
            'route': [p.id for p in valid_route],
            'execution_time': round(execution_time, 4)
        }

        self.results[algorithm_name] = result
        self.route_data[algorithm_name] = {
            'route': valid_route,
            'vehicle': vehicle,
            'router': router,
            'warehouse_id': vehicle.start_location_id if hasattr(vehicle, 'start_location_id') else None
        }

        if search_steps:
            self.search_history[algorithm_name] = search_steps

        print(f"\n📊 算法 '{algorithm_name}' 性能指标:")
        print(f"  总距离: {result['total_distance']:.2f} km")
        print(f"  总时间: {result['total_time']:.2f} 小时")
        print(f"  总成本: ¥{result['total_cost']:.2f}")
        print(f"  碳排放: {result['carbon_emission']:.2f} kg CO2")
        print(f"  配送点数: {result['num_deliveries']}")
        print(f"  执行时间: {result['execution_time']:.4f} 秒")

    def _haversine_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """计算Haversine距离（增加验证）"""
        if not self._validate_coordinates(loc1) or not self._validate_coordinates(loc2):
            return 0.0

        from math import radians, sin, cos, sqrt, atan2
        lat1, lon1 = radians(loc1[0]), radians(loc1[1])
        lat2, lon2 = radians(loc2[0]), radians(loc2[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return 6371 * c

    def visualize_results(self, output_dir: str = "output") -> None:
        """生成可视化图表（修复版）"""
        if not self.results:
            print("\n⚠️  没有结果可供可视化")
            return

        os.makedirs(output_dir, exist_ok=True)

        # 依次生成各种图表，添加异常处理
        try:
            self._plot_performance_comparison(output_dir)
        except Exception as e:
            print(f"  ⚠️ 性能对比图生成失败: {e}")

        try:
            self._plot_cost_breakdown(output_dir)
        except Exception as e:
            print(f"  ⚠️ 成本分解图生成失败: {e}")

        try:
            self._plot_radar_chart(output_dir)
        except Exception as e:
            print(f"  ⚠️ 雷达图生成失败: {e}")

        try:
            self._plot_route_network(output_dir)
        except Exception as e:
            print(f"  ⚠️ 路径网络图生成失败: {e}")

        try:
            self._plot_delivery_sequence(output_dir)
        except Exception as e:
            print(f"  ⚠️ 配送顺序图生成失败: {e}")

        try:
            self._plot_time_gantt(output_dir)
        except Exception as e:
            print(f"  ⚠️ 时间甘特图生成失败: {e}")

        try:
            self._plot_route_comparison_map(output_dir)
        except Exception as e:
            print(f"  ⚠️ 路径对比地图生成失败: {e}")

        try:
            self._plot_network_topology(output_dir)
        except Exception as e:
            print(f"  ⚠️ 网络拓扑图生成失败: {e}")

        if self.search_history:
            try:
                self._generate_search_animation(output_dir)
            except Exception as e:
                print(f"  ⚠️ 搜索动画生成失败: {e}")

        print(f"\n📈 可视化图表已保存到: {output_dir}/")

    def _plot_performance_comparison(self, output_dir: str):
        """绘制性能对比柱状图"""
        # Create a 2x2 grid of subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('算法性能对比分析', fontsize=16, fontweight='bold')

        algorithms = list(self.results.keys())  # Get the names of the algorithms

        # Subplot 1: Total Distance Comparison
        distances = [self.results[algo]['total_distance'] for algo in algorithms]
        axes[0, 0].bar(algorithms, distances, color='skyblue')
        axes[0, 0].set_title('总距离对比')
        axes[0, 0].set_ylabel('距离 (km)')
        axes[0, 0].grid(axis='y', alpha=0.3)  # Add horizontal grid lines

        # Subplot 2: Total Cost Comparison
        costs = [self.results[algo]['total_cost'] for algo in algorithms]
        axes[0, 1].bar(algorithms, costs, color='lightcoral')
        axes[0, 1].set_title('总成本对比')
        axes[0, 1].set_ylabel('成本 (¥)')
        axes[0, 1].grid(axis='y', alpha=0.3)

        # Subplot 3: Carbon Emission Comparison
        emissions = [self.results[algo]['carbon_emission'] for algo in algorithms]
        axes[1, 0].bar(algorithms, emissions, color='lightgreen')
        axes[1, 0].set_title('碳排放对比')
        axes[1, 0].set_ylabel('CO2 (kg)')
        axes[1, 0].grid(axis='y', alpha=0.3)

        # Subplot 4: Efficiency Metrics Comparison (Route Efficiency and Capacity Utilization)
        efficiency = [self.results[algo]['route_efficiency'] for algo in algorithms]
        capacity = [self.results[algo]['capacity_utilization'] for algo in algorithms]

        x = np.arange(len(algorithms))  # The x locations for the groups
        width = 0.35  # The width of the bars
        axes[1, 1].bar(x - width / 2, efficiency, width, label='路线效率', color='gold')
        axes[1, 1].bar(x + width / 2, capacity, width, label='载重率', color='orange')
        axes[1, 1].set_title('效率指标对比')
        axes[1, 1].set_ylabel('百分比 (%)')
        axes[1, 1].set_xticks(x)  # Set tick locations
        axes[1, 1].set_xticklabels(algorithms)  # Set tick labels
        axes[1, 1].legend()  # Show legend
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()  # Adjust layout to prevent overlapping titles/labels
        plt.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

        print(f"  ✅ 已生成: performance_comparison.png")

    def _plot_cost_breakdown(self, output_dir: str):
        """绘制成本分解饼图"""
        # Create subplots, one for each algorithm
        fig, axes = plt.subplots(1, len(self.results), figsize=(6 * len(self.results), 5))
        if len(self.results) == 1:  # If only one algorithm, axes is not an array
            axes = [axes]

        fig.suptitle('成本结构分析', fontsize=16, fontweight='bold')

        # Iterate through each algorithm's results
        for idx, (algo_name, result) in enumerate(self.results.items()):
            # Calculate cost components (fuel and driver)
            fuel_cost = result['total_distance'] * 0.8  # Fuel cost per km
            driver_cost = result['total_time'] * 50  # Driver cost per hour

            sizes = [fuel_cost, driver_cost]  # Data for the pie chart slices
            labels = ['燃油成本', '人工成本']  # Labels for the slices
            colors = ['#ff9999', '#66b3ff']  # Colors for the slices
            explode = (0.05, 0.05)  # Explode slices slightly

            # Plot the pie chart
            axes[idx].pie(sizes, explode=explode, labels=labels, colors=colors,
                          autopct='%1.1f%%', shadow=True, startangle=90)
            axes[idx].set_title(f'{algo_name}\n总成本: ¥{result["total_cost"]:.2f}')  # Set title with total cost

        plt.tight_layout()
        plt.savefig(f"{output_dir}/cost_breakdown.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✅ 已生成: cost_breakdown.png")

    def _plot_radar_chart(self, output_dir: str):
        """绘制雷达图 - 使用更稳定的实现方式"""
        try:
            if not self.results or len(self.results) == 0:
                print("  ⚠️  没有数据可绘制雷达图")
                return

            print("  📊 生成雷达图...")

            # 如果数据太少，跳过雷达图
            if len(self.results) < 1:
                print("  ⚠️  数据不足，跳过雷达图生成")
                return

            # Categories for the radar chart
            categories = ['距离优化', '时间效率', '成本控制', '环保指标', '载重利用', '路线效率']
            num_vars = len(categories)

            # Calculate angles
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]  # 闭合

            # 创建图表
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='polar')

            # 设置
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(np.linspace(0, 2 * np.pi, num_vars, endpoint=False))
            ax.set_xticklabels(categories, fontsize=10)
            ax.set_ylim(0, 100)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8, color='gray')
            ax.grid(True, linestyle='--', alpha=0.7)

            # 收集所有数据用于归一化
            all_distances = []
            all_times = []
            all_costs = []
            all_emissions = []

            for result in self.results.values():
                if result.get('total_distance') and result['total_distance'] > 0:
                    all_distances.append(result['total_distance'])
                if result.get('total_time') and result['total_time'] > 0:
                    all_times.append(result['total_time'])
                if result.get('total_cost') and result['total_cost'] > 0:
                    all_costs.append(result['total_cost'])
                if result.get('carbon_emission') and result['carbon_emission'] > 0:
                    all_emissions.append(result['carbon_emission'])

            # 设置最大值（避免除零）
            max_distance = max(all_distances) if all_distances else 100.0
            max_time = max(all_times) if all_times else 100.0
            max_cost = max(all_costs) if all_costs else 100.0
            max_emission = max(all_emissions) if all_emissions else 100.0

            # 绘制每个算法
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']

            for idx, (algo_name, result) in enumerate(self.results.items()):
                try:
                    # 提取并清理数据
                    distance = result.get('total_distance', 0) or 0
                    time = result.get('total_time', 0) or 0
                    cost = result.get('total_cost', 0) or 0
                    emission = result.get('carbon_emission', 0) or 0
                    capacity = result.get('capacity_utilization', 0) or 50
                    efficiency = result.get('route_efficiency', 0) or 50

                    # 归一化（反转成本指标，越小越好变成越大越好）
                    values = [
                        100 - min(100, (distance / max_distance * 100)) if max_distance > 0 else 50,
                        100 - min(100, (time / max_time * 100)) if max_time > 0 else 50,
                        100 - min(100, (cost / max_cost * 100)) if max_cost > 0 else 50,
                        100 - min(100, (emission / max_emission * 100)) if max_emission > 0 else 50,
                        min(100, max(0, capacity)),
                        min(100, max(0, efficiency))
                    ]

                    # 确保所有值有效且在范围内
                    values = [float(v) if np.isfinite(v) else 50.0 for v in values]
                    values = [max(0.0, min(100.0, v)) for v in values]

                    # 闭合数据
                    values += values[:1]

                    # 验证长度
                    if len(values) != len(angles):
                        print(f"  ⚠️  {algo_name} 数据长度不匹配，跳过")
                        continue

                    # 绘制
                    color = colors[idx % len(colors)]
                    ax.plot(angles, values, 'o-', linewidth=2,
                            label=algo_name, color=color, markersize=6)
                    ax.fill(angles, values, alpha=0.15, color=color)

                except Exception as e:
                    print(f"  ⚠️  绘制 {algo_name} 失败: {e}")
                    continue

            # 标题和图例
            ax.set_title('算法综合性能对比\n(数值越高表现越好)',
                         fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

            # 保存
            plt.tight_layout()
            output_path = os.path.join(output_dir, 'performance_radar.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"  ✅ 雷达图已保存: {output_path}")

        except Exception as e:
            print(f"  ❌ 雷达图生成失败: {e}")
            print("  ℹ️  继续生成其他图表...")
            # 不抛出异常，允许程序继续执行

    def _plot_route_network(self, output_dir: str):
        """绘制路径网络图（修复版）"""
        if not self.route_data:
            return

        fig, axes = plt.subplots(1, len(self.route_data), figsize=(8 * len(self.route_data), 7))
        if len(self.route_data) == 1:
            axes = [axes]

        fig.suptitle('路径网络图', fontsize=16, fontweight='bold')

        for idx, (algo_name, data) in enumerate(self.route_data.items()):
            route = data['route']
            vehicle = data['vehicle']
            warehouse_id = data.get('warehouse_id')

            G = nx.DiGraph()
            pos = {}

            # 验证并添加仓库节点
            if warehouse_id and self._validate_coordinates(vehicle.start_location):
                G.add_node(warehouse_id)
                pos[warehouse_id] = vehicle.start_location
            else:
                warehouse_id = "Warehouse"
                if self._validate_coordinates(vehicle.start_location):
                    G.add_node(warehouse_id)
                    pos[warehouse_id] = vehicle.start_location

            # 验证并添加配送点节点
            valid_route = []
            for point in route:
                if self._validate_coordinates(point.location):
                    G.add_node(point.id)
                    pos[point.id] = point.location
                    valid_route.append(point)
                else:
                    print(f"  ⚠️ 跳过无效坐标的节点: {point.id}")

            # 如果没有有效节点，跳过此图
            if len(valid_route) == 0:
                print(f"  ⚠️ {algo_name}: 没有有效的配送点，跳过网络图")
                axes[idx].text(0.5, 0.5, '无有效数据', ha='center', va='center')
                axes[idx].axis('off')
                continue

            # 添加边
            if valid_route and warehouse_id in pos:
                G.add_edge(warehouse_id, valid_route[0].id)
                for i in range(len(valid_route) - 1):
                    G.add_edge(valid_route[i].id, valid_route[i + 1].id)
                if valid_route:
                    G.add_edge(valid_route[-1].id, warehouse_id)

            try:
                # 绘制节点
                if warehouse_id in pos:
                    nx.draw_networkx_nodes(G, pos, nodelist=[warehouse_id],
                                           node_color='red', node_size=800,
                                           node_shape='s', ax=axes[idx], label='仓库')

                if valid_route:
                    nx.draw_networkx_nodes(G, pos, nodelist=[p.id for p in valid_route],
                                           node_color='lightblue', node_size=500,
                                           ax=axes[idx], label='配送点')

                # 绘制边
                nx.draw_networkx_edges(G, pos, edge_color='gray',
                                       arrows=True, arrowsize=20,
                                       arrowstyle='->', ax=axes[idx],
                                       connectionstyle='arc3,rad=0.1')

                # 绘制标签
                labels = {warehouse_id: '仓库'} if warehouse_id in pos else {}
                labels.update({p.id: p.id for p in valid_route})
                nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=axes[idx])

                axes[idx].set_title(
                    f'{algo_name}\n配送点数: {len(valid_route)}, 总距离: {self.results[algo_name]["total_distance"]:.2f}km')
                axes[idx].legend(loc='upper right')
                axes[idx].axis('off')
            except Exception as e:
                print(f"  ⚠️ 绘制 {algo_name} 网络图失败: {e}")
                axes[idx].text(0.5, 0.5, f'绘制失败: {str(e)[:30]}', ha='center', va='center')
                axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/route_network.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 已生成: route_network.png")

    def _plot_delivery_sequence(self, output_dir: str):
        """绘制配送顺序对比图"""
        if not self.route_data:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        algorithms = list(self.route_data.keys())
        y_positions = np.arange(len(algorithms))  # Y positions for each algorithm's sequence

        # Draw the sequence for each algorithm
        for idx, (algo_name, data) in enumerate(self.route_data.items()):
            route = data['route']

            # Plot delivery points as markers and label them
            for i, point in enumerate(route):
                ax.scatter(i, y_positions[idx], s=200, c='skyblue', edgecolors='black',
                           zorder=3)  # zorder to bring points to front
                ax.text(i, y_positions[idx], point.id, ha='center', va='center', fontsize=8)

            # Draw dashed lines connecting the delivery sequence
            if len(route) > 1:
                x_coords = list(range(len(route)))
                y_coords = [y_positions[idx]] * len(route)
                ax.plot(x_coords, y_coords, 'gray', linestyle='--', alpha=0.5, zorder=1)

        # Configure axes
        ax.set_yticks(y_positions)
        ax.set_yticklabels(algorithms)
        ax.set_xlabel('配送顺序', fontsize=12)
        ax.set_title('配送顺序对比图', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)  # Grid lines for x-axis (sequence order)
        # Adjust x-axis limits to fit all points nicely
        max_route_length = max(len(data['route']) for data in self.route_data.values()) if self.route_data else 0
        ax.set_xlim(-0.5, max_route_length - 0.5 if max_route_length > 0 else -0.5)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/delivery_sequence.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✅ 已生成: delivery_sequence.png")

    def _plot_time_gantt(self, output_dir: str):
        """绘制时间甘特图"""
        if not self.route_data:
            return

        fig, ax = plt.subplots(figsize=(14, 6))

        algorithms = list(self.route_data.keys())
        y_positions = np.arange(len(algorithms))

        # Generate distinct colors for each point's service time
        colors = plt.cm.Set3(np.linspace(0, 1, 10))  # Using a colormap for service time bars

        # Plot Gantt chart for each algorithm
        for idx, (algo_name, data) in enumerate(self.route_data.items()):
            route = data['route']
            vehicle = data['vehicle']

            current_time = 0.0  # Cumulative time in hours, starting from 0
            current_location = vehicle.start_location  # Start from warehouse

            # Plot warehouse start time (implicitly 0) - not explicitly drawn as a bar but sets the 'left' for the first segment.

            for i, point in enumerate(route):
                # Calculate travel time to the current point
                dist = self._haversine_distance(current_location, point.location)
                travel_time = dist / vehicle.speed if vehicle.speed > 0 else 0

                # Draw travel time segment (as a gray bar)
                ax.barh(y_positions[idx], travel_time, left=current_time,
                        height=0.3, color='lightgray', edgecolor='black',
                        label='行驶' if i == 0 else '')  # Add label only once for legend
                current_time += travel_time  # Update cumulative time

                # Draw service time segment (using distinct colors per point)
                service_time_hours = point.service_time / 60.0  # Convert minutes to hours
                ax.barh(y_positions[idx], service_time_hours, left=current_time,
                        height=0.3, color=colors[i % len(colors)], edgecolor='black')
                # Add point ID text inside the service bar
                ax.text(current_time + service_time_hours / 2, y_positions[idx],
                        point.id, ha='center', va='center', fontsize=8, fontweight='bold')

                current_time += service_time_hours  # Update cumulative time
                current_location = point.location  # Move to the current point

            # Add return trip to warehouse
            dist = self._haversine_distance(current_location, vehicle.start_location)
            travel_time = dist / vehicle.speed if vehicle.speed > 0 else 0
            ax.barh(y_positions[idx], travel_time, left=current_time,
                    height=0.3, color='lightgray', edgecolor='black')

        # Configure axes and labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(algorithms)
        ax.set_xlabel('时间 (小时)', fontsize=12)
        ax.set_title('配送时间甘特图', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)  # Grid lines for time axis

        # Add legend for travel and service
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='lightgray', edgecolor='black', label='行驶'),
                           Patch(facecolor=colors[0], edgecolor='black',
                                 label='服务')]  # Use first service color as representative
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/time_gantt.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✅ 已生成: time_gantt.png")

    def _plot_route_comparison_map(self, output_dir: str):
        """绘制路径对比地图（修复版）"""
        if not self.route_data:
            return

        fig, ax = plt.subplots(figsize=(12, 10))
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'x']

        has_valid_data = False

        for idx, (algo_name, data) in enumerate(self.route_data.items()):
            route = data['route']
            vehicle = data['vehicle']

            # 验证仓库位置
            if not self._validate_coordinates(vehicle.start_location):
                print(f"  ⚠️ {algo_name}: 仓库坐标无效，跳过")
                continue

            # 过滤有效的配送点
            valid_points = [p for p in route if self._validate_coordinates(p.location)]

            if not valid_points:
                print(f"  ⚠️ {algo_name}: 没有有效的配送点坐标，跳过")
                continue

            has_valid_data = True
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]

            # 构建路径坐标
            lats = [vehicle.start_location[0]]
            lons = [vehicle.start_location[1]]

            for p in valid_points:
                lats.append(p.location[0])
                lons.append(p.location[1])

            lats.append(vehicle.start_location[0])
            lons.append(vehicle.start_location[1])

            # 验证所有坐标
            if all(np.isfinite(lats)) and all(np.isfinite(lons)):
                try:
                    ax.plot(lons, lats, color=color, linewidth=2, alpha=0.6,
                            marker=marker, markersize=8,
                            label=f'{algo_name} ({self.results[algo_name]["total_distance"]:.2f}km)')
                except Exception as e:
                    print(f"  ⚠️ 绘制 {algo_name} 路径失败: {e}")
            else:
                print(f"  ⚠️ {algo_name}: 坐标包含无效值")

        if has_valid_data:
            # 绘制仓库
            warehouse_loc = list(self.route_data.values())[0]['vehicle'].start_location
            if self._validate_coordinates(warehouse_loc):
                ax.scatter(warehouse_loc[1], warehouse_loc[0], s=500, c='red',
                           marker='*', edgecolors='black', linewidths=2,
                           label='仓库', zorder=10)

            ax.set_xlabel('经度', fontsize=12)
            ax.set_ylabel('纬度', fontsize=12)
            ax.set_title('路径对比地图', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无有效路径数据', ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/route_comparison_map.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 已生成: route_comparison_map.png")

    # New method: Plot network topology
    def _plot_network_topology(self, output_dir: str):
        """绘制配送网络拓扑图（修复版）"""
        if not self.route_data:
            return

        first_algo_name = list(self.route_data.keys())[0]
        data = self.route_data[first_algo_name]
        router = data.get('router')
        vehicle = data['vehicle']
        all_points_in_route = data['route']

        if not router or not hasattr(router, 'graph') or not router.node_coordinates:
            print("  ⚠️  无法生成网络拓扑图：缺少路网或节点坐标数据")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('配送网络拓扑结构分析', fontsize=16, fontweight='bold')

        G = nx.Graph()
        pos = {}

        # 只添加有效坐标的节点
        valid_nodes = []
        for node_id, coords in router.node_coordinates.items():
            if self._validate_coordinates(coords):
                G.add_node(node_id)
                pos[node_id] = (coords[1], coords[0])  # (lon, lat)
                valid_nodes.append(node_id)
            else:
                print(f"  ⚠️ 跳过无效坐标的节点: {node_id}")

        if not valid_nodes:
            print("  ⚠️ 没有有效的节点坐标，无法绘制拓扑图")
            ax1.text(0.5, 0.5, '无有效节点数据', ha='center', va='center', transform=ax1.transAxes)
            ax1.axis('off')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/network_topology.png", dpi=300, bbox_inches='tight')
            plt.close()
            return

        # 只添加连接有效节点的边
        for node, neighbors in router.graph.items():
            if node not in valid_nodes:
                continue
            for neighbor, edge_data in neighbors.items():
                if neighbor in valid_nodes and node in pos and neighbor in pos:
                    G.add_edge(node, neighbor, weight=edge_data['distance'])

        # 识别节点类型
        warehouse_nodes = []
        if hasattr(vehicle, 'start_location_id') and vehicle.start_location_id in valid_nodes:
            warehouse_nodes = [vehicle.start_location_id]

        delivery_nodes_in_route = [p.id for p in all_points_in_route if p.id in valid_nodes]
        other_nodes = list(set(valid_nodes) - set(warehouse_nodes) - set(delivery_nodes_in_route))

        try:
            # 绘制节点
            if warehouse_nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=warehouse_nodes,
                                       node_color='red', node_size=800,
                                       node_shape='s', ax=ax1, label='仓库')

            if delivery_nodes_in_route:
                nx.draw_networkx_nodes(G, pos, nodelist=delivery_nodes_in_route,
                                       node_color='lightblue', node_size=500,
                                       node_shape='o', ax=ax1, label='配送点 (路线中)')

            if other_nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=other_nodes,
                                       node_color='lightgray', node_size=300,
                                       node_shape='o', ax=ax1, label='其他节点')

            # 绘制边
            nx.draw_networkx_edges(G, pos, edge_color='gray', width=1,
                                   alpha=0.5, ax=ax1)

            # 绘制标签
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax1)

            ax1.set_title(f'完整网络拓扑\n节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}')
            ax1.legend(loc='upper right')
            ax1.set_xlabel('经度')
            ax1.set_ylabel('纬度')
            ax1.grid(True, alpha=0.3)

            # 统计图
            degrees = [G.degree(n) for n in G.nodes()]
            ax2_sub1 = plt.subplot(2, 2, 2)
            if degrees:
                ax2_sub1.hist(degrees, bins=range(min(degrees), max(degrees) + 2) if degrees else [0, 1],
                              color='skyblue', edgecolor='black', alpha=0.7)
            ax2_sub1.set_title('节点度分布')
            ax2_sub1.set_xlabel('度数')
            ax2_sub1.set_ylabel('节点数量')
            ax2_sub1.grid(True, alpha=0.3)

            weights = [data['weight'] for _, _, data in G.edges(data=True)]
            ax2_sub2 = plt.subplot(2, 2, 4)
            if weights:
                ax2_sub2.hist(weights, bins=20, color='lightcoral',
                              edgecolor='black', alpha=0.7)
            ax2_sub2.set_title('边长度分布')
            ax2_sub2.set_xlabel('距离 (km)')
            ax2_sub2.set_ylabel('边数量')
            ax2_sub2.grid(True, alpha=0.3)

        except Exception as e:
            print(f"  ⚠️ 绘制拓扑图失败: {e}")
            ax1.text(0.5, 0.5, f'绘制失败: {str(e)[:50]}', ha='center', va='center', transform=ax1.transAxes)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/network_topology.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 已生成: network_topology.png")

    # New method: Generate search animation
    def _generate_search_animation(self, output_dir: str):
        """生成路径搜索过程的动态图 (GIF)"""
        # Iterate through each algorithm for which search history is available
        for algo_name, steps in self.search_history.items():
            if not steps:
                continue  # Skip if no search steps recorded for this algorithm

            print(f"  🎬 正在生成 {algo_name} 搜索动画...")

            # Retrieve necessary data for animation from route_data
            data = self.route_data.get(algo_name)
            if not data:
                print(f"  ⚠️  无法找到 {algo_name} 的路由数据，跳过动画生成。")
                continue

            vehicle = data['vehicle']
            final_route_points = data['route']  # The final, best route (list of DeliveryPoint objects)

            # Set up the animation figure with two subplots: map view and performance plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            fig.suptitle(f'{algo_name} - 路径搜索过程动态演示',
                         fontsize=14, fontweight='bold')

            # Configure left plot (map view)
            ax1.set_xlabel('经度')
            ax1.set_ylabel('纬度')
            ax1.set_title('搜索路径演化')
            ax1.grid(True, alpha=0.3)

            # Configure right plot (performance plot)
            ax2.set_xlabel('迭代次数')
            ax2.set_ylabel('总距离 (km)')  # Assuming fitness = distance (to be minimized)
            ax2.set_title('优化过程')
            ax2.grid(True, alpha=0.3)

            # Draw static elements on the map plot (warehouse, delivery points)
            # Warehouse marker
            ax1.scatter(vehicle.start_location[1], vehicle.start_location[0],
                        s=500, c='red', marker='*', edgecolors='black',
                        linewidths=2, label='仓库', zorder=10)

            # Delivery points markers (final locations)
            for point in final_route_points:
                ax1.scatter(point.location[1], point.location[0],
                            s=200, c='lightblue', edgecolors='black',
                            linewidths=1.5, zorder=5)
                ax1.text(point.location[1], point.location[0], point.id,
                         ha='center', va='center', fontsize=8, fontweight='bold')

            # Animation update function: called for each frame
            def update(frame):
                # Ensure frame index is within the bounds of recorded steps
                frame_idx = min(frame, len(steps) - 1)

                step_data = steps[frame_idx]  # Get data for the current frame
                current_route_ids = step_data.get('route', [])  # List of point IDs in the current route
                distance = step_data.get('distance', 0)  # Current best distance (fitness)
                iteration = step_data.get('iteration', frame_idx)  # Current iteration number

                # --- Update Map Plot (ax1) ---
                artists_to_return = []  # List to hold artists that need to be redrawn for blitting (if used)

                # Clear previous path drawings to redraw the new path
                for artist in ax1.lines:  # Remove all existing lines (paths)
                    artist.remove()

                # Draw the current path if route IDs are available
                if current_route_ids:
                    lats = [vehicle.start_location[0]]  # Start latitude from warehouse
                    lons = [vehicle.start_location[1]]  # Start longitude from warehouse

                    # Map route IDs back to DeliveryPoint objects to get their locations
                    points_in_current_route = []
                    for point_id in current_route_ids:
                        # Find the DeliveryPoint object corresponding to the ID
                        point_obj = next((p for p in final_route_points if p.id == point_id), None)
                        if point_obj:
                            points_in_current_route.append(point_obj)
                            lats.append(point_obj.location[0])  # Add latitude
                            lons.append(point_obj.location[1])  # Add longitude

                    # Add return to warehouse location to complete the loop
                    lats.append(vehicle.start_location[0])
                    lons.append(vehicle.start_location[1])

                    # Plot the current path as a blue line with markers
                    line, = ax1.plot(lons, lats, 'b-', linewidth=2, alpha=0.6,
                                     marker='o', markersize=5)
                    artists_to_return.append(line)  # Add the new line to the list of artists to return

                # Update the title of the map plot with iteration info
                info_text = f'迭代: {iteration} | 当前距离: {distance:.2f} km'
                ax1.set_title(f'搜索路径演化\n{info_text}')

                # --- Update Performance Plot (ax2) ---
                # Clear the previous performance plot to redraw the updated history
                ax2.clear()
                # Extract history up to the current frame for plotting
                history_iterations = [s.get('iteration', i) for i, s in enumerate(steps[:frame_idx + 1])]
                history_distances = [s.get('distance', 0) for s in steps[:frame_idx + 1]]

                # Plot the performance history as a green line
                ax2.plot(history_iterations, history_distances, 'g-', linewidth=2, marker='o')
                # Highlight the current best point on the performance plot
                ax2.scatter(iteration, distance, s=100, c='red', zorder=10)  # Red circle for current best

                # Re-set labels and title for ax2 after clearing
                ax2.set_xlabel('迭代次数')
                ax2.set_ylabel('总距离 (km)')
                ax2.set_title('优化过程')
                ax2.grid(True, alpha=0.3)

                # Add artists from ax2 to the return list
                artists_to_return.extend(ax2.get_lines() + ax2.collections)

                return artists_to_return  # Return all artists that were modified or created

            # Create the animation
            # Limit the number of frames to avoid excessively long animations and high memory usage.
            # Select frames evenly spaced throughout the history.
            max_frames = 150  # Maximum number of frames in the GIF
            if len(steps) > max_frames:
                frame_indices = np.linspace(0, len(steps) - 1, max_frames, dtype=int)
            else:
                frame_indices = np.arange(len(steps))  # Use all steps if fewer than max_frames

            anim = FuncAnimation(fig, update, frames=frame_indices,
                                 interval=200, blit=False, repeat=True)  # blit=False is often more reliable

            # Save the animation as a GIF
            safe_name = algo_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')',
                                                                                               '')  # Sanitize algorithm name for filename
            gif_path = f"{output_dir}/search_animation_{safe_name}.gif"

            try:
                # Use PillowWriter for GIF export, it's generally more robust
                writer = PillowWriter(fps=5)  # Set frames per second
                anim.save(gif_path, writer=writer, dpi=100)  # Save with specified DPI
                print(f"  ✅ 已生成: search_animation_{safe_name}.gif")
            except Exception as e:
                print(f"  ⚠️  生成动画失败: {e}")

            plt.close(fig)  # Close the figure to free up memory after saving

    def export_report(self, filename: str = "performance_report.json"):
        """Export detailed performance results to a JSON file."""
        report = {
            'timestamp': datetime.now().isoformat(),  # Timestamp of report generation
            'algorithms': self.results,  # Dictionary of results for each algorithm
            'summary': {  # Summary of best algorithms for key metrics
                'best_distance': min(self.results.items(), key=lambda x: x[1]['total_distance'] if x[1][
                                                                                                       'total_distance'] is not None else float(
                    'inf'))[0] if self.results else None,
                'best_time': min(self.results.items(),
                                 key=lambda x: x[1]['total_time'] if x[1]['total_time'] is not None else float('inf'))[
                    0] if self.results else None,
                'best_cost': min(self.results.items(),
                                 key=lambda x: x[1]['total_cost'] if x[1]['total_cost'] is not None else float('inf'))[
                    0] if self.results else None,
                'best_emission': min(self.results.items(), key=lambda x: x[1]['carbon_emission'] if x[1][
                                                                                                        'carbon_emission'] is not None else float(
                    'inf'))[0] if self.results else None
            }
        }

        # Write the report to a JSON file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2,
                      ensure_ascii=False)  # Use indent for readability, ensure_ascii=False for Chinese characters

        print(f"\n📄 性能报告已导出: {filename}")

    def compare_algorithms(self) -> None:
        """Compare the performance of different algorithms."""
        if len(self.results) < 2:
            print("\n⚠️  需要至少2个算法结果才能进行对比")
            return

        print("\n" + "=" * 80)
        print("📊 算法性能对比分析")
        print("=" * 80)

        # Define metrics to display and their keys in the results dictionary
        metrics = ['总距离(km)', '总时间(h)', '总成本(¥)', '碳排放(kg)', '载重率(%)', '路线效率(%)']
        metric_keys = ['total_distance', 'total_time', 'total_cost', 'carbon_emission',
                       'capacity_utilization', 'route_efficiency']

        # Print header row for algorithms
        print(f"\n{'指标':<15}", end='')
        for algo_name in self.results.keys():
            print(f"{algo_name:>20}", end='')  # Align algorithm names
        print()
        # Print separator line
        print("-" * (15 + 20 * len(self.results)))

        # Print each metric's values for all algorithms
        for metric, key in zip(metrics, metric_keys):
            print(f"{metric:<15}", end='')
            for algo_name in self.results.keys():
                value = self.results[algo_name].get(key, 0)  # Use .get to handle potentially missing keys
                print(f"{value:>20.2f}", end='')  # Format value to 2 decimal places
            print()

        # Determine and print the best algorithm for key metrics
        print("\n" + "=" * 80)
        print("🏆 最优算法:")
        print("-" * 80)

        # Find best algorithm for each metric (handle cases where results might be empty or missing)
        best_distance_item = min(self.results.items(), key=lambda x: x[1]['total_distance']) if self.results else None
        best_time_item = min(self.results.items(), key=lambda x: x[1]['total_time']) if self.results else None
        best_cost_item = min(self.results.items(), key=lambda x: x[1]['total_cost']) if self.results else None
        best_emission_item = min(self.results.items(), key=lambda x: x[1]['carbon_emission']) if self.results else None

        if best_distance_item: print(
            f"  最短距离: {best_distance_item[0]} ({best_distance_item[1]['total_distance']:.2f} km)")
        if best_time_item: print(f"  最短时间: {best_time_item[0]} ({best_time_item[1]['total_time']:.2f} h)")
        if best_cost_item: print(f"  最低成本: {best_cost_item[0]} (¥{best_cost_item[1]['total_cost']:.2f})")
        if best_emission_item: print(
            f"  最低排放: {best_emission_item[0]} ({best_emission_item[1]['carbon_emission']:.2f} kg CO2)")

        # Calculate and print savings percentages relative to the worst-performing algorithm
        print("\n" + "=" * 80)
        print("💰 节约分析 (相对于最差算法):")
        print("-" * 80)

        worst_distance_item = max(self.results.items(), key=lambda x: x[1]['total_distance']) if self.results else None
        worst_cost_item = max(self.results.items(), key=lambda x: x[1]['total_cost']) if self.results else None

        # Calculate distance savings if data is available and worst distance is positive
        if worst_distance_item and best_distance_item and worst_distance_item[1]['total_distance'] > 0:
            distance_saving = ((worst_distance_item[1]['total_distance'] - best_distance_item[1]['total_distance'])
                               / worst_distance_item[1]['total_distance'] * 100)
            print(
                f"  距离节约: {distance_saving:.2f}% ({worst_distance_item[1]['total_distance'] - best_distance_item[1]['total_distance']:.2f} km)")

        # Calculate cost savings if data is available and worst cost is positive
        if worst_cost_item and best_cost_item and worst_cost_item[1]['total_cost'] > 0:
            cost_saving = ((worst_cost_item[1]['total_cost'] - best_cost_item[1]['total_cost'])
                           / worst_cost_item[1]['total_cost'] * 100)
            print(
                f"  成本节约: {cost_saving:.2f}% (¥{worst_cost_item[1]['total_cost'] - best_cost_item[1]['total_cost']:.2f})")


class InteractiveConfig:
    """Interactive configuration input utilities."""

    @staticmethod
    def get_int_input(prompt: str, default: int = None, min_val: int = None, max_val: int = None) -> int:
        """Get integer input from the user with validation."""
        while True:
            try:
                if default is not None:
                    value = input(f"{prompt} (默认: {default}): ").strip()
                    if not value:  # If user presses Enter, use default
                        return default
                else:
                    value = input(f"{prompt}: ").strip()  # No default value

                result = int(value)  # Convert input to integer

                # Check min/max value constraints
                if min_val is not None and result < min_val:
                    print(f"❌ 输入值不能小于 {min_val}")
                    continue
                if max_val is not None and result > max_val:
                    print(f"❌ 输入值不能大于 {max_val}")
                    continue

                return result  # Return valid integer
            except ValueError:
                print("❌ 请输入有效的整数!")

    @staticmethod
    def get_float_input(prompt: str, default: float = None, min_val: float = None, max_val: float = None) -> float:
        """Get float input from the user with validation."""
        while True:
            try:
                if default is not None:
                    value = input(f"{prompt} (默认: {default}): ").strip()
                    if not value:
                        return default
                else:
                    value = input(f"{prompt}: ").strip()

                result = float(value)  # Convert input to float

                # Check min/max value constraints
                if min_val is not None and result < min_val:
                    print(f"❌ 输入值不能小于 {min_val}")
                    continue
                if max_val is not None and result > max_val:
                    print(f"❌ 输入值不能大于 {max_val}")
                    continue

                return result  # Return valid float
            except ValueError:
                print("❌ 请输入有效的数字!")

    @staticmethod
    def get_choice(prompt: str, choices: List[str]) -> str:
        """Get a choice from a list of options."""
        print(f"\n{prompt}")
        # Display choices with numbered options
        for idx, choice in enumerate(choices, 1):
            print(f"  {idx}. {choice}")

        while True:
            try:
                value = input(f"请选择 (1-{len(choices)}): ").strip()
                choice_idx = int(value) - 1  # Convert input to 0-based index

                # Validate choice index
                if 0 <= choice_idx < len(choices):
                    return choices[choice_idx]  # Return the selected choice
                else:
                    print(f"❌ 请输入 1 到 {len(choices)} 之间的数字!")
            except ValueError:
                print("❌ 请输入有效的数字!")

    @staticmethod
    def get_yes_no(prompt: str, default: bool = True) -> bool:
        """Get a yes/no answer from the user."""
        default_str = "Y/n" if default else "y/N"  # Show default option in prompt
        value = input(f"{prompt} ({default_str}): ").strip().lower()

        if not value:  # If input is empty, return default
            return default

        return value in ['y', 'yes', '是']  # True for affirmative answers

    @staticmethod
    def get_coordinates(prompt: str) -> Tuple[float, float]:
        """Get latitude and longitude coordinates from the user."""
        print(f"\n{prompt}")
        lat = InteractiveConfig.get_float_input("  纬度", min_val=-90, max_val=90)  # Validate latitude
        lon = InteractiveConfig.get_float_input("  经度", min_val=-180, max_val=180)  # Validate longitude
        return (lat, lon)

    @staticmethod
    def load_delivery_points_from_csv(file_path: str) -> List[DeliveryPoint]:
        """Load delivery points from a CSV file."""
        delivery_points = []
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                delivery_points.append(DeliveryPoint(
                    id=row['id'],
                    location=(float(row['latitude']), float(row['longitude'])),
                    demand=float(row['demand']),
                    # Parse datetime strings for time windows
                    time_window=(datetime.strptime(row['start_time'], '%Y-%m-%d %H:%M:%S'),
                                 datetime.strptime(row['end_time'], '%Y-%m-%d %H:%M:%S')),
                    service_time=int(row['service_time']),
                    priority=int(row['priority'])
                ))
        return delivery_points

    @staticmethod
    def load_delivery_points_from_json(file_path: str) -> List[DeliveryPoint]:
        """Load delivery points from a JSON file."""
        delivery_points = []
        with open(file_path, 'r') as jsonfile:
            data = json.load(jsonfile)
            # Parse data from each item in the JSON array
            for item in data:
                delivery_points.append(DeliveryPoint(
                    id=item['id'],
                    location=(item['latitude'], item['longitude']),
                    demand=item['demand'],
                    # Parse datetime strings for time windows
                    time_window=(datetime.strptime(item['start_time'], '%Y-%m-%d %H:%M:%S'),
                                 datetime.strptime(item['end_time'], '%Y-%m-%d %H:%M:%S')),
                    service_time=item['service_time'],
                    priority=item['priority']
                ))
        return delivery_points


class FileImporter:
    """File importer utilities for network and delivery data."""

    @staticmethod
    def import_from_csv(file_path: str) -> Dict:
        """
        Import network configuration from CSV files located in a directory.

        Expected CSV files:
        1. nodes.csv: id, latitude, longitude, type (warehouse/delivery/transit)
        2. edges.csv: from_node, to_node, distance, min_time, max_time
        3. deliveries.csv: id, demand, service_time, priority, time_window_start, time_window_end
        """
        config = {
            'nodes': {},  # Stores node ID to {lat, lon, type} mapping
            'edges': [],  # List of edge dictionaries
            'deliveries': []  # List of delivery point dictionaries
        }

        # Import nodes.csv
        nodes_file = os.path.join(file_path, 'nodes.csv')  # Construct path to nodes.csv
        if os.path.exists(nodes_file):
            with open(nodes_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    config['nodes'][row['id']] = {
                        'latitude': float(row['latitude']),
                        'longitude': float(row['longitude']),
                        'type': row.get('type', 'delivery')  # Default type to 'delivery' if not specified
                    }
            print(f"✅ 成功导入 {len(config['nodes'])} 个节点")
        else:
            print(f"⚠️  未找到 nodes.csv 文件: {nodes_file}")

        # Import edges.csv
        edges_file = os.path.join(file_path, 'edges.csv')  # Construct path to edges.csv
        if os.path.exists(edges_file):
            with open(edges_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    config['edges'].append({
                        'from': row['from_node'],
                        'to': row['to_node'],
                        'distance': float(row['distance']),
                        'min_time': float(row['min_time']),  # Minimum travel time
                        'max_time': float(row['max_time'])  # Maximum travel time (e.g., with congestion)
                    })
            print(f"✅ 成功导入 {len(config['edges'])} 条边")
        else:
            print(f"⚠️  未找到 edges.csv 文件: {edges_file}")

        # Import deliveries.csv
        deliveries_file = os.path.join(file_path, 'deliveries.csv')  # Construct path to deliveries.csv
        if os.path.exists(deliveries_file):
            with open(deliveries_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    config['deliveries'].append({
                        'point_id': row['id'],  # ID of the delivery point
                        'demand': float(row['demand']),
                        'service_time': float(row['service_time']),
                        'priority': row.get('priority', 'medium'),  # Default priority to 'medium'
                        'time_window_start': row.get('time_window_start', '08:00'),  # Default start time
                        'time_window_end': row.get('time_window_end', '18:00')  # Default end time
                    })
            print(f"✅ 成功导入 {len(config['deliveries'])} 个配送点")
        else:
            print(f"⚠️  未找到 deliveries.csv 文件: {deliveries_file}")

        return config

    @staticmethod
    def import_distance_matrix(file_path: str) -> tuple:
        """
        Import a distance matrix from a CSV file.
        CSV format: First row and first column are node IDs.
        Values are distances (float). Empty cells can be treated as infinity.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            tuple: (distance_matrix (np.ndarray), node_ids (List[str]))
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)  # Read all rows into a list

        # Extract node IDs from the first row (skipping the first element, which is blank or header)
        node_ids = rows[0][1:]

        # Build the distance matrix from the remaining rows
        matrix = []
        for row in rows[1:]:
            # Convert values to float. Treat empty strings as float('inf').
            matrix.append([float(x) if x else float('inf') for x in row[1:]])

        print(f"✅ 导入距离矩阵: {len(node_ids)} x {len(node_ids)} 节点")
        return np.array(matrix), node_ids  # Return matrix and node IDs as NumPy array and list

    @staticmethod
    def import_delivery_config(file_path: str) -> list:
        """
        Import delivery point configuration from a CSV file.
        CSV format: id, demand, priority, service_time, time_window_start, time_window_end

        Args:
            file_path (str): Path to the delivery configuration CSV file.

        Returns:
            list: A list of dictionaries, each representing a delivery point's configuration.
        """
        deliveries = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                deliveries.append({
                    'point_id': row['id'],
                    'demand': float(row['demand']),
                    'priority': row.get('priority', 'medium'),  # Default to 'medium' if missing
                    'service_time': float(row['service_time']),
                    'time_window_start': row.get('time_window_start', '08:00'),  # Default start time
                    'time_window_end': row.get('time_window_end', '18:00')  # Default end time
                })

        print(f"✅ 导入配送点配置: {len(deliveries)} 个配送点")
        return deliveries

    @staticmethod
    def export_template_csv(output_dir: str):
        """Export template CSV files for network and delivery data."""
        os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

        # Export nodes.csv template
        with open(os.path.join(output_dir, 'nodes.csv'), 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'latitude', 'longitude', 'type'])  # Header row
            # Example data
            writer.writerow(['warehouse', '23.1291', '113.2644', 'warehouse'])
            writer.writerow(['point_a', '23.1350', '113.2700', 'delivery'])
            writer.writerow(['point_b', '23.1200', '113.2500', 'delivery'])

        # Export edges.csv template
        with open(os.path.join(output_dir, 'edges.csv'), 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['from_node', 'to_node', 'distance', 'min_time', 'max_time'])  # Header
            # Example data
            writer.writerow(['warehouse', 'point_a', '5.2', '8', '12'])
            writer.writerow(['warehouse', 'point_b', '7.5', '12', '18'])

        # Export deliveries.csv template
        with open(os.path.join(output_dir, 'deliveries.csv'), 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['id', 'demand', 'service_time', 'priority', 'time_window_start', 'time_window_end'])  # Header
            # Example data
            writer.writerow(['point_a', '50', '15', 'high', '09:00', '12:00'])
            writer.writerow(['point_b', '30', '10', 'medium', '10:00', '14:00'])

        # Export distance_matrix.csv template
        with open(os.path.join(output_dir, 'distance_matrix.csv'), 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # Header row with node IDs
            writer.writerow(['', 'warehouse', 'point_a', 'point_b', 'point_c'])
            # Data rows, first element is node ID, then distances
            writer.writerow(['warehouse', '0', '5.2', '7.5', '9.8'])
            writer.writerow(['point_a', '5.2', '0', '6.1', '4.1'])
            writer.writerow(['point_b', '7.5', '6.1', '0', '3.8'])
            writer.writerow(['point_c', '9.8', '4.1', '3.8', '0'])

        print(f"✅ CSV模板已导出到: {output_dir}")


# 在 run_optimization 函数开始处添加诊断
def run_optimization(router, delivery_points, vehicle, warehouse_id):
    """执行路径优化（增加诊断版）"""
    print("\n" + "=" * 60)
    print("开始路径优化")
    print("=" * 60 + "\n")

    # ===== 添加预检查诊断 =====
    print("🔍 执行配置检查...")

    # 检查1: 配送点是否为空
    if not delivery_points:
        print("❌ 错误：没有配送点！")
        return

    # 检查2: 车辆参数是否合理
    if vehicle.capacity <= 0:
        print(f"❌ 错误：车辆容量无效 ({vehicle.capacity})")
        return

    if vehicle.max_distance <= 0:
        print(f"❌ 错误：最大行驶距离无效 ({vehicle.max_distance})")
        return

    if vehicle.speed <= 0:
        print(f"❌ 错误：车辆速度无效 ({vehicle.speed})")
        return

    # 检查3: 基本可行性
    print("\n📊 配置概览:")
    print(f"  配送点数量: {len(delivery_points)}")
    print(f"  车辆容量: {vehicle.capacity} kg")
    print(f"  最大距离: {vehicle.max_distance} km")
    print(f"  平均速度: {vehicle.speed} km/h")

    total_demand = sum(p.demand for p in delivery_points)
    max_demand = max(p.demand for p in delivery_points)

    print(f"  总需求: {total_demand:.1f} kg")
    print(f"  最大单点需求: {max_demand:.1f} kg")

    # 检查4: 识别潜在问题
    issues = []
    warnings = []

    if max_demand > vehicle.capacity:
        issues.append(f"有配送点需求({max_demand}kg)超过车辆容量({vehicle.capacity}kg)")

    if total_demand > vehicle.capacity:
        warnings.append(f"总需求({total_demand:.1f}kg)超过容量，无法一次完成所有配送")

    # 计算距离
    from math import radians, sin, cos, sqrt, atan2
    def calc_dist(loc1, loc2):
        lat1, lon1 = loc1
        lat2, lon2 = loc2
        R = 6371
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    max_dist_from_start = max(calc_dist(vehicle.start_location, p.location)
                              for p in delivery_points)

    print(f"  最远配送点: {max_dist_from_start:.2f} km")

    if max_dist_from_start * 2 > vehicle.max_distance:
        issues.append(f"最远点往返({max_dist_from_start * 2:.1f}km)超过最大距离({vehicle.max_distance}km)")

    # 时间窗检查
    now = datetime.now()
    expired_points = [p for p in delivery_points if now > p.time_window[1]]
    if expired_points:
        warnings.append(f"{len(expired_points)}个配送点的时间窗已过期")

    # 显示问题
    if issues:
        print("\n❌ 发现严重问题:")
        for issue in issues:
            print(f"  • {issue}")

        # 提供自动修复选项
        print("\n🔧 建议的修复方案:")
        if max_demand > vehicle.capacity:
            print(f"  1. 将车辆容量调整为: {max_demand * 1.2:.0f} kg")
        if max_dist_from_start * 2 > vehicle.max_distance:
            print(f"  2. 将最大行驶距离调整为: {max_dist_from_start * 3:.0f} km")

        auto_fix = input("\n是否自动应用修复方案? (y/n, 默认n): ").strip().lower()

        if auto_fix == 'y':
            if max_demand > vehicle.capacity:
                old_capacity = vehicle.capacity
                vehicle.capacity = max_demand * 1.2
                print(f"✅ 容量已调整: {old_capacity} → {vehicle.capacity:.0f} kg")

            if max_dist_from_start * 2 > vehicle.max_distance:
                old_distance = vehicle.max_distance
                vehicle.max_distance = max_dist_from_start * 3
                print(f"✅ 最大距离已调整: {old_distance} → {vehicle.max_distance:.0f} km")

    if warnings:
        print("\n⚠️  警告:")
        for warning in warnings:
            print(f"  • {warning}")

    if not issues:
        print("\n✅ 配置检查通过！")

    # 继续原有的优化流程
    print("\n" + "=" * 60)

    analyzer = PerformanceAnalyzer()

    algorithm_choice = InteractiveConfig.get_choice(
        "选择要运行的优化算法",
        ["最近邻算法 (NN)", "遗传算法 (GA)", "多目标优化 (MO)", "全部运行对比"]
    )

    all_algorithms_selected = (algorithm_choice == "全部运行对比")
    run_nn = algorithm_choice == "最近邻算法 (NN)" or all_algorithms_selected
    run_ga = algorithm_choice == "遗传算法 (GA)" or all_algorithms_selected
    run_mo = algorithm_choice == "多目标优化 (MO)" or all_algorithms_selected

    if not hasattr(vehicle, 'start_location_id') or vehicle.start_location_id is None:
        vehicle.start_location_id = warehouse_id

    if run_nn:
        print("\n=== 运行: 最近邻算法 ===")
        start_time = datetime.now()
        vrp_solver = VRPTWSolver([vehicle], delivery_points)
        nn_route = vrp_solver.nearest_neighbor_heuristic(vehicle)
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # 只有在路径非空时才添加结果
        if nn_route:
            analyzer.add_result(
                algorithm_name="最近邻算法",
                route=nn_route,
                vehicle=vehicle,
                router=router,
                execution_time=execution_time
            )
        else:
            print("\n⚠️  最近邻算法未生成有效路径，跳过结果记录")

    if run_ga:
        print("\n=== 运行: 遗传算法 ===")
        start_time = datetime.now()
        ga_solver = GeneticAlgorithm(
            points=delivery_points,
            router=router,
            population_size=50,
            generations=100,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        ga_route = ga_solver.evolve(vehicle)
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        if ga_route:
            analyzer.add_result(
                algorithm_name="遗传算法",
                route=ga_route,
                vehicle=vehicle,
                router=router,
                execution_time=execution_time,
                search_steps=ga_solver.search_steps
            )
        else:
            print("\n⚠️  遗传算法未生成有效路径，跳过结果记录")

    if run_mo:
        print("\n=== 运行: 多目标优化 ===")
        start_time = datetime.now()
        mo_optimizer = MultiObjectiveOptimizer(
            weight_cost=0.5,
            weight_time=0.3,
            weight_emission=0.2
        )
        mo_route = mo_optimizer.optimize_route(delivery_points, vehicle)
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        if mo_route:
            analyzer.add_result(
                algorithm_name="多目标优化",
                route=mo_route,
                vehicle=vehicle,
                router=router,
                execution_time=execution_time
            )
        else:
            print("\n⚠️  多目标优化未生成有效路径，跳过结果记录")

    if len(analyzer.results) > 0:
        if all_algorithms_selected:
            analyzer.compare_algorithms()

        if InteractiveConfig.get_yes_no("是否生成可视化图表", default=True):
            analyzer.visualize_results()

        if InteractiveConfig.get_yes_no("是否导出性能报告 (JSON)", default=True):
            analyzer.export_report()
    else:
        print("\n⚠️  所有算法均未生成有效路径。")
        print("请检查配置参数或使用自动修复功能。")


def main(points=None):
    """Default configuration demo (retains original functionality)"""
    print("\n🔄 使用默认配置运行演示...\n")

    # 1. Build road network using AStarRouter
    router = AStarRouter()

    # Define node locations (latitude, longitude)
    locations = {
        'warehouse': (23.1291, 113.2644),  # Guangzhou warehouse coordinates
        'point_a': (23.1350, 113.2700),
        'point_b': (23.1200, 113.2500),
        'point_c': (23.1400, 113.2800),
        'point_d': (23.1100, 113.2600)
    }

    # Add node coordinates to the router
    for node, coords in locations.items():
        router.add_node_coordinates(node, coords[0], coords[1])

    # Add road network edges with distance, time, and cost
    router.add_edge('warehouse', 'point_a', 5.2, 8, 12)
    router.add_edge('warehouse', 'point_b', 7.5, 12, 18)
    router.add_edge('point_a', 'point_c', 4.1, 6, 10)
    router.add_edge('point_b', 'point_d', 3.8, 5, 9)
    router.add_edge('point_c', 'point_d', 2.5, 4, 7)

    # 2. Create delivery points
    if points is None:  # If no points are provided externally, create default ones
        now = datetime.now()  # Get current time
        delivery_points = [
            DeliveryPoint(
                id='point_a',
                location=locations['point_a'],
                demand=50,  # Demand in kg
                # Time windows relative to current time
                time_window=(now + timedelta(hours=1), now + timedelta(hours=3)),
                service_time=15,  # Service time in minutes
                priority=1  # Priority level
            ),
            DeliveryPoint(
                id='point_b',
                location=locations['point_b'],
                demand=30,
                time_window=(now + timedelta(hours=2), now + timedelta(hours=4)),
                service_time=10,
                priority=2
            ),
            DeliveryPoint(
                id='point_c',
                location=locations['point_c'],
                demand=40,
                time_window=(now + timedelta(hours=1), now + timedelta(hours=2)),
                service_time=20,
                priority=1
            )
        ]
    else:  # Use provided delivery points
        delivery_points = points

    # 3. Create vehicle object
    warehouse_id = 'warehouse'  # ID of the warehouse node
    vehicle = Vehicle(
        id='truck_001',
        capacity=150,  # Vehicle capacity in kg
        start_location=locations[warehouse_id],  # Start location coordinates
        max_distance=100,  # Maximum travel distance in km
        speed=40,  # Average speed in km/h
        start_location_id=warehouse_id  # Set start_location_id attribute
    )

    # 4. Perform basic path planning using A*
    print("=== Basic Path Planning ===")
    # Find shortest path from warehouse to point_c using distance as weight
    path, distance = router.a_star_search('warehouse', 'point_c', 'distance')
    print(f"Shortest path: {' -> '.join(path)}")
    print(f"Total distance: {distance:.2f} km\n")

    # 5. VRP Optimization using Nearest Neighbor heuristic
    print("=== VRP Path Optimization (Nearest Neighbor) ===")
    vrp_solver = VRPTWSolver([vehicle], delivery_points)  # Initialize VRP solver
    nn_route = vrp_solver.nearest_neighbor_heuristic(vehicle)  # Get NN route

    print(f"Optimized delivery sequence (NN):")
    # Print the optimized sequence and details
    for idx, point in enumerate(nn_route, 1):
        print(f"  {idx}. {point.id} - Demand:{point.demand}kg, Service Time:{point.service_time}min")

    # 6. Genetic Algorithm Optimization
    print("\n=== Genetic Algorithm Optimization ===")
    ga_solver = GeneticAlgorithm(
        points=delivery_points,  # Pass delivery points
        router=router,
        population_size=50,
        generations=100
    )
    ga_route = ga_solver.evolve(vehicle)  # Evolve the route using GA, passing vehicle

    print(f"GA optimized path:")
    # Print the GA optimized route IDs
    for idx, point in enumerate(ga_route, 1):
        print(f"  {idx}. {point.id}")

    # 7. Multi-objective Optimization
    print("\n=== Multi-objective Optimization ===")
    mo_optimizer = MultiObjectiveOptimizer(
        weight_cost=0.5,  # Weights for objectives
        weight_time=0.3,
        weight_emission=0.2
    )
    mo_route = mo_optimizer.optimize_route(delivery_points, vehicle)  # Optimize route

    # Calculate and print objectives for the multi-objective optimized route
    cost = mo_optimizer._calculate_cost(mo_route, vehicle)
    time = mo_optimizer._calculate_time(mo_route, vehicle)
    emission = mo_optimizer._calculate_emission(mo_route, vehicle)

    print(f"Multi-objective optimized route:")
    for idx, point in enumerate(mo_route, 1):
        print(f"  {idx}. {point.id}")
    print(f"Total cost: ¥{cost:.2f}")
    print(f"Total time: {time:.2f} hours")
    print(f"Carbon emissions: {emission:.2f}kg CO2")


def main_with_interactive_config():
    """Handles interactive configuration and runs optimization."""
    print("\n" + "=" * 60)
    print("交互式配置模式")
    print("=" * 60 + "\n")

    # 1. Set up the road network
    router = AStarRouter()
    print("--- 路网配置 ---")
    num_nodes = InteractiveConfig.get_int_input("请输入节点数量", default=5, min_val=2)

    nodes_data = {}  # Store node details {id: {lat, lon, type}}
    delivery_points_list = []  # List of DeliveryPoint objects
    warehouse_id = None
    warehouse_location = None

    # Configure nodes interactively
    for i in range(num_nodes):
        node_id = input(f"请输入节点 {i + 1} 的ID (例如: warehouse, P001): ").strip()
        if not node_id:
            print("❌ 节点ID不能为空")
            continue

        # Check if node ID is already in use
        if node_id in nodes_data:
            print(f"⚠️  节点ID '{node_id}' 已存在，请重新输入。")
            continue

        lat, lon = InteractiveConfig.get_coordinates("请输入节点坐标")
        node_type = InteractiveConfig.get_choice("节点类型", ["warehouse", "delivery", "transit"]).lower()

        router.add_node_coordinates(node_id, lat, lon)  # Add coordinates to router
        nodes_data[node_id] = {'latitude': lat, 'longitude': lon, 'type': node_type}

        # If it's a warehouse, store its details
        if node_type == 'warehouse':
            if warehouse_id:  # Warn if multiple warehouses are defined
                print("⚠️  已定义多个仓库节点。将使用最后一个定义的作为主仓库。")
            warehouse_id = node_id
            warehouse_location = (lat, lon)

        # If it's a delivery node, prompt for delivery details
        elif node_type == 'delivery':
            demand = InteractiveConfig.get_float_input("  需求量 (kg)", default=50.0, min_val=0)
            service_time = InteractiveConfig.get_float_input("  服务时间 (分钟)", default=15.0, min_val=0)
            priority_str = InteractiveConfig.get_choice("  优先级", ["high", "medium", "low"])
            priority_map = {'high': 1, 'medium': 2, 'low': 3}
            priority = priority_map[priority_str]

            # Get time window interactively
            print("  请输入时间窗:")
            start_hour = InteractiveConfig.get_int_input("    开始小时 (0-23)", default=8, min_val=0, max_val=23)
            start_minute = InteractiveConfig.get_int_input("    开始分钟 (0-59)", default=0, min_val=0, max_val=59)
            end_hour = InteractiveConfig.get_int_input("    结束小时 (0-23)", default=18, min_val=0, max_val=23)
            end_minute = InteractiveConfig.get_int_input("    结束分钟 (0-59)", default=0, min_val=0, max_val=59)

            now = datetime.now()
            tw_start = now.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
            tw_end = now.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)

            # Adjust time window if end is before start (e.g., overnight)
            if tw_end < tw_start:
                tw_end += timedelta(days=1)

            time_window = (tw_start, tw_end)

            delivery_points_list.append(DeliveryPoint(
                id=node_id,
                location=(lat, lon),
                demand=demand,
                time_window=time_window,
                service_time=int(service_time),
                priority=priority
            ))

    # Check if a warehouse was defined
    if not warehouse_id:
        print("❌ 错误: 未定义仓库节点。请至少定义一个类型为 'warehouse' 的节点。")
        return

    # Configure edges interactively
    print("\n--- 边配置 ---")
    num_edges = InteractiveConfig.get_int_input("请输入边 (路段) 数量",
                                                default=min(len(nodes_data) * (len(nodes_data) - 1), 5), min_val=0)

    edge_configs = []  # Store edge configurations
    for i in range(num_edges):
        # Get source and destination nodes from defined nodes
        from_node = InteractiveConfig.get_choice(f"边 {i + 1}:请输入起点节点", list(nodes_data.keys()))
        # Filter out the from_node from choices for the to_node to avoid self-loops
        to_node_choices = [node for node in nodes_data.keys() if node != from_node]
        if not to_node_choices:
            print("❌ 无法添加边，因为没有其他节点可作为终点。")
            break
        to_node = InteractiveConfig.get_choice(f"边 {i + 1}:请输入终点节点", to_node_choices)

        # Get edge attributes
        distance = InteractiveConfig.get_float_input("  距离 (km)", min_val=0)
        min_time = InteractiveConfig.get_float_input("  最短时间 (小时)", min_val=0)
        max_time = InteractiveConfig.get_float_input("  最长时间 (小时)", min_val=min_time)

        # Add edge to router
        router.add_edge(from_node, to_node, distance, min_time, max_time)
        edge_configs.append(
            {'from': from_node, 'to': to_node, 'distance': distance, 'min_time': min_time, 'max_time': max_time})

    # 2. Configure Vehicle
    print("\n--- 车辆配置 ---")
    vehicle_capacity = InteractiveConfig.get_float_input("车辆载重 (kg)", default=200.0, min_val=0)
    vehicle_max_distance = InteractiveConfig.get_float_input("车辆最大行驶距离 (km)", default=150.0, min_val=0)
    vehicle_speed = InteractiveConfig.get_float_input("车辆平均速度 (km/h)", default=40.0, min_val=1)

    vehicle = Vehicle(
        id='vehicle_001',
        capacity=vehicle_capacity,
        start_location=warehouse_location,  # Use warehouse location
        max_distance=vehicle_max_distance,
        speed=vehicle_speed,
        start_location_id=warehouse_id  # Set start_location_id
    )

    # Run optimization
    run_optimization(router, delivery_points_list, vehicle, warehouse_id)


def validate_and_import_csv(csv_dir):
    """验证并导入CSV数据"""
    config = FileImporter.import_from_csv(csv_dir)

    # 验证节点坐标
    valid_nodes = {}
    invalid_count = 0
    for node_id, node_data in config['nodes'].items():
        lat = node_data['latitude']
        lon = node_data['longitude']

        # 检查坐标有效性
        if (isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and
                not np.isnan(lat) and not np.isnan(lon) and
                not np.isinf(lat) and not np.isinf(lon) and
                -90 <= lat <= 90 and -180 <= lon <= 180):
            valid_nodes[node_id] = node_data
        else:
            invalid_count += 1
            print(f"⚠️ 跳过无效坐标的节点: {node_id} ({lat}, {lon})")

    config['nodes'] = valid_nodes

    if invalid_count > 0:
        print(f"⚠️ 共有 {invalid_count} 个节点坐标无效，已跳过")

    # 过滤边：只保留两端节点都有效的边
    valid_edges = []
    for edge in config['edges']:
        if edge['from'] in valid_nodes and edge['to'] in valid_nodes:
            valid_edges.append(edge)

    config['edges'] = valid_edges

    # 过滤配送点：只保留在有效节点中的配送点
    valid_deliveries = []
    for delivery in config['deliveries']:
        if delivery['point_id'] in valid_nodes:
            valid_deliveries.append(delivery)

    config['deliveries'] = valid_deliveries

    print(f"\n✅ 验证后的数据:")
    print(f"   - 有效节点数: {len(config['nodes'])}")
    print(f"   - 有效边数: {len(config['edges'])}")
    print(f"   - 有效配送点数: {len(config['deliveries'])}")

    return config

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚚 物流路径优化系统")
    print("=" * 60 + "\n")

    # Present options to the user for running the system
    print("请选择运行模式:")
    print("  1. 交互式手动配置 (输入参数和数据)")
    print("  2. 从CSV文件导入完整网络数据 (nodes.csv + edges.csv + deliveries.csv)")
    print("  3. 从距离矩阵导入 (distance_matrix.csv) + 配送点配置")
    print("  4. 使用默认配置演示 (快速运行示例)")
    print("  5. 导出CSV模板文件 (用于选项2)")

    choice = input("\n请输入选项 (1/2/3/4/5, 默认: 1): ").strip()

    # --- Option 2: Import from CSV files ---
    if choice == "2":
        print("\n📂 从CSV文件导入网络数据")
        print("需要的文件: nodes.csv, edges.csv, deliveries.csv")
        csv_dir = input("请输入CSV文件所在目录路径 (例如: example_data): ").strip()

        if not csv_dir:
            print("❌ 未输入目录路径")
        elif not os.path.exists(csv_dir):
            print(f"❌ 目录不存在: {csv_dir}")
        else:
            try:

                # 使用新的验证导入函数
                config = validate_and_import_csv(csv_dir)

                if len(config['nodes']) == 0:
                    print("❌ 没有有效的节点数据，无法继续")
                elif len(config['deliveries']) == 0:
                    print("❌ 没有有效的配送点数据，无法继续")
                else:
                    # 初始化路由器
                    router = AStarRouter()
                    for node_id, node_data in config['nodes'].items():
                        router.add_node_coordinates(
                            node_id,
                            node_data['latitude'],
                            node_data['longitude']
                        )

                    for edge in config['edges']:
                        router.add_edge(
                            edge['from'],
                            edge['to'],
                            edge['distance'],
                            edge['min_time'],
                            edge['max_time']
                        )

                    # 处理配送点
                    delivery_points = []
                    warehouse_location = None
                    warehouse_id = None

                    # 查找仓库
                    for node_id, node_data in config['nodes'].items():
                        if node_data.get('type') == 'warehouse':
                            warehouse_location = (node_data['latitude'], node_data['longitude'])
                            warehouse_id = node_id
                            print(f"✅ 找到仓库节点: {warehouse_id}")
                            break

                    if not warehouse_location:
                        print("❌ 未找到仓库节点 (type='warehouse')")
                    else:
                        # 创建配送点对象
                        now = datetime.now()
                        for delivery in config['deliveries']:
                            point_id = delivery['point_id']
                            if point_id in config['nodes']:
                                node = config['nodes'][point_id]

                                # 解析时间窗
                                now = datetime.now()
                                try:
                                    if ' ' in delivery['time_window_start']:
                                        time_window_start_dt = parser.parse(delivery['time_window_start'])
                                    else:
                                        today = datetime.today().date()
                                        t = datetime.strptime(delivery['time_window_start'], '%H:%M').time()
                                        time_window_start_dt = datetime.combine(today, t)

                                    if ' ' in delivery['time_window_end']:
                                        time_window_end_dt = parser.parse(delivery['time_window_end'])
                                    else:
                                        today = datetime.today().date()
                                        t = datetime.strptime(delivery['time_window_end'], '%H:%M').time()
                                        time_window_end_dt = datetime.combine(today, t)

                                    time_window = (
                                        now.replace(hour=time_window_start_dt.hour, minute=time_window_start_dt.minute,
                                                    second=0, microsecond=0) if ' ' not in delivery[
                                            'time_window_start'] else time_window_start_dt,
                                        now.replace(hour=time_window_end_dt.hour, minute=time_window_end_dt.minute,
                                                    second=0, microsecond=0) if ' ' not in delivery[
                                            'time_window_end'] else time_window_end_dt
                                    )
                                    if time_window[1] < time_window[0]:  # Adjust if time window spans midnight
                                        time_window = (time_window[0], time_window[1] + timedelta(days=1))
                                except ValueError:
                                    print(
                                        f"⚠️  Invalid time window format for {point_id}: {delivery['time_window_start']} - {delivery['time_window_end']}. Using default (08:00-18:00).")
                                    time_window = (now.replace(hour=8, minute=0), now.replace(hour=18, minute=0))

                                priority_map = {'high': 1, 'medium': 2, 'low': 3}
                                priority = priority_map.get(delivery.get('priority', 'medium'), 2)

                                delivery_points.append(DeliveryPoint(
                                    id=point_id,
                                    location=(node['latitude'], node['longitude']),
                                    demand=delivery['demand'],
                                    time_window=time_window,
                                    service_time=int(delivery['service_time']),
                                    priority=priority
                                ))

                        if not delivery_points:
                            print("❌ 没有有效的配送点，无法继续")
                        else:
                            print(f"✅ 成功创建 {len(delivery_points)} 个配送点")

                            # 获取车辆参数
                            vehicle_capacity = InteractiveConfig.get_float_input("车辆载重 (kg)", 200.0)
                            vehicle_max_distance = InteractiveConfig.get_float_input("车辆最大行驶距离 (km)", 150.0)
                            vehicle_speed = InteractiveConfig.get_float_input("车辆平均速度 (km/h)", 40.0)

                            vehicle = Vehicle(
                                id='vehicle_001',
                                capacity=vehicle_capacity,
                                start_location=warehouse_location,
                                max_distance=vehicle_max_distance,
                                speed=vehicle_speed,
                                start_location_id=warehouse_id
                            )

                            # 执行优化
                            run_optimization(router, delivery_points, vehicle, warehouse_id)

            except Exception as e:
                print(f"❌ 处理失败: {e}")
                import traceback

                traceback.print_exc()

    # --- Option 3: Import from distance matrix ---
    elif choice == "3":
        print("\n📊 从距离矩阵导入")
        matrix_path = input("请输入距离矩阵文件路径 (例如: example_data/simple_distance_matrix.csv): ").strip()

        if not matrix_path:
            print("❌ 未输入文件路径")
        elif not os.path.exists(matrix_path):
            print(f"❌ 文件不存在: {matrix_path}")
        else:
            try:
                # Import distance matrix and node IDs
                distance_matrix, node_ids = FileImporter.import_distance_matrix(matrix_path)
                print(f"\n✅ 成功导入距离矩阵 ({len(node_ids)}个节点)")
                print(f"   节点列表: {', '.join(node_ids)}")

                # Check if a delivery configuration file exists
                has_delivery_config = input("\n是否有配送点配置文件 (deliveries.csv)? (y/n, 默认n): ").strip().lower()

                # Prepare config dictionary
                config = {
                    'nodes': {},  # Node data (dummy coords if not provided)
                    'edges': [],  # Edges derived from distance matrix
                    'deliveries': []  # Delivery point configurations
                }

                # Populate nodes with dummy coordinates if they are not provided elsewhere
                for i, node_id in enumerate(node_ids):
                    config['nodes'][node_id] = {
                        'latitude': 23.0 + i * 0.01,  # Dummy latitude based on index
                        'longitude': 113.0 + i * 0.01,  # Dummy longitude based on index
                        # Infer type: assume warehouse if ID starts with 'w' or contains 'warehouse'
                        'type': 'warehouse' if node_id.lower().startswith(
                            'w') or 'warehouse' in node_id.lower() else 'delivery'
                    }

                # Generate edges from the distance matrix
                for i, from_node in enumerate(node_ids):
                    for j, to_node in enumerate(node_ids):
                        # If distance is finite and not a self-loop
                        if i != j and distance_matrix[i][j] < float('inf'):
                            config['edges'].append({
                                'from': from_node,
                                'to': to_node,
                                'distance': distance_matrix[i][j],
                                # Placeholder times: Assume time is roughly proportional to distance
                                'min_time': distance_matrix[i][j] * 1.5,  # Example: 1.5 min/km
                                'max_time': distance_matrix[i][j] * 2.5  # Example: 2.5 min/km
                            })

                # Import delivery configuration if specified
                if has_delivery_config == 'y':
                    delivery_config_path = input("请输入配送点配置文件 (deliveries.csv) 路径: ").strip()
                    if os.path.exists(delivery_config_path):
                        config['deliveries'] = FileImporter.import_delivery_config(delivery_config_path)
                        print(f"✅ 成功加载 {len(config['deliveries'])} 个配送点配置")
                    else:
                        print("❌ 文件不存在，使用默认配送点配置。")
                        has_delivery_config = 'n'  # Reset flag if file not found

                # If no delivery config file provided or found, use default settings
                if has_delivery_config != 'y':
                    print("\n使用默认配送点配置...")
                    for node_id in node_ids:
                        # Add default delivery config for nodes that are not warehouses
                        if config['nodes'][node_id]['type'] != 'warehouse':
                            config['deliveries'].append({
                                'point_id': node_id,
                                'demand': 50.0,  # Default demand
                                'priority': 'medium',
                                'service_time': 15.0,  # Default service time
                                'time_window_start': '08:00',  # Default start time
                                'time_window_end': '18:00'  # Default end time
                            })

                print("\n✅ 配置数据已准备就绪。")
                print(f"\n✅ 成功导入:")
                print(f"   - 节点数: {len(config['nodes'])}")
                print(f"   - 边数: {len(config['edges'])}")
                print(f"   - 配送点数: {len(config['deliveries'])}")

                # Initialize router with imported nodes and edges
                router = AStarRouter()
                for node_id, node_data in config['nodes'].items():
                    router.add_node_coordinates(
                        node_id,
                        node_data['latitude'],
                        node_data['longitude']
                    )

                for edge in config['edges']:
                    router.add_edge(
                        edge['from'],
                        edge['to'],
                        edge['distance'],
                        edge['min_time'],
                        edge['max_time']
                    )

                # Prepare delivery points list
                delivery_points = []
                warehouse_location = None
                warehouse_id = None

                # Find the warehouse node
                for node_id, node_data in config['nodes'].items():
                    if node_data.get('type') == 'warehouse':
                        warehouse_location = (node_data['latitude'], node_data['longitude'])
                        warehouse_id = node_id
                        break

                if not warehouse_location:
                    print("❌ 未找到仓库节点 (type='warehouse')")
                else:
                    # Create DeliveryPoint objects
                    for delivery in config['deliveries']:
                        point_id = delivery['point_id']

                        if point_id in config['nodes']:  # Ensure delivery point exists in nodes
                            node = config['nodes'][point_id]

                            # Parse time window strings
                            now = datetime.now()
                            try:
                                if ' ' in delivery['time_window_start']:
                                    time_window_start_dt = parser.parse(delivery['time_window_start'])
                                else:
                                    today = datetime.today().date()
                                    t = datetime.strptime(delivery['time_window_start'], '%H:%M').time()
                                    time_window_start_dt = datetime.combine(today, t)

                                if ' ' in delivery['time_window_end']:
                                    time_window_end_dt = parser.parse(delivery['time_window_end'])
                                else:
                                    today = datetime.today().date()
                                    t = datetime.strptime(delivery['time_window_end'], '%H:%M').time()
                                    time_window_end_dt = datetime.combine(today, t)

                                time_window = (
                                    now.replace(hour=time_window_start_dt.hour, minute=time_window_start_dt.minute,
                                                second=0, microsecond=0) if ' ' not in delivery[
                                        'time_window_start'] else time_window_start_dt,
                                    now.replace(hour=time_window_end_dt.hour, minute=time_window_end_dt.minute,
                                                second=0, microsecond=0) if ' ' not in delivery[
                                        'time_window_end'] else time_window_end_dt
                                )
                                if time_window[1] < time_window[0]:  # Adjust if time window spans midnight
                                    time_window = (time_window[0], time_window[1] + timedelta(days=1))
                            except ValueError:
                                print(
                                    f"⚠️  Invalid time window format for {point_id}: {delivery['time_window_start']} - {delivery['time_window_end']}. Using default (08:00-18:00).")
                                time_window = (now.replace(hour=8, minute=0), now.replace(hour=18, minute=0))

                            priority_map = {'high': 1, 'medium': 2, 'low': 3}
                            priority = priority_map.get(delivery.get('priority', 'medium'), 2)

                            delivery_points.append(DeliveryPoint(
                                id=point_id,
                                location=(node['latitude'], node['longitude']),
                                demand=delivery['demand'],
                                time_window=time_window,
                                service_time=delivery['service_time'],
                                priority=priority
                            ))
                        else:
                            print(f"⚠️  配送点 '{point_id}' 在距离矩阵节点中未找到，跳过。")

                    # Get vehicle parameters interactively
                    vehicle_capacity = InteractiveConfig.get_float_input("车辆载重 (kg)", 200.0)
                    vehicle_max_distance = InteractiveConfig.get_float_input("车辆最大行驶距离 (km)", 150.0)
                    vehicle_speed = InteractiveConfig.get_float_input("车辆平均速度 (km/h)", 40.0)

                    # Create vehicle object
                    vehicle = Vehicle(
                        id='vehicle_001',
                        capacity=vehicle_capacity,
                        start_location=warehouse_location,
                        max_distance=vehicle_max_distance,
                        speed=vehicle_speed,
                        start_location_id=warehouse_id  # Set start_location_id
                    )

                    # Execute the optimization process
                    run_optimization(router, delivery_points, vehicle, warehouse_id)

            except Exception as e:
                print(f"❌ 处理失败: {e}")
                import traceback

                traceback.print_exc()

    # --- Option 4: Default demo ---
    elif choice == "4":
        main()  # Run the default demo function

    # --- Option 5: Export CSV templates ---
    elif choice == "5":
        print("\n📤 导出CSV模板文件")
        output_dir = input("请输入导出目录 (默认: ./templates): ").strip() or './templates'
        FileImporter.export_template_csv(output_dir)  # Export template files
        print("\n✅ 模板文件已导出!")
        print("提示: 请编辑模板文件后使用选项2导入数据")

    # --- Option 1: Interactive manual configuration (default) ---
    else:
        # If choice is not 2, 3, 4, or 5, default to interactive mode
        # Call the interactive configuration function
        main_with_interactive_config()

if __name__ == "__main__":
    # Example of running with pre-loaded points (e.g., from a file)
    # try:
    #     # Load points from a JSON file (adjust path as needed)
    #     loaded_points = InteractiveConfig.load_delivery_points_from_json("example_data/delivery_points.json")
    #     # Then call main with these points
    #     # main(points=loaded_points)
    #     pass
    # except FileNotFoundError:
    #     print("Example points file not found, running default main().")
    #     main() # Fallback to default main if file not found
    # except Exception as e:
    #     print(f"An error occurred loading points: {e}")
    #     main() # Fallback to default main if error occurs

    # Default execution: prompt user for mode
    if __name__ == "__main__":
        pass  # Control flow handled at the top level of the script
