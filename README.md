# 智能物流路径规划系统 📦

一个完整的物流配送路径规划解决方案，集成多种经典算法和现代优化技术。

## 🚀 功能特性

### 1. 基础路径规划
- **Dijkstra算法**: 单源最短路径，支持多维权重（距离/时间/成本）
- **A*算法**: 启发式搜索，结合GPS坐标的快速路径查询
- **Floyd-Warshall**: 全源最短路径，适合配送中心网络优化

### 2. 车辆路径问题 (VRP)
- **VRP-TW**: 带时间窗的车辆路径问题
- **最近邻启发式**: 快速生成初始解
- **遗传算法**: 进化算法优化路径质量

### 3. 高级功能
- **动态重规划**: 实时路况适应和路径调整
- **多目标优化**: 平衡成本、时间、碳排放
- **约束处理**: 容量限制、时间窗、服务时长

## 📦 安装

```bash
# 克隆项目
git clone https://github.com/your-repo/logistics-router.git
cd logistics-router

# 安装依赖
pip install numpy scipy
```

## 🎯 快速开始

```python
from logistics_router import AStarRouter, Vehicle, DeliveryPoint, VRPTWSolver
from datetime import datetime, timedelta

# 1. 创建路由器
router = AStarRouter()
router.add_node_coordinates('warehouse', 23.1291, 113.2644)
router.add_node_coordinates('store_a', 23.1350, 113.2700)
router.add_edge('warehouse', 'store_a', distance=5.2, time=8, cost=12)

# 2. 查询路径
path, distance = router.a_star_search('warehouse', 'store_a', 'distance')
print(f"最短路径: {' -> '.join(path)}, 距离: {distance}km")

# 3. 配送优化
points = [
    DeliveryPoint(
        id='store_a',
        location=(23.1350, 113.2700),
        demand=50,
        time_window=(datetime.now(), datetime.now() + timedelta(hours=2)),
        service_time=15
    )
]

vehicle = Vehicle(
    id='truck_01',
    capacity=150,
    start_location=(23.1291, 113.2644),
    max_distance=100,
    speed=40
)

solver = VRPTWSolver([vehicle], points)
route = solver.nearest_neighbor_heuristic(vehicle)
print(f"配送路径: {[p.id for p in route]}")
```

## 📚 核心模块

### LogisticsRouter
基础图路由，实现Dijkstra算法。

```python
router = LogisticsRouter()
router.add_edge('A', 'B', distance=10, time=15, cost=20)
path, weight = router.dijkstra_shortest_path('A', 'B', weight_type='distance')
```

### AStarRouter
A*启发式搜索，需要GPS坐标。

```python
router = AStarRouter()
router.add_node_coordinates('A', 39.9042, 116.4074)  # 北京
router.add_node_coordinates('B', 31.2304, 121.4737)  # 上海
path, distance = router.a_star_search('A', 'B')
```

### VRPTWSolver
车辆路径问题求解器，处理时间窗约束。

```python
solver = VRPTWSolver(vehicles, delivery_points)
route = solver.nearest_neighbor_heuristic(vehicle)
```

### GeneticVRPSolver
遗传算法优化器，适合复杂场景。

```python
ga_solver = GeneticVRPSolver(population_size=100, generations=500)
optimized_route = ga_solver.solve(points, vehicle)
```

### MultiObjectiveOptimizer
多目标优化，平衡成本、时间、环保。

```python
optimizer = MultiObjectiveOptimizer(
    weight_cost=0.4,
    weight_time=0.4,
    weight_emission=0.2
)
best_route = optimizer.optimize_route(points, vehicle)
```

## 🔧 算法对比

| 算法 | 时间复杂度 | 适用场景 | 解质量 |
|------|-----------|---------|--------|
| Dijkstra | O((V+E)logV) | 单源最短路径 | 最优 |
| A* | O(b^d) | 启发式搜索 | 最优* |
| Floyd-Warshall | O(V³) | 全源最短路径 | 最优 |
| 最近邻 | O(n²) | 快速VRP初解 | 120-150%最优 |
| 遗传算法 | O(g×p×n) | VRP全局优化 | 接近最优 |

*需要可采纳的启发函数

## 📊 应用场景

- ✅ **快递配送**: 最后一公里路径优化
- ✅ **外卖配送**: 实时动态路径调整
- ✅ **冷链物流**: 时间窗约束下的路径规划
- ✅ **货运调度**: 多车辆多仓库协同优化
- ✅ **应急物流**: 快速响应路径生成

## 🐛 常见问题

**Q: 如何处理大规模配送点（>100个）？**  
A: 建议先用聚类算法分区，再对每个区域独立优化。

**Q: 时间窗约束无法满足怎么办？**  
A: 检查车辆数量是否足够，或放宽时间窗设置。

**Q: 如何集成实时路况？**  
A: 使用`DynamicRoutePlanner`类，定期调用`update_traffic()`更新路况系数。

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

**⭐ 如果这个项目对您有帮助，请给个Star！**
