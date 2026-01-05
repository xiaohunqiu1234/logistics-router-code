# API参考文档

## 核心类API

### LogisticsRouter

基础路径规划引擎，实现Dijkstra最短路径算法。

#### 构造函数
```python
router = LogisticsRouter()
```

#### 方法

##### add_edge()
添加道路边到路网图。

```python
router.add_edge(
    from_node: str,      # 起点节点ID
    to_node: str,        # 终点节点ID
    distance: float,     # 距离（公里）
    time: float,         # 行驶时间（分钟）
    cost: float          # 成本（元）
)
```

示例：
```python
router.add_edge("warehouse", "point_1", distance=5.2, time=10, cost=8.5)
```

##### add_node_coordinates()
添加节点GPS坐标（用于可视化）。

```python
router.add_node_coordinates(
    node: str,          # 节点ID
    lat: float,         # 纬度（-90到90）
    lon: float          # 经度（-180到180）
)
```

##### dijkstra_shortest_path()
使用Dijkstra算法计算最短路径。

```python
path, total_weight = router.dijkstra_shortest_path(
    start: str,                    # 起点节点ID
    end: str,                      # 终点节点ID
    weight_type: str = 'distance'  # 优化目标：'distance'/'time'/'cost'
)
```

返回值：
- `path: List[str]` - 节点ID序列，空列表表示无路径
- `total_weight: float` - 路径总权重，无路径时为inf

示例：
```python
path, distance = router.dijkstra_shortest_path("warehouse", "point_5", "distance")
if path:
    print(f"路径: {' -> '.join(path)}, 距离: {distance:.2f}km")
```

---

### AStarRouter

继承自LogisticsRouter，实现A*启发式搜索算法。

#### 构造函数
```python
router = AStarRouter()
```

#### 额外方法

##### haversine_distance()
计算两点间球面距离（用作启发函数）。

```python
distance = router.haversine_distance(
    node1: str,  # 节点1 ID
    node2: str   # 节点2 ID
)
```

返回值：`float` - 球面距离（公里）

##### a_star_search()
使用A*算法搜索最优路径。

```python
path, total_weight = router.a_star_search(
    start: str,
    end: str,
    weight_type: str = 'distance'
)
```

参数和返回值同dijkstra_shortest_path()。

---

### DeliveryPoint

配送点数据类。

#### 构造函数
```python
point = DeliveryPoint(
    id: str,                              # 配送点唯一ID
    location: Tuple[float, float],        # GPS坐标 (纬度, 经度)
    demand: float,                        # 需求量（kg）
    time_window: Tuple[datetime, datetime], # 时间窗 (最早, 最晚)
    service_time: int,                    # 服务时长（分钟）
    priority: int = 1                     # 优先级（1-5）
)
```

示例：
```python
from datetime import datetime

point = DeliveryPoint(
    id="delivery_001",
    location=(39.9042, 116.4074),
    demand=50.0,
    time_window=(
        datetime(2024, 1, 10, 9, 0),   # 9:00 AM
        datetime(2024, 1, 10, 12, 0)   # 12:00 PM
    ),
    service_time=15,
    priority=2
)
```

---

### Vehicle

车辆数据类。

#### 构造函数
```python
vehicle = Vehicle(
    id: str,                          # 车辆ID
    capacity: float,                  # 载重容量（kg）
    start_location: Tuple[float, float], # 起始位置GPS坐标
    max_distance: float,              # 最大行驶距离（km）
    speed: float = 60,                # 平均速度（km/h）
    start_location_id: Optional[str] = None  # 起始位置节点ID
)
```

示例：
```python
vehicle = Vehicle(
    id="truck_01",
    capacity=1000.0,
    start_location=(39.9042, 116.4074),
    max_distance=200.0,
    speed=50.0,
    start_location_id="warehouse"
)
```

---

### VRPTWSolver

带时间窗的车辆路径问题求解器。

#### 构造函数
```python
solver = VRPTWSolver(
    vehicles: List[Vehicle],
    points: List[DeliveryPoint]
)
```

#### 方法

##### nearest_neighbor_heuristic()
最近邻启发式算法，快速生成可行路径。

```python
route = solver.nearest_neighbor_heuristic(vehicle: Vehicle)
```

返回值：`List[DeliveryPoint]` - 优化后的配送顺序

特点：
- 贪心策略，每次选择最近的可行配送点
- 考虑容量、距离、时间窗约束
- 输出详细诊断信息

---

### GeneticAlgorithm

遗传算法求解器。

#### 构造函数
```python
ga = GeneticAlgorithm(
    points: List[DeliveryPoint],
    router,                        # LogisticsRouter实例
    population_size: int = 50,     # 种群大小
    generations: int = 100,        # 进化代数
    mutation_rate: float = 0.1,    # 变异率
    crossover_rate: float = 0.8    # 交叉率
)
```

#### 方法

##### evolve()
执行遗传算法优化。

```python
route = ga.evolve(vehicle: Vehicle)
```

返回值：`List[DeliveryPoint]` - 优化后的配送顺序

算法流程：
1. 初始化随机种群
2. 计算适应度（距离+约束惩罚）
3. 锦标赛选择
4. 顺序交叉（OX）
5. 交换变异
6. 迭代进化

---

### MultiObjectiveOptimizer

多目标优化器（基于差分进化算法）。

#### 构造函数
```python
optimizer = MultiObjectiveOptimizer(
    weight_cost: float = 0.4,       # 成本权重
    weight_time: float = 0.4,       # 时间权重
    weight_emission: float = 0.2    # 碳排放权重
)
```

#### 方法

##### optimize_route()
多目标路径优化。

```python
route = optimizer.optimize_route(
    points: List[DeliveryPoint],
    vehicle: Vehicle
)
```

返回值：`List[DeliveryPoint]` - 平衡多个目标的优化路径

优化目标：
- **成本** = 燃料成本 + 人工成本
- **时间** = 行驶时间 + 服务时间
- **碳排放** = 距离 × 排放因子

---

### PerformanceAnalyzer

性能分析与可视化工具。

#### 构造函数
```python
analyzer = PerformanceAnalyzer()
```

#### 方法

##### add_result()
添加算法结果进行分析。

```python
analyzer.add_result(
    algorithm_name: str,           # 算法名称
    route: List[DeliveryPoint],    # 优化路径
    vehicle: Vehicle,              # 车辆信息
    router=None,                   # 可选：路由器实例
    execution_time: float = 0.0,   # 执行时间（秒）
    search_steps: List = None      # 可选：搜索步骤记录
)
```

##### generate_comparison_table()
生成算法对比表格。

```python
analyzer.generate_comparison_table()
```

输出示例：
```
算法性能对比:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
算法            距离(km)  时间(h)  成本(元)  排放(kg)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dijkstra         45.2     1.8      120.5     6.8
A*               43.8     1.7      115.2     6.6  ⭐
遗传算法         42.5     1.6      112.0     6.4  ⭐
```

##### plot_performance_comparison()
生成性能对比柱状图。

```python
analyzer.plot_performance_comparison(output_dir: str = "output")
```

##### plot_route_network()
绘制路径网络图。

```python
analyzer.plot_route_network(
    router,                  # 路由器实例
    algorithm_name: str,     # 算法名称
    output_dir: str = "output"
)
```

##### plot_network_topology()
绘制配送网络拓扑分析图。

```python
analyzer.plot_network_topology(
    router,                  # 路由器实例
    output_dir: str = "output"
)
```

##### create_search_animation()
创建路径搜索动态GIF。

```python
analyzer.create_search_animation(
    algorithm_name: str,
    router,
    vehicle: Vehicle,
    output_dir: str = "output"
)
```

---

### FileImporter

CSV/Excel数据导入工具。

#### 静态方法

##### import_from_csv()
从CSV目录导入网络配置。

```python
config = FileImporter.import_from_csv(file_path: str)
```

参数：
- `file_path: str` - CSV文件所在目录路径

返回值：
```python
{
    'nodes': {
        'node_id': {'latitude': float, 'longitude': float, 'type': str}
    },
    'edges': [
        {'from': str, 'to': str, 'distance': float, 'min_time': float, 'max_time': float}
    ],
    'deliveries': [
        {'point_id': str, 'demand': float, 'service_time': float, ...}
    ]
}
```

##### import_distance_matrix()
导入距离矩阵。

```python
matrix, node_ids = FileImporter.import_distance_matrix(file_path: str)
```

返回值：
- `matrix: np.ndarray` - 距离矩阵
- `node_ids: List[str]` - 节点ID列表

##### export_csv_template()
导出CSV模板文件。

```python
FileImporter.export_csv_template(output_dir: str = "templates")
```

生成文件：
- `nodes_template.csv`
- `edges_template.csv`
- `deliveries_template.csv`
- `distance_matrix_template.csv`

---

### InteractiveConfig

交互式配置工具。

#### 静态方法

##### get_int_input()
获取整数输入。

```python
value = InteractiveConfig.get_int_input(
    prompt: str,
    default: int = None,
    min_val: int = None,
    max_val: int = None
)
```

##### get_float_input()
获取浮点数输入。

```python
value = InteractiveConfig.get_float_input(
    prompt: str,
    default: float = None,
    min_val: float = None,
    max_val: float = None
)
```

##### get_choice()
获取选择输入。

```python
choice = InteractiveConfig.get_choice(
    prompt: str,
    choices: List[str]
)
```

示例：
```python
algorithm = InteractiveConfig.get_choice(
    "选择算法",
    ["Dijkstra", "A*", "遗传算法", "全部运行"]
)
```

---

## 工具函数

### diagnose_vrp_constraints()
诊断VRP约束是否合理。

```python
diagnose_vrp_constraints(
    delivery_points: List[DeliveryPoint],
    vehicle: Vehicle
)
```

输出内容：
- 需求分析（总需求、平均需求、最大单点需求）
- 距离分析（最近/最远/平均距离）
- 时间窗分析（过期配送点检查）
- 优化建议

---

## 使用示例

### 完整工作流程

```python
from datetime import datetime, timedelta

# 1. 创建路由器
router = AStarRouter()

# 2. 添加节点和边
router.add_node_coordinates("warehouse", 39.9042, 116.4074)
router.add_node_coordinates("point_1", 39.9100, 116.4200)
router.add_edge("warehouse", "point_1", distance=5.2, time=10, cost=8.5)

# 3. 创建配送点
now = datetime.now()
points = [
    DeliveryPoint(
        id="point_1",
        location=(39.9100, 116.4200),
        demand=50.0,
        time_window=(now + timedelta(hours=1), now + timedelta(hours=4)),
        service_time=15
    )
]

# 4. 创建车辆
vehicle = Vehicle(
    id="truck_01",
    capacity=1000.0,
    start_location=(39.9042, 116.4074),
    max_distance=200.0
)

# 5. 诊断约束
diagnose_vrp_constraints(points, vehicle)

# 6. 运行算法
ga = GeneticAlgorithm(points, router)
optimized_route = ga.evolve(vehicle)

# 7. 性能分析
analyzer = PerformanceAnalyzer()
analyzer.add_result("遗传算法", optimized_route, vehicle, router)
analyzer.generate_comparison_table()
analyzer.plot_performance_comparison()
```

---

## 最佳实践

### 参数调优

**小规模问题（<10个点）**
```python
ga = GeneticAlgorithm(
    points, router,
    population_size=30,
    generations=100,
    mutation_rate=0.1
)
```

**中等规模（10-30个点）**
```python
ga = GeneticAlgorithm(
    points, router,
    population_size=50,
    generations=200,
    mutation_rate=0.15
)
```

**大规模问题（30+个点）**
```python
ga = GeneticAlgorithm(
    points, router,
    population_size=100,
    generations=500,
    mutation_rate=0.2
)
```

### 错误处理

```python
try:
    route = ga.evolve(vehicle)
    if not route:
        print("警告：算法未能生成有效路径")
        # 使用备用算法
        solver = VRPTWSolver([vehicle], points)
        route = solver.nearest_neighbor_heuristic(vehicle)
except Exception as e:
    print(f"错误：{e}")
    # 记录日志或回退到默认策略
```
---