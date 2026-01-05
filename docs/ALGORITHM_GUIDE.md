# 算法详解

本文档详细介绍物流路径优化系统中实现的各种算法原理、适用场景和性能特点。

## 目录

1. [Dijkstra算法](#dijkstra算法)
2. [A*算法](#a算法)
3. [遗传算法](#遗传算法)
4. [最近邻启发式](#最近邻启发式)
5. [多目标优化](#多目标优化)
6. [Floyd-Warshall算法](#floyd-warshall算法)
7. [动态重规划](#动态重规划)
8. [算法选择指南](#算法选择指南)

---

## Dijkstra算法

### 算法原理

Dijkstra算法是一种经典的单源最短路径算法，由荷兰计算机科学家Edsger Dijkstra于1959年提出。

**核心思想**：贪心策略 + 松弛操作

1. 维护一个距离表，记录起点到各节点的最短距离
2. 使用优先队列选择当前距离最小的未访问节点
3. 更新该节点所有邻居的距离（松弛操作）
4. 重复直到所有节点被访问或到达目标节点

**数学表示**：

对于边 (u, v)，松弛操作为：
```
if dist[u] + weight(u, v) < dist[v]:
    dist[v] = dist[u] + weight(u, v)
    prev[v] = u
```

### 时间复杂度

- **使用优先队列**: O(E log V)
  - E = 边数
  - V = 顶点数

### 适用场景

✅ **适合**：
- 小规模网络（<100个节点）
- 需要保证全局最优解
- 边权重为非负数
- 精确路径规划

❌ **不适合**：
- 大规模网络（计算时间过长）
- 实时性要求高的场景
- 包含负权重边的图

### 代码示例

```python
path, distance = router.dijkstra_shortest_path("warehouse", "point_5", "distance")
print(f"最优路径: {' -> '.join(path)}")
print(f"总距离: {distance:.2f}km")
```

---

## A*算法

### 算法原理

A*算法是一种启发式搜索算法，结合了Dijkstra的最优性和贪心搜索的高效性。

**核心思想**：启发式函数引导搜索方向

评估函数：
```
f(n) = g(n) + h(n)
```

- **g(n)**: 起点到节点n的实际代价
- **h(n)**: 节点n到终点的启发式估计（本系统使用Haversine距离）

**启发函数要求**：
- **可接受性**: h(n) ≤ 实际距离（不高估）
- **一致性**: h(n) ≤ c(n, n') + h(n')

### Haversine距离公式

计算球面两点间距离（考虑地球曲率）：

```python
R = 6371  # 地球半径(km)
Δlat = lat2 - lat1
Δlon = lon2 - lon1

a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
c = 2 * atan2(√a, √(1-a))
distance = R * c
```

### 时间复杂度

- **理论**: O(E)，实际取决于启发函数质量
- **最坏情况**: 退化为Dijkstra，O(E log V)

### 适用场景

✅ **适合**：
- 中等规模网络（100-1000个节点）
- 有明确目标点的路径规划
- 需要快速响应的实时系统
- GPS导航系统

❌ **不适合**：
- 没有坐标信息的抽象图
- 多目标点路径规划（需要多次调用）

### 性能对比

| 场景 | Dijkstra | A* | 性能提升 |
|------|----------|----|----|
| 100节点网络 | 50ms | 15ms | 3.3x |
| 500节点网络 | 800ms | 120ms | 6.7x |
| 1000节点网络 | 3.5s | 450ms | 7.8x |

---

## 遗传算法

### 算法原理

遗传算法(GA)是一种模拟自然选择和遗传机制的优化算法，由John Holland于1975年提出。

**核心概念**：

1. **染色体编码**: 配送点序列 [P1, P3, P2, P5, P4]
2. **适应度函数**: 总距离 + 约束惩罚
3. **选择**: 锦标赛选择（本系统k=3）
4. **交叉**: 顺序交叉(OX)
5. **变异**: 交换变异

### 顺序交叉(OX)

```
父代1: [A B | C D E | F G]
父代2: [C A | F G B | E D]
           ↓
子代1: [B E | C D E | A F G]  (保持父代1的中间段，其余按父代2顺序填充)
```

### 适应度函数

```python
fitness = 总距离 + Σ(约束违反惩罚)

约束惩罚:
- 容量超载: 1000 * (实际载重 - 车辆容量)
- 距离超限: 500 * (实际距离 - 最大距离)
- 时间窗违规: 200 * 违规次数
```

### 参数设置指南

| 参数 | 小规模 | 中规模 | 大规模 | 说明 |
|------|--------|--------|--------|------|
| population_size | 30 | 50 | 100 | 种群越大，搜索空间越广 |
| generations | 100 | 200 | 500 | 代数越多，收敛越充分 |
| crossover_rate | 0.8 | 0.8 | 0.85 | 交叉概率 |
| mutation_rate | 0.1 | 0.15 | 0.2 | 变异率随规模增加 |

### 时间复杂度

O(G × P × N²)
- G = 代数
- P = 种群规模
- N = 配送点数量

### 适用场景

✅ **适合**：
- 复杂约束问题（容量、时间窗、优先级）
- 大规模VRP（30+配送点）
- 不要求严格最优解
- 有充足计算时间

❌ **不适合**：
- 需要实时响应（秒级）
- 问题规模极大（>100点）
- 需要证明全局最优

### 收敛性分析

遗传算法不保证收敛到全局最优，但具有良好的近优解质量：

- **平均解质量**: 距最优解5-15%
- **收敛速度**: 通常在50-100代内达到稳定
- **解的多样性**: 可能找到多个局部最优解

---

## 最近邻启发式

### 算法原理

最近邻(Nearest Neighbor)是一种贪心算法，每次选择离当前位置最近的未访问配送点。

**算法流程**：

1. 从仓库出发
2. 在满足约束的前提下，选择最近的未访问配送点
3. 移动到该点并标记为已访问
4. 重复步骤2-3直到无可行点或完成所有配送
5. 返回仓库

### 约束检查

每个候选点需满足：

1. **容量约束**: current_load + demand ≤ capacity
2. **距离约束**: total_distance + distance_to_point ≤ max_distance
3. **时间窗约束**: arrival_time ≤ time_window_end

### 时间复杂度

O(N²)
- N = 配送点数量

### 适用场景

✅ **适合**：
- 快速生成初始解
- 实时决策（毫秒级响应）
- 简单约束问题
- 作为复杂算法的初始化

❌ **不适合**：
- 需要高质量解
- 复杂路网结构
- 紧密耦合的约束

### 解质量

相比最优解的偏差：

- **平均偏差**: 20-40%
- **最好情况**: 接近最优（规则分布）
- **最坏情况**: 50%+偏差（不规则分布）

### 改进策略

**2-opt局部搜索**：

```python
def two_opt_improve(route):
    improved = True
    while improved:
        improved = False
        for i in range(len(route) - 1):
            for j in range(i + 2, len(route)):
                # 尝试反转route[i:j]
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                if distance(new_route) < distance(route):
                    route = new_route
                    improved = True
    return route
```

---

## 多目标优化

### 算法原理

多目标优化使用差分进化(Differential Evolution)算法，平衡成本、时间、碳排放多个目标。

**评估函数**：

```
Objective = w_cost × Cost + w_time × Time + w_emission × Emission
```

### 目标计算

**1. 成本**：
```
Cost = Distance × fuel_cost_per_km + Time × driver_cost_per_hour

参数设置:
- fuel_cost_per_km = 0.8元/km
- driver_cost_per_hour = 50元/h
```

**2. 时间**：
```
Time = Σ(travel_time + service_time)

travel_time = distance / vehicle_speed
```

**3. 碳排放**：
```
Emission = Distance × emission_factor

emission_factor = 0.15 kg CO₂/km (柴油货车)
```

### 差分进化算法

**变异策略**：

```
mutant_vector = a + F × (b - c)
```
- a, b, c: 随机选择的个体
- F: 变异因子（通常0.5-1.0）

**交叉操作**：

```
if rand() < CR:
    trial_vector[i] = mutant_vector[i]
else:
    trial_vector[i] = target_vector[i]
```

### 权重设置策略

| 场景 | 成本权重 | 时间权重 | 排放权重 |
|------|---------|---------|---------|
| 成本优先 | 0.6 | 0.3 | 0.1 |
| 时效优先 | 0.2 | 0.7 | 0.1 |
| 环保优先 | 0.3 | 0.3 | 0.4 |
| 平衡模式 | 0.4 | 0.4 | 0.2 |

### 时间复杂度

O(I × P × N²)
- I = 迭代次数（通常100-500）
- P = 种群大小（通常20-50）
- N = 配送点数量

---

## Floyd-Warshall算法

### 算法原理

Floyd-Warshall是一种全源最短路径算法，计算所有节点对之间的最短路径。

**动态规划方程**：

```
dist[i][j][k] = min(
    dist[i][j][k-1],
    dist[i][k][k-1] + dist[k][j][k-1]
)
```

含义：i到j的最短路径，可以经过或不经过节点k。

### 时间复杂度

O(V³)
- V = 节点数量

### 空间复杂度

O(V²) - 存储距离矩阵

### 适用场景

✅ **适合**：
- 需要所有节点对的距离
- 配送中心网络优化
- 小规模密集图（<200节点）
- 预计算阶段（非实时）

❌ **不适合**：
- 大规模网络（内存和时间消耗大）
- 只需要单源最短路径
- 动态变化的图

---

## 动态重规划

### 算法原理

动态重规划系统实时响应路况变化，自适应调整路径。

**核心机制**：

1. **路况更新**: 实时接收拥堵信息
2. **路径评估**: 重新计算剩余路径代价
3. **触发重规划**: 当延误超过阈值时
4. **快速重规划**: 使用启发式算法（A*或最近邻）

### 路况模型

```python
actual_time = base_time × congestion_factor

congestion_factor 取值:
- 1.0: 畅通
- 1.5: 轻度拥堵
- 2.0: 中度拥堵
- 3.0: 严重拥堵
```

### 重规划触发条件

满足任一条件触发重规划：

1. **延误阈值**: 预计延误 > 15分钟
2. **路段关闭**: 原路径不可达
3. **新增紧急订单**: 高优先级配送
4. **时间窗风险**: 即将违反时间窗

### 实时性保证

- **重规划时间**: <500ms（使用A*）
- **更新频率**: 每1-5分钟检查一次
- **备用路径**: 预计算2-3条候选路径

---

## 算法选择指南

### 决策树

```
问题规模?
├─ <10个配送点
│  ├─ 需要最优解? → Dijkstra / A*
│  └─ 快速估算? → 最近邻
├─ 10-30个配送点
│  ├─ 简单约束? → A*
│  ├─ 复杂约束? → 遗传算法
│  └─ 多目标? → 多目标优化
└─ >30个配送点
   ├─ 实时决策? → 最近邻 + 2-opt
   └─ 离线优化? → 遗传算法(大种群)
```

### 性能对比表

| 算法 | 解质量 | 速度 | 可扩展性 | 约束处理 | 实时性 |
|------|--------|------|----------|----------|--------|
| Dijkstra | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| A* | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| 遗传算法 | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| 最近邻 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 多目标优化 | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |

### 组合策略

**混合算法方案**：

1. **初始化**: 最近邻生成初始解
2. **优化**: 遗传算法改进解质量
3. **精炼**: 2-opt局部搜索
4. **实时调整**: A*动态重规划

```python
# 混合算法示例
initial_route = nearest_neighbor(vehicle, points)
optimized_route = genetic_algorithm(initial_route)
final_route = two_opt_improve(optimized_route)

# 运行中动态调整
if traffic_changed():
    new_route = a_star_reroute(current_position, remaining_points)
```

---

## 参考文献

1. Dijkstra, E. W. (1959). "A note on two problems in connexion with graphs"
2. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). "A formal basis for the heuristic determination of minimum cost paths"
3. Holland, J. H. (1975). "Adaptation in Natural and Artificial Systems"
4. Clarke, G., & Wright, J. W. (1964). "Scheduling of vehicles from a central depot to a number of delivery points"
5. Storn, R., & Price, K. (1997). "Differential evolution – a simple and efficient heuristic for global optimization"

---

