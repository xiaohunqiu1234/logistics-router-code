# 物流路径优化系统 - 完整使用指南

## 目录

1. [系统概述](#系统概述)
2. [安装与配置](#安装与配置)
3. [数据准备](#数据准备)
4. [运行模式](#运行模式)
5. [算法详解](#算法详解)
6. [性能优化](#性能优化)
7. [可视化分析](#可视化分析)
8. [常见问题](#常见问题)

---

## 系统概述

本系统是一个完整的物流配送路径规划解决方案，集成了以下核心功能：

### 核心算法
- **Dijkstra**: 经典最短路径算法
- **A***: 启发式搜索算法
- **遗传算法**: 全局优化算法
- **最近邻**: 快速启发式算法
- **多目标优化**: 平衡成本、时间、排放
- **Floyd-Warshall**: 全源最短路径

### 约束处理
- ✅ 车辆容量限制
- ✅ 时间窗约束
- ✅ 最大行驶距离
- ✅ 服务时长
- ✅ 配送优先级

### 可视化功能
- 📊 性能对比图表
- 🗺️ 路径网络图
- 📈 时间甘特图
- 🎯 综合雷达图
- 🌐 网络拓扑图
- 🎬 搜索过程动画

---

## 安装与配置

### 系统要求

- Python 3.8+
- 内存: 最少2GB，推荐4GB+
- 存储: 100MB+

### 安装步骤

```bash
# 1. 克隆项目/下载项目
# / 克隆项目
git https://github.com/xiaohunqiu1234/logistics-router-code
cd logistics-router-code
# 下载项目,并解压

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 或手动安装
pip install numpy scipy matplotlib networkx
```

### 验证安装

```bash
python scripts/logistics-router-interactive.py
```

如果看到主菜单，说明安装成功。

---

## 数据准备

### CSV文件准备

系统支持三种数据导入方式：

#### 方式1: 完整网络数据（推荐）

准备三个CSV文件：

**1. nodes.csv** - 节点信息
```csv
id,latitude,longitude,type
warehouse,39.9042,116.4074,warehouse
point_1,39.9100,116.4200,delivery
point_2,39.9150,116.4150,delivery
transit_1,39.9080,116.4120,transit
```

字段说明：
- `id`: 节点唯一标识符
- `latitude`: 纬度（-90到90）
- `longitude`: 经度（-180到180）
- `type`: 节点类型（warehouse/delivery/transit）

**2. edges.csv** - 道路信息
```csv
from_node,to_node,distance,min_time,max_time
warehouse,point_1,5.2,10,15
warehouse,point_2,6.8,12,18
point_1,point_2,3.5,8,12
```

字段说明：
- `from_node`: 起点节点ID
- `to_node`: 终点节点ID
- `distance`: 距离（公里）
- `min_time`: 最短时间（分钟，畅通时）
- `max_time`: 最长时间（分钟，拥堵时）

**3. deliveries.csv** - 配送需求
```csv
id,demand,service_time,priority,time_window_start,time_window_end
point_1,50.0,15,high,09:00,12:00
point_2,30.0,10,medium,10:00,14:00
point_3,45.0,20,high,08:00,11:00
```

字段说明：
- `id`: 配送点ID（必须在nodes.csv中存在）
- `demand`: 需求量（kg）
- `service_time`: 服务时长（分钟）
- `priority`: 优先级（high/medium/low）
- `time_window_start`: 时间窗开始（HH:MM格式）
- `time_window_end`: 时间窗结束（HH:MM格式）

#### 方式2: 距离矩阵

如果已有预计算的距离矩阵：

**distance_matrix.csv**
```csv
,warehouse,point_1,point_2,point_3
warehouse,0,5.2,6.8,8.3
point_1,5.2,0,3.5,4.2
point_2,6.8,3.5,0,5.1
point_3,8.3,4.2,5.1,0
```

第一行和第一列为节点ID，其余为距离值。

### 导出模板

系统提供模板导出功能：

```bash
python scripts/logistics-router-interactive.py
# 选择选项 4 - 导出CSV模板
```

会在`templates/`目录生成示例文件。

---

## 运行模式

### 模式1: 交互式配置

最灵活的方式，适合初次使用或参数调优。

```bash
python scripts/logistics-router-interactive.py
# 选择选项 1
```

系统会引导您配置：
1. 基础参数（配送点数量、车辆容量等）
2. 仓库位置
3. 配送点信息（位置、需求、时间窗）
4. 算法选择

### 模式2: CSV导入完整网络

适合生产环境或批量测试。

```bash
python scripts/logistics-router-interactive.py
# 选择选项 2
# 输入CSV文件目录: data/
```

系统会：
1. 读取nodes.csv构建路网
2. 读取edges.csv添加道路
3. 读取deliveries.csv创建配送需求
4. 自动运行所选算法

### 模式3: 距离矩阵导入

适合简化场景或已有距离数据。

```bash
python scripts/logistics-router-interactive.py
# 选择选项 3
# 输入距离矩阵文件路径: data/distance_matrix.csv
# （可选）输入配送配置文件: data/delivery_config.csv
```

### 模式4: 默认演示

快速查看系统功能。

```bash
python scripts/logistics-router-interactive.py
# 选择选项 4
```

使用内置示例数据运行所有算法。

---

## 算法详解

### 何时使用哪种算法？

#### Dijkstra算法
**适用场景**：
- 需要精确的点到点最短路径
- 网络规模较小（<100节点）
- 不需要启发信息

**不适用**：
- 大规模网络（计算慢）
- 需要快速响应

**示例**：
```python
router = LogisticsRouter()
path, distance = router.dijkstra_shortest_path("warehouse", "point_5", "distance")
```

#### A*算法
**适用场景**：
- 有GPS坐标信息
- 需要快速路径查询
- 中等规模网络（100-1000节点）

**不适用**：
- 没有坐标信息
- 启发函数不可靠

**示例**：
```python
router = AStarRouter()
router.add_node_coordinates("warehouse", 39.9042, 116.4074)
path, distance = router.a_star_search("warehouse", "point_5")
```

#### 遗传算法
**适用场景**：
- 复杂VRP问题（>10个配送点）
- 有多种约束（容量、时间窗）
- 可接受较长计算时间

**不适用**：
- 需要实时响应
- 问题规模极大（>100点）

**参数调优**：
```python
ga = GeneticAlgorithm(
    points, router,
    population_size=50,   # 种群规模：30-100
    generations=200,      # 代数：100-500
    mutation_rate=0.15,   # 变异率：0.1-0.3
    crossover_rate=0.8    # 交叉率：0.7-0.9
)
```

#### 最近邻算法
**适用场景**：
- 需要快速初始解
- 实时决策
- 简单约束

**不适用**：
- 需要高质量解

**示例**：
```python
solver = VRPTWSolver([vehicle], points)
route = solver.nearest_neighbor_heuristic(vehicle)
```

#### 多目标优化
**适用场景**：
- 需要平衡多个目标（成本、时间、排放）
- 中等规模问题（10-30点）

**权重设置**：
```python
# 成本优先
optimizer = MultiObjectiveOptimizer(
    weight_cost=0.6,
    weight_time=0.3,
    weight_emission=0.1
)

# 时效优先
optimizer = MultiObjectiveOptimizer(
    weight_cost=0.2,
    weight_time=0.7,
    weight_emission=0.1
)

# 环保优先
optimizer = MultiObjectiveOptimizer(
    weight_cost=0.3,
    weight_time=0.3,
    weight_emission=0.4
)
```

---

## 性能优化

### 大规模问题优化策略

#### 1. 分区策略
```python
# 使用K-means聚类将配送点分组
from sklearn.cluster import KMeans

# 提取坐标
coords = np.array([p.location for p in points])

# 聚类
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(coords)

# 分别优化每个区域
for cluster_id in range(3):
    cluster_points = [p for i, p in enumerate(points) if clusters[i] == cluster_id]
    route = ga.evolve(vehicle)
```

#### 2. 减少算法参数
```python
# 大规模问题使用较小参数
ga = GeneticAlgorithm(
    points, router,
    population_size=30,   # 减少种群
    generations=100,      # 减少代数
    mutation_rate=0.2
)
```

#### 3. 使用混合策略
```python
# 先用最近邻获取初始解
initial_route = solver.nearest_neighbor_heuristic(vehicle)

# 再用遗传算法优化
ga.population[0] = initial_route  # 将初始解加入种群
optimized_route = ga.evolve(vehicle)
```

### 内存优化

```python
# 避免存储所有中间结果
analyzer = PerformanceAnalyzer()
# 只添加最终结果
analyzer.add_result("GA", final_route, vehicle, router)
```

---

## 可视化分析

系统自动生成多种可视化图表，保存在`output/`目录。

### 1. 性能对比图
`output/performance_comparison.png`

展示不同算法在距离、时间、成本、排放等指标上的对比。

### 2. 路径网络图
`output/route_network_{algorithm}.png`

显示具体算法的配送路径，包括：
- 仓库位置（红色三角）
- 配送点（蓝色圆点）
- 路径连线

### 3. 配送顺序对比图
`output/delivery_sequence_comparison.png`

横向对比不同算法的配送点访问顺序。

### 4. 时间甘特图
`output/time_gantt_{algorithm}.png`

显示每个配送点的：
- 行驶时间（蓝色）
- 服务时间（绿色）
- 累积时间

### 5. 综合雷达图
`output/radar_chart.png`

多维度性能评估：
- 距离效率
- 时间效率
- 成本效率
- 环保指数
- 载重率

### 6. 网络拓扑图
`output/network_topology.png`

展示完整网络结构：
- 节点连接关系
- 节点度分布
- 边长度分布

### 7. 搜索动态GIF
`output/search_animation_{algorithm}.gif`

遗传算法优化过程动画，展示解的演化过程。

---

## 常见问题

### 数据问题

**Q: CSV文件导入失败**
```
错误：UnicodeDecodeError
解决：确保CSV文件编码为UTF-8
```

**Q: 节点未找到**
```
错误：Node 'point_1' not found
解决：检查deliveries.csv中的ID是否在nodes.csv中存在
```

### 算法问题

**Q: 遗传算法不收敛**
```
现象：距离一直不下降
解决：
1. 增加种群规模（50->100）
2. 增加代数（100->200）
3. 调整变异率（0.1->0.2）
```

**Q: 最近邻算法返回空路径**
```
现象：route为空列表
解决：
1. 检查车辆容量是否足够
2. 检查最大行驶距离
3. 检查时间窗设置
4. 使用diagnose_vrp_constraints()诊断
```

### 性能问题

**Q: 程序运行很慢**
```
解决：
1. 减少配送点数量（分批处理）
2. 降低算法参数
3. 使用最近邻算法快速测试
```

**Q: 内存不足**
```
解决：
1. 减少遗传算法种群规模
2. 不保存中间结果
3. 分区处理大问题
```

### 可视化问题

**Q: 雷达图显示错误**
```
错误：Invalid vertices array
解决：系统已修复，确保使用最新版本
```

**Q: 图表不显示中文**
```
解决：系统已配置SimHei字体，确保系统有该字体
```

---

## 技术支持

如需更多帮助，请查看：
- API参考: `docs/API_REFERENCE.md`
- 算法详解: `docs/ALGORITHM_GUIDE.md`
- GitHub Issues: https://github.com/your-repo/issues

---

**文档版本**: 2.0  
**最后更新**: 2024年1月
