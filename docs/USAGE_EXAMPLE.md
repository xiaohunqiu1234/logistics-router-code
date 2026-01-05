# 使用示例：从CSV导入数据运行路径优化

## 快速开始

### 步骤1：准备CSV文件

将示例数据文件放在 `example_data/` 目录：
- `nodes.csv` - 节点坐标和类型
- `edges.csv` - 道路网络
- `deliveries.csv` - 配送需求

### 步骤2：运行程序

```bash
python scripts/logistics-router-interactive.py
```

### 步骤3：选择导入模式

```
=== 物流路径优化系统 ===

请选择运行模式:
1. 交互式配置
2. 从CSV导入完整网络
3. 从距离矩阵导入
4. 导出CSV模板
5. 默认演示

请选择 (1-5): 2
```

### 步骤4：指定文件路径

```
请输入nodes.csv文件路径: example_data/nodes.csv
✅ 成功导入 5 个节点
✅ 成功导入 9 条边
✅ 成功导入 4 个配送点
```

### 步骤5：配置车辆参数

```
请输入车辆数量 (默认: 2): 2
请输入车辆容量 (默认: 100): 150
请输入最大行驶距离/公里 (默认: 50): 60
```

### 步骤6：选择优化算法

```
请选择优化算法:
1. 贪心算法（快速）
2. 动态规划（精确，适合小规模）
3. 遗传算法（平衡性能和质量）
4. 模拟退火（全局搜索）

请选择 (1-4): 3
```

### 步骤7：查看优化结果

```
========================================
🚚 遗传算法优化结果
========================================

车辆 1 路径:
  warehouse → point_a (5.2 km, 10 min) → point_c (4.1 km, 8 min) → warehouse
  总距离: 15.4 km
  总需求: 95
  总时间: 41 min（含服务时间）

车辆 2 路径:
  warehouse → point_b (7.5 km, 15 min) → point_d (3.8 km, 7 min) → warehouse
  总距离: 19.2 km
  总需求: 55
  总时间: 48 min

----------------------------------------
总距离: 34.6 km
总时间: 89 min
车辆利用率: 75%
========================================
```

---

## 完整Python代码示例

```python
from logistics_router import LogisticsRouter, DeliveryPoint, FileImporter

# 方式1：从CSV导入
config = FileImporter.import_from_csv('example_data/nodes.csv')

# 构建路网
router = LogisticsRouter()
for node_id, node_data in config['nodes'].items():
    router.add_location(node_id, node_data['latitude'], node_data['longitude'])

for edge in config['edges']:
    router.add_route(edge['from'], edge['to'], edge['distance'], 
                     edge['min_time'], edge['max_time'])

# 创建配送点
delivery_points = []
for delivery in config['deliveries']:
    point = DeliveryPoint(
        id=delivery['id'],
        demand=delivery['demand'],
        service_time=delivery['service_time'],
        priority=delivery['priority']
    )
    delivery_points.append(point)

# 运行优化
paths = router.genetic_algorithm_vrp(delivery_points, num_vehicles=2, 
                                     vehicle_capacity=150)

# 输出结果
for i, path in enumerate(paths, 1):
    print(f"车辆 {i}: {' → '.join(path)}")
```

---

## 方式2：距离矩阵导入示例

```python
# 导入距离矩阵
matrix, node_ids = FileImporter.import_distance_matrix('example_data/distance_matrix.csv')

# 创建简化路网
router = LogisticsRouter()
for i, node_id in enumerate(node_ids):
    router.add_location(node_id, 0, 0)  # 坐标不重要

# 添加距离
for i in range(len(node_ids)):
    for j in range(len(node_ids)):
        if i != j and matrix[i][j] != float('inf'):
            router.add_route(node_ids[i], node_ids[j], matrix[i][j], 
                           matrix[i][j] * 2, matrix[i][j] * 3)

# 其余步骤相同...
