# 距离矩阵导入使用指南

## 概述

距离矩阵导入方式适合以下场景：
- 已有完整的节点间距离数据
- 不需要复杂的道路网络信息
- 快速测试小规模路径优化问题
- 数据来自其他系统（如Google Maps API、高德地图API等）

## 使用步骤

### 第一步：准备距离矩阵文件

创建一个CSV文件，格式如下：

```csv
,W001,D001,D002,D003
W001,0,5.2,8.3,12.1
D001,5.2,0,4.1,9.8
D002,8.3,4.1,0,6.5
D003,12.1,9.8,6.5,0
```

**格式说明：**
- 第一行第一列为空
- 第一行其余列为节点ID（建议用W开头表示仓库，D开头表示配送点）
- 第一列为节点ID（顺序与第一行相同）
- 其余单元格为对应节点间的距离（公里）
- 对角线为0（节点到自己的距离）
- 可以是非对称矩阵（A到B的距离 ≠ B到A的距离）

### 第二步：准备配送点配置文件（可选）

创建一个配送点配置CSV文件：

```csv
node_id,demand,priority,service_time,time_window_start,time_window_end
D001,50,high,15,09:00,12:00
D002,30,medium,10,10:00,14:00
D003,80,high,20,08:00,11:00
```

**字段说明：**
- `node_id`: 配送点ID（必须与距离矩阵中的节点ID匹配）
- `demand`: 配送需求量（kg或件数）
- `priority`: 优先级（high/medium/low）
- `service_time`: 服务时间（分钟）
- `time_window_start`: 时间窗开始时间（格式：HH:MM）
- `time_window_end`: 时间窗结束时间（格式：HH:MM）

如果不提供配置文件，系统会使用默认值：
- demand: 50
- priority: medium
- service_time: 15分钟
- time_window: 08:00-18:00

### 第三步：运行程序导入

```bash
python scripts/logistics-router-interactive.py
```

选择导入方式：

```
请选择配置方式:
1. 从CSV导入完整网络
2. 导出CSV模板
3. 从距离矩阵导入  ← 选择此项
4. 手动输入配置
```

输入文件路径：

```
请输入距离矩阵CSV文件路径: example_data/simple_distance_matrix.csv
✅ 导入距离矩阵: 6 x 6

是否有配送点配置文件? (y/n, 默认n): y
请输入配送点配置文件路径: example_data/delivery_config_for_matrix.csv
✅ 导入配送点配置: 5 个配送点
```

## 文件命名建议

### 推荐的文件命名规范

1. **按项目命名：**
   ```
   project_name_distance_matrix.csv
   project_name_delivery_config.csv
   ```
   例如：
   ```
   guangzhou_logistics_distance_matrix.csv
   guangzhou_logistics_delivery_config.csv
   ```

2. **按日期命名：**
   ```
   distance_matrix_20250115.csv
   delivery_config_20250115.csv
   ```

3. **按区域命名：**
   ```
   tianhe_district_distance_matrix.csv
   tianhe_district_delivery_config.csv
   ```

4. **按业务类型命名：**
   ```
   fresh_food_distance_matrix.csv
   fresh_food_delivery_config.csv
   ```

### 文件存放位置建议

```
your_project/
├── example_data/              # 示例数据
│   ├── simple_distance_matrix.csv
│   └── delivery_config_for_matrix.csv
├── real_data/                 # 实际业务数据
│   ├── guangzhou_20250115/
│   │   ├── distance_matrix.csv
│   │   └── delivery_config.csv
│   └── shenzhen_20250115/
│       ├── distance_matrix.csv
│       └── delivery_config.csv
└── scripts/
    └── logistics-router-interactive.py
```

## 完整示例

### 示例1：快递配送（5个配送点）

**distance_matrix.csv:**
```csv
,W001,D001,D002,D003,D004,D005
W001,0,5.2,8.3,12.1,15.6,18.2
D001,5.2,0,4.1,9.8,13.2,16.4
D002,8.3,4.1,0,6.5,10.3,14.1
D003,12.1,9.8,6.5,0,5.4,9.2
D004,15.6,13.2,10.3,5.4,0,6.8
D005,18.2,16.4,14.1,9.2,6.8,0
```

**delivery_config.csv:**
```csv
node_id,demand,priority,service_time,time_window_start,time_window_end
D001,50,high,15,09:00,12:00
D002,30,medium,10,10:00,14:00
D003,80,high,20,08:00,11:00
D004,45,low,12,13:00,17:00
D005,60,medium,18,09:30,13:30
```

### 示例2：生鲜配送（时间窗严格）

**fresh_delivery_distance_matrix.csv:**
```csv
,W001,Store1,Store2,Store3
W001,0,3.5,6.2,8.9
Store1,3.5,0,4.8,7.1
Store2,6.2,4.8,0,5.3
Store3,8.9,7.1,5.3,0
```

**fresh_delivery_config.csv:**
```csv
node_id,demand,priority,service_time,time_window_start,time_window_end
Store1,120,high,30,06:00,08:00
Store2,95,high,25,06:00,08:00
Store3,150,high,35,06:00,08:00
```

## 常见问题

**Q: 节点ID必须以W或D开头吗？**
A: 不是必须的，但建议使用前缀便于识别。系统会自动识别以W开头的为仓库，其他为配送点。

**Q: 距离矩阵必须是对称的吗？**
A: 不需要。系统支持非对称距离（如单行道、不同方向拥堵情况不同）。

**Q: 如果不提供配送点配置文件会怎样？**
A: 系统会为所有非仓库节点生成默认配置（需求50、优先级medium、服务时间15分钟、时间窗08:00-18:00）。

**Q: 可以只导入距离矩阵，之后再手动调整配送点参数吗？**
A: 目前不支持，建议在导入时就准备好配置文件，或使用默认值后修改代码中的参数。

**Q: 距离矩阵的单位是什么？**
A: 默认为公里（km）。系统会根据距离计算行驶时间（距离 × 1.5~2.5）。

## 与其他导入方式的对比

| 特性 | 距离矩阵导入 | 完整CSV导入 | 手动输入 |
|------|------------|------------|----------|
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 灵活性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 数据详细度 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 适合规模 | 小-中 | 中-大 | 小 |
| 网络复杂度 | 简单 | 复杂 | 简单 |

选择建议：
- 小规模问题（<20个配送点）→ 距离矩阵导入
- 复杂网络（多层级、非完全图）→ 完整CSV导入
- 快速测试 → 手动输入
