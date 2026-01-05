# CSV/Excel 导入指南

## 文件格式说明

### 1. nodes.csv (节点信息)
定义网络中的所有节点(仓库、配送点、中转站等)

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| id | 字符串 | 节点唯一标识 | warehouse, point_a |
| latitude | 浮点数 | 纬度 | 23.1291 |
| longitude | 浮点数 | 经度 | 113.2644 |
| type | 字符串 | 节点类型 | warehouse/delivery/transit |

示例:
```csv
id,latitude,longitude,type
warehouse,23.1291,113.2644,warehouse
point_a,23.1350,113.2700,delivery
point_b,23.1200,113.2500,delivery
```

### 2. edges.csv (边/道路信息)
定义节点之间的连接关系和路径属性

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| from_node | 字符串 | 起点节点ID | warehouse |
| to_node | 字符串 | 终点节点ID | point_a |
| distance | 浮点数 | 距离(km) | 5.2 |
| min_time | 浮点数 | 最短时间(分钟) | 8 |
| max_time | 浮点数 | 最长时间(分钟) | 12 |

示例:
```csv
from_node,to_node,distance,min_time,max_time
warehouse,point_a,5.2,8,12
warehouse,point_b,7.5,12,18
point_a,point_c,4.1,6,10
```

### 3. deliveries.csv (配送点信息)
定义配送任务的详细需求

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| id | 字符串 | 配送点ID(需在nodes.csv中存在) | point_a |
| demand | 浮点数 | 货物需求量(kg) | 50 |
| service_time | 浮点数 | 服务时间(分钟) | 15 |
| priority | 整数 | 优先级(1-5) | 1 |
| time_window_start | 日期时间 | 时间窗开始 | 2024-01-01 09:00:00 |
| time_window_end | 日期时间 | 时间窗结束 | 2024-01-01 12:00:00 |

示例:
```csv
id,demand,service_time,priority,time_window_start,time_window_end
point_a,50,15,1,2024-01-01 09:00:00,2024-01-01 12:00:00
point_b,30,10,2,2024-01-01 10:00:00,2024-01-01 14:00:00
```

### 4. distance_matrix.csv (距离矩阵)
直接定义节点之间的距离矩阵(可选格式)

```csv
,warehouse,point_a,point_b,point_c
warehouse,0,5.2,7.5,9.8
point_a,5.2,0,6.1,4.1
point_b,7.5,6.1,0,3.8
point_c,9.8,4.1,3.8,0
```

## 使用方法

### 方式1: 导出模板
```
选择: 4. 导出CSV模板
输入导出目录
编辑生成的模板文件
重新运行程序导入
```

### 方式2: 从CSV导入
```
选择: 2. 从CSV文件导入
输入包含三个CSV文件的目录路径
系统自动读取并构建网络
```

### 方式3: 从距离矩阵导入
```
选择: 3. 从距离矩阵导入
输入distance_matrix.csv文件路径
系统自动生成节点和边
补充配送点配置信息
```

## 复杂网络支持

系统支持以下复杂网络结构:

1. **多层网络**: 仓库 → 中转站 → 配送点
2. **非完全图**: 不是所有节点之间都直接连通
3. **非对称距离**: A到B和B到A的距离可以不同
4. **动态时间**: 同一路径的时间有最小和最大值范围
5. **多约束**: 时间窗、容量、优先级等多种约束

## 注意事项

1. 所有CSV文件必须使用UTF-8编码
2. 配送点ID必须在nodes.csv中定义
3. edges.csv定义的节点必须在nodes.csv中存在
4. 时间格式统一为: YYYY-MM-DD HH:MM:SS
5. 距离和时间单位要保持一致
6. 建议先使用导出模板功能查看标准格式
