
            NASA POWER气象数据仓库技术文档
            
            1. 数据仓库架构
            - ODS层 (ods_raw_data): 存储原始抽取数据
            - DWD层 (dwd_processed_data): 存储清洗转换后的明细数据
            - DWS层 (dws_daily_summary): 存储按日汇总的数据
            
            2. ETL流程
            - 抽取: 从NASA POWER源系统获取原始数据
            - 转换: 数据清洗、聚合、格式转换
            - 加载: 加载到目标数据表
            
            3. 数据分析
            - 随机森林回归: 预测辐射量
            - K-means聚类: 天气模式分类
            - PyTorch模型: 深度学习预测
            
            4. 数据字典
            | 表名               | 字段                | 描述                     |
            |--------------------|---------------------|--------------------------|
            | dwd_processed_data | allsky_sfc_sw_dwn   | 地表短波辐射 (W/m²)      |
            | dwd_processed_data | cloud_amt           | 云量 (%)                 |
            | dwd_processed_data | pw                  | 可降水量 (cm)            |
            | dws_daily_summary  | cluster_id          | 天气模式分类ID           |
            | dws_daily_summary  | prediction          | 辐射量预测值             |
            | ...                | ...                 | ...                      |
            
            5. 使用说明
            - 每日运行ETL流程更新数据
            - 使用Power BI连接 daily_weather_summary.csv 进行可视化
            