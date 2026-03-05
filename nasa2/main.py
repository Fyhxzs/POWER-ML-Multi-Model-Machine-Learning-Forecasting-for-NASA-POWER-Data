"""
NASA POWER气象数据开发项目
功能：数据仓库构建、ETL流程、数据分析、可视化

"""

import sqlite3
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, silhouette_score
try:
    from sklearn.cluster import KMeans
    HAS_KMEANS = True  # 标记KMeans是否可用
except ImportError:
    print("⚠️ scikit-learn版本不支持KMeans，聚类分析将不可用")
    HAS_KMEANS = False
try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True  # 标记LightGBM是否可用
except ImportError:
    print("⚠️ LightGBM不可用，将使用其他预测方法")
    HAS_LIGHTGBM = False
from datetime import datetime
import time
import os
import warnings
from sqlalchemy import create_engine  # 用于SQLAlchemy数据库连接
import dask  # 用于并行计算
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import traceback  # 用于异常追踪

# 设置中文字体支持，确保图表能显示中文
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    print("✅ 中文字体支持已启用")
except:
    print("⚠️ 无法设置中文字体，图表可能无法正确显示中文")

# === 环境配置 ===
warnings.filterwarnings('ignore')  # 忽略警告
dask.config.set(scheduler='threads')  # 设置Dask使用多线程调度器
os.makedirs('etl_logs', exist_ok=True)  # 创建ETL日志目录
os.makedirs('data_warehouse', exist_ok=True)  # 创建数据仓库目录
os.makedirs('powerbi_output', exist_ok=True)  # 创建Power BI输出目录
os.makedirs('visualizations', exist_ok=True)  # 创建可视化目录

# ====================== 1. 数据库开发与ETL流程 ======================
class DataWarehouse:
    """数据仓库管理类，负责数据库初始化、ETL流程和数据处理"""

    def __init__(self):
        """初始化数据仓库连接"""
        self.db_path = 'data_warehouse/nasa_power_dw.db'  # SQLite数据库路径
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)  # 确保目录存在
        self.conn = sqlite3.connect(self.db_path)  # 创建SQLite连接
        self.engine = create_engine(f'sqlite:///{self.db_path}')  # 创建SQLAlchemy引擎
        self._initialize_database()  # 初始化数据库结构

    def _initialize_database(self):
        """初始化数据仓库结构 - 创建必要的表和索引"""
        try:
            with self.conn:  # 使用事务
                # === 创建ODS层（操作数据存储）表 ===
                # 存储原始抽取数据
                self.conn.execute("""
                CREATE TABLE IF NOT EXISTS ods_raw_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自增主键
                    variable TEXT NOT NULL,                -- 变量名称
                    value REAL NOT NULL,                   -- 变量值
                    latitude REAL NOT NULL,                -- 纬度
                    longitude REAL NOT NULL,                -- 经度
                    date DATE NOT NULL,                    -- 日期
                    load_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- 加载时间戳
                )
                """)

                # === 创建DWD层（数据仓库明细层）表 ===
                # 存储清洗转换后的数据
                self.conn.execute("""
                CREATE TABLE IF NOT EXISTS dwd_processed_data (
                    date DATE PRIMARY KEY,                  -- 日期（主键）
                    allsky_sfc_sw_dwn REAL,                 -- 地表短波辐射
                    cloud_amt REAL,                         -- 云量
                    allsky_sfc_par_tot REAL,                -- 光合有效辐射总量
                    clrsky_sfc_sw_dwn REAL,                 -- 晴空地表短波辐射
                    toa_sw_dwn REAL,                       -- 大气顶短波辐射
                    allsky_sfc_lw_dwn REAL,                 -- 地表长波辐射
                    allsky_sfc_sw_up REAL,                  -- 地表向上短波辐射
                    allsky_sfc_uv_index REAL,               -- UV指数
                    aod_55 REAL,                            -- 550nm气溶胶光学厚度
                    pw REAL,                                -- 可降水量
                    etl_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- ETL时间戳
                )
                """)

                # === 创建DWS层（数据仓库服务层）表 ===
                # 存储每日汇总数据
                self.conn.execute("""
                CREATE TABLE IF NOT EXISTS dws_daily_summary (
                    date DATE PRIMARY KEY,                  -- 日期（主键）
                    avg_radiation REAL,                     -- 平均辐射量
                    avg_cloud_cover REAL,                   -- 平均云量
                    avg_precipitable_water REAL,            -- 平均可降水量
                    max_uv_index REAL,                      -- 最大UV指数
                    cluster_id INTEGER DEFAULT -1,          -- 聚类ID
                    prediction REAL DEFAULT -1,              -- 预测值
                    update_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- 更新时间戳
                )
                """)

                # === 创建ETL日志表 ===
                self.conn.execute("""
                CREATE TABLE IF NOT EXISTS etl_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,    -- 日志ID
                    process_name TEXT NOT NULL,              -- 处理名称
                    status TEXT NOT NULL,                   -- 状态
                    records_processed INTEGER,              -- 处理记录数
                    start_time TIMESTAMP,                   -- 开始时间
                    end_time TIMESTAMP,                     -- 结束时间
                    error_message TEXT                      -- 错误信息
                )
                """)

                # === 创建索引以提高查询性能 ===
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ods_date ON ods_raw_data(date)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ods_variable ON ods_raw_data(variable)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_dwd_date ON dwd_processed_data(date)")

            print("✅ 数据仓库初始化完成")
            return True
        except Exception as e:
            print(f"❌ 数据仓库初始化失败: {str(e)}")
            traceback.print_exc()
            return False

    def create_stored_procedures(self):
        """创建存储过程（模拟）"""
        try:
            print("创建存储过程...")
            with self.conn:
                # SQL: 创建存储过程日志表
                self.conn.execute("""
                CREATE TABLE IF NOT EXISTS sp_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sp_name TEXT NOT NULL,
                    run_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)

                # SQL: 插入示例日志
                self.conn.execute("""
                INSERT INTO sp_log (sp_name) VALUES ('generate_daily_summary')
                """)

            self.log_etl_process("create_stored_procedures", "COMPLETED")
            print("✅ 存储过程创建完成")
            return True
        except Exception as e:
            self.log_etl_process("create_stored_procedures", "FAILED", error=e)
            print(f"❌ 存储过程创建失败: {str(e)}")
            return False

    def log_etl_process(self, process_name, status, records=None, error=None):
        """记录ETL日志"""
        start_time = datetime.now()
        try:
            with self.conn:
                cursor = self.conn.cursor()

                # SQL: 检查日志表是否存在
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='etl_log'")
                table_exists = cursor.fetchone()

                if not table_exists:
                    # 如果日志表不存在，重新创建
                    self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS etl_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        process_name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        records_processed INTEGER,
                        start_time TIMESTAMP,
                        end_time TIMESTAMP,
                        error_message TEXT
                    )
                    """)

                # SQL: 插入新的日志记录
                cursor.execute("""
                INSERT INTO etl_log (process_name, status, records_processed, start_time)
                VALUES (?, ?, ?, ?)
                """, (process_name, status, records, start_time))

                log_id = cursor.lastrowid

                # 如果处理完成，更新结束时间
                if status == 'COMPLETED':
                    cursor.execute("""
                    UPDATE etl_log SET end_time = ? 
                    WHERE id = ?
                    """, (datetime.now(), log_id))

                # 如果有错误信息，更新错误字段
                if error:
                    cursor.execute("""
                    UPDATE etl_log SET error_message = ? 
                    WHERE id = ?
                    """, (str(error)[:500], log_id))  # 限制错误信息长度

            return True
        except Exception as e:
            print(f"ETL日志记录失败: {str(e)}")
            return False

    def etl_extract(self, ds, start_date, end_date, lat_range=(20, 50), lon_range=(100, 130)):
        """ETL抽取阶段：从NASA POWER数据源获取数据"""
        try:
            start_time = time.time()
            print(f"开始ETL抽取阶段({start_date} 至 {end_date})...")

            # 选择指定区域和时间范围
            ds_region = ds.sel(lat=slice(*lat_range), lon=slice(*lon_range))
            ds_full = ds_region.sel(time=slice(start_date, end_date))

            # 目标气象变量
            target_vars = [
                'ALLSKY_SFC_SW_DWN', 'CLOUD_AMT', 'ALLSKY_SFC_PAR_TOT',
                'CLRSKY_SFC_SW_DWN', 'TOA_SW_DWN', 'ALLSKY_SFC_LW_DWN',
                'ALLSKY_SFC_SW_UP', 'ALLSKY_SFC_UV_INDEX', 'AOD_55', 'PW'
            ]

            records = 0
            for var in target_vars:
                if var in ds_full.variables:
                    print(f"  处理变量: {var}")
                    # 计算区域平均值
                    var_data = ds_full[var].mean(dim=['lat', 'lon']).compute()

                    # 转换为DataFrame
                    df_var = var_data.to_dataframe().reset_index()
                    df_var = df_var[['time', var]]
                    df_var.columns = ['date', 'value']
                    df_var['variable'] = var
                    df_var['latitude'] = (lat_range[0] + lat_range[1]) / 2  # 区域平均纬度
                    df_var['longitude'] = (lon_range[0] + lon_range[1]) / 2  # 区域平均经度

                    # SQL: 存入ODS层表
                    df_var[['variable', 'value', 'latitude', 'longitude', 'date']].to_sql(
                        'ods_raw_data', self.engine, if_exists='append', index=False)

                    records += len(df_var)
                    print(f"    ✅ 已处理 {len(df_var)} 条记录")

            self.log_etl_process("etl_extract", "COMPLETED", records)
            print(f"✅ ETL抽取完成，共处理{records}条记录")
            return True
        except Exception as e:
            self.log_etl_process("etl_extract", "FAILED", error=e)
            print(f"❌ ETL抽取失败: {str(e)}")
            traceback.print_exc()
            return False

    def etl_transform(self):
        """ETL转换阶段：数据清洗、转换并加载到DWD层"""
        try:
            start_time = time.time()
            print("开始ETL转换阶段...")

            # SQL: 从ODS层读取数据，按日期和变量分组求平均值
            query = """
            SELECT 
                date,
                variable,
                AVG(value) AS avg_value
            FROM ods_raw_data
            GROUP BY date, variable
            """
            df_raw = pd.read_sql(query, self.conn)

            # 如果没有数据，生成样本数据
            if df_raw.empty:
                print("⚠️ ODS层无数据，使用样本数据")
                return self.create_sample_data()

            # 数据透视：行转列（日期为索引，变量为列）
            df_pivot = df_raw.pivot_table(
                index='date', columns='variable', values='avg_value'
            ).reset_index()

            # 数据清洗：处理缺失值（先向前填充，再向后填充）
            df_cleaned = df_pivot.fillna(method='ffill').fillna(method='bfill')

            # 列名映射：原始变量名 -> 数据库字段名
            column_mapping = {
                'ALLSKY_SFC_SW_DWN': 'allsky_sfc_sw_dwn',
                'CLOUD_AMT': 'cloud_amt',
                'ALLSKY_SFC_PAR_TOT': 'allsky_sfc_par_tot',
                'CLRSKY_SFC_SW_DWN': 'clrsky_sfc_sw_dwn',
                'TOA_SW_DWN': 'toa_sw_dwn',
                'ALLSKY_SFC_LW_DWN': 'allsky_sfc_lw_dwn',
                'ALLSKY_SFC_SW_UP': 'allsky_sfc_sw_up',
                'ALLSKY_SFC_UV_INDEX': 'allsky_sfc_uv_index',
                'AOD_55': 'aod_55',
                'PW': 'pw'
            }

            # 只保留存在的列
            existing_columns = [col for col in column_mapping.keys() if col in df_cleaned.columns]
            df_cleaned = df_cleaned[['date'] + existing_columns]

            # 重命名列
            df_cleaned.rename(columns={col: column_mapping[col] for col in existing_columns}, inplace=True)

            # SQL: 保存到DWD层表（替换模式）
            df_cleaned.to_sql('dwd_processed_data', self.engine, if_exists='replace', index=False)

            self.log_etl_process("etl_transform", "COMPLETED", len(df_cleaned))
            print(f"✅ ETL转换完成，共处理{len(df_cleaned)}条记录")
            return df_cleaned
        except Exception as e:
            self.log_etl_process("etl_transform", "FAILED", error=e)
            print(f"❌ ETL转换失败: {str(e)}")
            return self.create_sample_data()

    def create_sample_data(self):
        """创建样本数据（增强版）- 当真实数据不可用时使用"""
        print("生成增强版样本数据...")
        date_range = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        t = np.linspace(0, 4 * np.pi, len(date_range))  # 时间序列参数

        # 生成模拟气象数据（带季节性和噪声）
        radiation = 150 + 100 * np.sin(t)
        cloud_cover = 50 + 40 * np.sin(0.75 * t)
        precipitable_water = 2.5 + 2 * np.sin(0.5 * t)
        uv_index = 3 + 3 * np.sin(t)
        aod = 0.3 + 0.2 * np.sin(0.25 * t)

        # 添加季节性和随机噪声
        seasonal_factor = np.sin(2 * np.pi * np.arange(len(date_range)) / 365)

        sample_data = {
            'date': date_range,
            'allsky_sfc_sw_dwn': np.clip(radiation + 20 * seasonal_factor + np.random.normal(0, 15, len(date_range)), 50, 350),
            'cloud_amt': np.clip(cloud_cover + 10 * seasonal_factor + np.random.normal(0, 8, len(date_range)), 0, 100),
            'pw': np.clip(precipitable_water + 0.5 * seasonal_factor + np.random.normal(0, 0.4, len(date_range)), 0.5, 5.5),
            'aod_55': np.clip(aod + 0.1 * seasonal_factor + np.random.normal(0, 0.05, len(date_range)), 0.1, 0.8),
            'allsky_sfc_uv_index': np.clip(uv_index + 1.5 * seasonal_factor + np.random.normal(0, 0.6, len(date_range)), 1, 12),
            'clrsky_sfc_sw_dwn': np.clip(radiation * 1.2 + 25 * seasonal_factor + np.random.normal(0, 18, len(date_range)), 60, 400),
            'toa_sw_dwn': np.clip(radiation * 1.5 + 30 * seasonal_factor + np.random.normal(0, 22, len(date_range)), 80, 500)
        }
        df_sample = pd.DataFrame(sample_data)

        # SQL: 保存样本数据到DWD层
        df_sample.to_sql('dwd_processed_data', self.engine, if_exists='replace', index=False)
        print(f"✅ 增强版样本数据已保存到DWD层 ({len(df_sample)}条记录)")
        return df_sample

    def generate_daily_summary(self):
        """生成每日汇总数据（DWS层）"""
        try:
            print("生成每日汇总数据...")
            with self.conn:
                cursor = self.conn.cursor()

                # SQL: 检查DWD表是否存在
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='dwd_processed_data'")
                dwd_exists = cursor.fetchone()

                if not dwd_exists:
                    print("⚠️ DWD表不存在，无法生成每日汇总")
                    return False

                # SQL: 清空现有DWS数据
                cursor.execute("DELETE FROM dws_daily_summary")

                # SQL: 从DWD层插入汇总数据
                cursor.execute("""
                INSERT INTO dws_daily_summary (date, avg_radiation, avg_cloud_cover, avg_precipitable_water)
                SELECT 
                    date,
                    allsky_sfc_sw_dwn AS avg_radiation,    -- 平均辐射量
                    cloud_amt AS avg_cloud_cover,          -- 平均云量
                    pw AS avg_precipitable_water           -- 平均可降水量
                FROM dwd_processed_data
                """)

                records = cursor.rowcount
                self.log_etl_process("generate_daily_summary", "COMPLETED", records)
                print(f"✅ 每日汇总生成完成，共处理{records}条记录")
                return True
        except Exception as e:
            self.log_etl_process("generate_daily_summary", "FAILED", error=e)
            print(f"❌ 每日汇总生成失败: {str(e)}")
            return False

    def update_cluster_predictions(self, df_cluster_pred):
        """更新聚类和预测结果到DWS层"""
        try:
            if df_cluster_pred.empty:
                print("⚠️ 没有聚类和预测数据可更新")
                return False

            print("更新聚类和预测结果到数据仓库...")
            with self.conn:
                cursor = self.conn.cursor()

                # SQL: 创建临时表存储新数据
                df_cluster_pred.to_sql('temp_cluster_pred', self.engine, if_exists='replace', index=False)

                # SQL: 使用批量更新DWS表中的聚类和预测结果
                cursor.execute("""
                UPDATE dws_daily_summary
                SET cluster_id = (
                    SELECT cluster_id FROM temp_cluster_pred 
                    WHERE temp_cluster_pred.date = dws_daily_summary.date
                ),
                prediction = (
                    SELECT prediction FROM temp_cluster_pred 
                    WHERE temp_cluster_pred.date = dws_daily_summary.date
                )
                WHERE EXISTS (
                    SELECT 1 FROM temp_cluster_pred 
                    WHERE temp_cluster_pred.date = dws_daily_summary.date
                )
                """)

                # SQL: 删除临时表
                cursor.execute("DROP TABLE IF EXISTS temp_cluster_pred")

                updated_count = cursor.rowcount
                print(f"✅ 成功更新{updated_count}条记录的聚类和预测结果")
                return True
        except Exception as e:
            print(f"❌ 更新聚类和预测结果失败: {str(e)}")
            return False

    def prepare_powerbi_data(self):
        """准备Power BI分析数据"""
        try:
            print("准备Power BI数据...")
            with self.conn:
                cursor = self.conn.cursor()

                # SQL: 检查必要的表是否存在
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='dws_daily_summary'")
                dws_exists = cursor.fetchone()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='dwd_processed_data'")
                dwd_exists = cursor.fetchone()

                if not dws_exists or not dwd_exists:
                    print("⚠️ 必要的表不存在，无法准备Power BI数据")
                    return False

            # SQL: 从DWS和DWD层联合查询数据
            query = """
            SELECT 
                s.date,
                s.avg_radiation,
                s.avg_cloud_cover,
                s.avg_precipitable_water,
                COALESCE(s.cluster_id, -1) AS cluster_id,             -- 处理空值
                COALESCE(s.prediction, -1) AS prediction,             -- 处理空值
                COALESCE(d.allsky_sfc_uv_index, -1) AS uv_index,      -- UV指数
                COALESCE(d.aod_55, -1) AS aerosol_depth               -- 气溶胶深度
            FROM dws_daily_summary s
            LEFT JOIN dwd_processed_data d ON s.date = d.date  -- 左连接获取更多细节数据
            """

            df_summary = pd.read_sql(query, self.conn)

            # 添加日期相关特征
            df_summary['date'] = pd.to_datetime(df_summary['date'])
            df_summary['year'] = df_summary['date'].dt.year
            df_summary['month'] = df_summary['date'].dt.month
            df_summary['week'] = df_summary['date'].dt.isocalendar().week
            df_summary['day_of_year'] = df_summary['date'].dt.dayofyear
            df_summary['season'] = df_summary['month'] % 12 // 3 + 1  # 季节计算 (1-4)

            # 处理空值
            df_summary['prediction'] = df_summary['prediction'].fillna(-1)
            df_summary['cluster_id'] = df_summary['cluster_id'].fillna(-1)

            # 计算预测误差（实际值 - 预测值）
            df_summary['pred_error'] = np.where(
                df_summary['prediction'] != -1,
                df_summary['avg_radiation'] - df_summary['prediction'],
                np.nan
            )

            # 保存为Power BI可读取的CSV格式（带BOM头）
            output_path = 'powerbi_output/daily_weather_summary.csv'
            df_summary.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ Power BI数据准备完成，保存到: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Power BI数据准备失败: {str(e)}")
            return False

    def generate_technical_documentation(self):
        """生成技术文档"""
        try:
            print("生成技术文档...")
            # 技术文档内容
            doc_content = """
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
            """

            # 写入Markdown文件
            with open('data_warehouse/technical_documentation.md', 'w') as f:
                f.write(doc_content)

            print("✅ 技术文档生成完成")
            return True
        except Exception as e:
            print(f"❌ 技术文档生成失败: {str(e)}")
            return False

    def inspect_database(self):
        """检查数据库内容"""
        try:
            print("\n=== 数据库内容检查 ===")
            # 检查ODS层
            try:
                # SQL: 查询ODS层记录数
                ods_count = pd.read_sql("SELECT COUNT(*) FROM ods_raw_data", self.conn).iloc[0,0]
                print(f"ODS层记录数: {ods_count}")
            except:
                print("ODS层记录数: 表不存在或为空")

            # 检查DWD层
            try:
                # SQL: 查询DWD层记录数
                dwd_count = pd.read_sql("SELECT COUNT(*) FROM dwd_processed_data", self.conn).iloc[0,0]
                print(f"DWD层记录数: {dwd_count}")

                # SQL: 查询DWD层样本数据
                dwd_sample = pd.read_sql("SELECT * FROM dwd_processed_data LIMIT 5", self.conn)
                print("\nDWD层样本数据:")
                print(dwd_sample)
            except:
                print("DWD层记录数: 表不存在或为空")

            # 检查DWS层
            try:
                # SQL: 查询DWS层记录数
                dws_count = pd.read_sql("SELECT COUNT(*) FROM dws_daily_summary", self.conn).iloc[0,0]
                print(f"\nDWS层记录数: {dws_count}")

                # SQL: 检查空值情况
                null_check = pd.read_sql("""
                SELECT 
                    COUNT(CASE WHEN cluster_id = -1 THEN 1 END) AS null_cluster,
                    COUNT(CASE WHEN prediction = -1 THEN 1 END) AS null_prediction
                FROM dws_daily_summary
                """, self.conn)

                print(f"空cluster_id记录数: {null_check['null_cluster'][0]}")
                print(f"空prediction记录数: {null_check['null_prediction'][0]}")

                # SQL: 查询DWS层样本数据
                dws_sample = pd.read_sql("SELECT * FROM dws_daily_summary LIMIT 5", self.conn)
                print("\nDWS层样本数据:")
                print(dws_sample)
            except:
                print("DWS层记录数: 表不存在或为空")

            # 检查ETL日志
            try:
                # SQL: 查询ETL日志
                etl_logs = pd.read_sql("SELECT process_name, status, records_processed FROM etl_log", self.conn)
                print("\nETL日志记录:")
                print(etl_logs)
            except:
                print("ETL日志记录: 表不存在或为空")

            return True
        except Exception as e:
            print(f"❌ 数据库检查失败: {str(e)}")
            return False

# ====================== 2. 数据分析模块 ======================
class SimpleTorchModel(nn.Module):
    """简单的PyTorch模型用于辐射量预测"""
    def __init__(self, input_size):
        super().__init__()
        # 定义三层全连接网络
        self.fc1 = nn.Linear(input_size, 64)  # 输入层到隐藏层1
        self.fc2 = nn.Linear(64, 32)          # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(32, 1)           # 隐藏层2到输出层
        self.relu = nn.ReLU()                  # ReLU激活函数
        self.dropout = nn.Dropout(0.2)         # Dropout防止过拟合

    def forward(self, x):
        """前向传播"""
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def generate_comprehensive_visualization(df):
    """生成综合可视化报告（增强版）"""
    try:
        # 确保数据框有必要的列
        required_columns = ['date', 'avg_radiation', 'avg_cloud_cover', 'avg_precipitable_water']
        for col in required_columns:
            if col not in df.columns:
                print(f"⚠️ 缺少必要列 {col}，跳过可视化生成")
                return

        # 确保日期是datetime类型
        df['date'] = pd.to_datetime(df['date'])

        # 检查是否有足够数据点
        if len(df) < 10:
            print(f"⚠️ 数据点不足 ({len(df)}个)，跳过可视化生成")
            return

        # 创建图表（6个子图）
        plt.figure(figsize=(16, 14))
        plt.suptitle('NASA POWER气象数据综合分析报告', fontsize=18, fontweight='bold')

        # === 子图1: 辐射量趋势 ===
        plt.subplot(3, 2, 1)
        plt.plot(df['date'], df['avg_radiation'], 'b-', alpha=0.7, linewidth=1.5, label='实际值')

        # 如果存在有效的预测值，添加到图表
        if 'prediction' in df.columns and df['prediction'].notnull().any():
            valid_mask = (df['prediction'] != -1) & (df['prediction'].notnull())
            if valid_mask.sum() > 0:
                plt.plot(df.loc[valid_mask, 'date'], df.loc[valid_mask, 'prediction'],
                        'r--', alpha=0.7, linewidth=1.5, label='预测值')

                # 添加训练集/测试集分割线（如果数据包含2023年之前）
                if df['date'].min() < pd.Timestamp('2023-01-01'):
                    plt.axvline(x=pd.Timestamp('2023-01-01'), color='k', linestyle='--', alpha=0.7)
                    plt.text(pd.Timestamp('2023-01-01'), plt.ylim()[1]*0.95, '测试集开始',
                            horizontalalignment='right', fontsize=10,
                            bbox=dict(facecolor='white', alpha=0.8))

        plt.title('每日平均辐射量趋势', fontsize=14)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('辐射量 (W/m²)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xticks(rotation=45)
        plt.legend()

        # === 子图2: 实际值 vs 预测值 ===
        plt.subplot(3, 2, 2)
        if 'prediction' in df.columns and df['prediction'].notnull().any():
            valid_mask = (df['prediction'] != -1) & (df['prediction'].notnull())
            valid_data = df[valid_mask]

            if len(valid_data) >= 5:
                # 添加数据点密度热力图
                sns.kdeplot(
                    x=valid_data['avg_radiation'],
                    y=valid_data['prediction'],
                    cmap="Blues",
                    fill=True,
                    alpha=0.3,
                    levels=10
                )

                # 添加散点图
                plt.scatter(
                    valid_data['avg_radiation'],
                    valid_data['prediction'],
                    alpha=0.6,
                    c='orange',
                    edgecolor='k',
                    s=50
                )

                # 添加对角线（理想预测线）
                min_val = min(valid_data['avg_radiation'].min(), valid_data['prediction'].min())
                max_val = max(valid_data['avg_radiation'].max(), valid_data['prediction'].max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)

                plt.title('实际值 vs 预测值', fontsize=14)
                plt.xlabel('实际辐射量 (W/m²)', fontsize=12)
                plt.ylabel('预测辐射量 (W/m²)', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.3)

                # 添加性能指标
                r2 = r2_score(valid_data['avg_radiation'], valid_data['prediction'])
                rmse = np.sqrt(mean_squared_error(valid_data['avg_radiation'], valid_data['prediction']))
                mae = mean_absolute_error(valid_data['avg_radiation'], valid_data['prediction'])

                text_str = f'R² = {r2:.4f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}'
                plt.text(0.05, 0.95, text_str,
                         transform=plt.gca().transAxes,
                         verticalalignment='top',
                         fontsize=11,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            else:
                plt.text(0.5, 0.5, '无有效预测数据',
                         horizontalalignment='center', verticalalignment='center',
                         fontsize=14, color='gray')
                plt.axis('off')
        else:
            plt.text(0.5, 0.5, '无预测数据',
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=14, color='gray')
            plt.axis('off')

        # === 子图3: 近期预测对比 ===
        plt.subplot(3, 2, 3)
        if 'prediction' in df.columns and df['prediction'].notnull().any():
            valid_mask = (df['prediction'] != -1) & (df['prediction'].notnull())
            valid_data = df[valid_mask]

            if len(valid_data) >= 30:
                # 选择最近90天数据
                recent_data = valid_data.sort_values('date').tail(90)

                plt.plot(recent_data['date'], recent_data['avg_radiation'],
                         'b-', alpha=0.7, linewidth=1.5, label='实际值')
                plt.plot(recent_data['date'], recent_data['prediction'],
                         'r--', alpha=0.7, linewidth=1.5, label='预测值')

                plt.title('近期预测对比', fontsize=14)
                plt.xlabel('日期', fontsize=12)
                plt.ylabel('辐射量 (W/m²)', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.xticks(rotation=45)
                plt.legend()
            else:
                plt.text(0.5, 0.5, '数据不足',
                         horizontalalignment='center', verticalalignment='center',
                         fontsize=14, color='gray')
                plt.axis('off')
        else:
            plt.text(0.5, 0.5, '无预测数据',
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=14, color='gray')
            plt.axis('off')

        # === 子图4: 聚类结果可视化 ===
        plt.subplot(3, 2, 4)
        if 'cluster_id' in df.columns and df['cluster_id'].nunique() > 1:
            # 为不同聚类分配颜色
            unique_clusters = sorted(df['cluster_id'].unique())
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

            # 绘制聚类分布
            for cluster, color in zip(unique_clusters, colors):
                cluster_data = df[df['cluster_id'] == cluster]
                plt.scatter(
                    cluster_data['date'],
                    cluster_data['avg_radiation'],
                    color=color,
                    alpha=0.6,
                    s=30,
                    label=f'聚类 {cluster}'
                )

            plt.title('聚类结果分布', fontsize=14)
            plt.xlabel('日期', fontsize=12)
            plt.ylabel('辐射量 (W/m²)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.xticks(rotation=45)
            plt.legend(loc='upper left', fontsize=9)
        else:
            plt.text(0.5, 0.5, '无聚类数据',
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=14, color='gray')
            plt.axis('off')

        # === 子图5: 云量分布 ===
        plt.subplot(3, 2, 5)
        plt.hist(df['avg_cloud_cover'], bins=30, alpha=0.7, color='green', edgecolor='k')
        plt.title('云量分布', fontsize=14)
        plt.xlabel('云量 (%)', fontsize=12)
        plt.ylabel('频率', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)

        # 添加统计信息
        mean_cloud = df['avg_cloud_cover'].mean()
        median_cloud = df['avg_cloud_cover'].median()
        plt.axvline(mean_cloud, color='r', linestyle='--', label=f'均值: {mean_cloud:.1f}')
        plt.axvline(median_cloud, color='b', linestyle='--', label=f'中位数: {median_cloud:.1f}')
        plt.legend()

        # === 子图6: 辐射量与可降水量关系 ===
        plt.subplot(3, 2, 6)
        plt.scatter(df['avg_radiation'], df['avg_precipitable_water'],
                   alpha=0.6, c='purple', s=50)
        plt.title('辐射量与可降水量关系', fontsize=14)
        plt.xlabel('辐射量 (W/m²)', fontsize=12)
        plt.ylabel('可降水量 (cm)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)

        # 添加回归线
        try:
            from scipy.stats import linregress
            # 过滤有效数据点
            mask = ~np.isnan(df['avg_radiation']) & ~np.isnan(df['avg_precipitable_water'])
            # 计算线性回归
            slope, intercept, r_value, p_value, std_err = linregress(
                df.loc[mask, 'avg_radiation'], df.loc[mask, 'avg_precipitable_water'])
            x = np.linspace(df['avg_radiation'].min(), df['avg_radiation'].max(), 100)
            plt.plot(x, intercept + slope*x, 'r-', label=f'回归线 (R²={r_value**2:.2f})')
            plt.legend()
        except Exception as e:
            print(f"回归线添加失败: {str(e)}")

        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # 保存图表
        plt.savefig('visualizations/comprehensive_weather_report.png', dpi=150, bbox_inches='tight')
        print("✅ 综合气象报告图表已生成")

    except Exception as e:
        print(f"❌ 综合可视化生成失败: {str(e)}")
        traceback.print_exc()

def analyze_data(df, model_type='random_forest'):
    """
    执行数据分析
    model_type: 可选 'random_forest', 'lightgbm', 'pytorch'
    """
    print(f"\n=== 执行数据分析（使用 {model_type}） ===")
    result_df = pd.DataFrame()  # 存储结果

    # 数据验证
    if 'date' not in df.columns:
        print("⚠️ 数据框中缺少'date'列，跳过分析")
        return result_df

    # 准备数据
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()  # 设置日期索引并排序

    if len(df) < 30:
        print(f"⚠️ 数据不足 ({len(df)}条)，跳过分析")
        return result_df

    # 目标变量：地表短波辐射
    target_var = 'allsky_sfc_sw_dwn'

    # 检查目标变量是否存在
    if target_var not in df.columns:
        print(f"⚠️ 目标变量 {target_var} 不存在，跳过预测")
        df['prediction'] = -1  # 设置默认值
    else:
        # 选择特征（排除目标变量和聚类ID）
        features = [col for col in df.columns
                  if col != target_var
                  and col != 'cluster_id'
                  and not col.startswith('date')]

        if not features:
            print("⚠️ 没有可用的特征变量，跳过预测")
            df['prediction'] = -1
        else:
            print(f"执行预测: {target_var} (使用特征: {features})")
            X = df[features].values
            y = df[target_var].values

            # 处理缺失值
            X = np.nan_to_num(X, nan=0.0)
            y = np.nan_to_num(y, nan=0.0)

            # 根据模型类型选择预测方法
            if model_type == 'random_forest':
                df = random_forest_predictor(X, y, df)
            elif model_type == 'lightgbm' and HAS_LIGHTGBM:
                df = lightgbm_predictor(X, y, df)
            elif model_type == 'pytorch':
                df = torch_predictor(X, y, df, features)
            else:
                print(f"⚠️ 模型 {model_type} 不可用，默认使用随机森林")
                df = random_forest_predictor(X, y, df)

    # 聚类分析
    if HAS_KMEANS:
        cluster_features = ['allsky_sfc_sw_dwn', 'cloud_amt', 'pw', 'aod_55', 'allsky_sfc_uv_index']
        available_features = [col for col in cluster_features if col in df.columns]

        if len(available_features) >= 2:
            print(f"执行K-means聚类，使用特征: {available_features}")
            df = perform_clustering(df, available_features)
        else:
            print(f"⚠️ 可用于聚类的特征不足: {available_features}")
            df['cluster_id'] = 0
    else:
        print("⚠️ KMeans不可用，跳过聚类分析")
        df['cluster_id'] = 0

    # 准备结果
    result_df = df.reset_index()[['date', 'cluster_id', 'prediction']]

    # 处理空值
    result_df['prediction'] = result_df['prediction'].fillna(-1)
    result_df['cluster_id'] = result_df['cluster_id'].fillna(-1)

    return result_df

def random_forest_predictor(X, y, df):
    """使用随机森林进行辐射量预测"""
    try:
        print("使用随机森林进行预测...")
        # 划分训练集和测试集（2023年前为训练，2023年后为测试）
        train_mask = df.index < pd.Timestamp('2023-01-01')
        test_mask = df.index >= pd.Timestamp('2023-01-01')

        if not any(train_mask) or not any(test_mask):
            print("⚠️ 无法划分训练集和测试集，使用全量数据")
            X_train, y_train = X, y
            X_test = X
        else:
            X_train, y_train = X[train_mask], y[train_mask]
            X_test = X[test_mask]

        # 创建随机森林模型
        model = RandomForestRegressor(
            n_estimators=150,    # 树的数量
            max_depth=10,        # 树的最大深度
            min_samples_leaf=5,  # 叶节点最小样本数
            random_state=42,     # 随机种子
            n_jobs=-1            # 使用所有CPU核心
        )

        # 训练模型
        model.fit(X_train, y_train)

        # 预测整个数据集
        predictions = model.predict(X)

        # 处理异常预测值（辐射量应在0-500 W/m²之间）
        valid_mask = (predictions >= 0) & (predictions <= 500)
        predictions[~valid_mask] = -1  # 无效值标记为-1

        df['prediction'] = predictions

        # 评估测试集性能
        if any(test_mask):
            y_test = y[test_mask]
            test_pred = predictions[test_mask]
            valid_test_mask = test_pred != -1  # 只考虑有效预测

            if any(valid_test_mask):
                r2 = r2_score(y_test[valid_test_mask], test_pred[valid_test_mask])
                rmse = np.sqrt(mean_squared_error(y_test[valid_test_mask], test_pred[valid_test_mask]))
                print(f"✅ 随机森林预测完成，测试集R²={r2:.4f}, RMSE={rmse:.4f}")
            else:
                print("✅ 随机森林预测完成，但无有效测试集预测")
        else:
            print("✅ 随机森林预测完成")

        return df
    except Exception as e:
        print(f"❌ 随机森林预测失败: {str(e)}")
        df['prediction'] = -1  # 预测失败时设置默认值
        return df

def lightgbm_predictor(X, y, df):
    """使用LightGBM进行辐射量预测"""
    try:
        print("使用LightGBM进行预测...")
        # 划分训练集和测试集
        train_mask = df.index < pd.Timestamp('2023-01-01')
        test_mask = df.index >= pd.Timestamp('2023-01-01')

        if not any(train_mask) or not any(test_mask):
            print("⚠️ 无法划分训练集和测试集，使用全量数据")
            X_train, y_train = X, y
            X_test = X
        else:
            X_train, y_train = X[train_mask], y[train_mask]
            X_test = X[test_mask]

        # 创建LightGBM模型
        model = LGBMRegressor(
            n_estimators=200,     # 树的数量
            max_depth=7,          # 树的最大深度
            learning_rate=0.05,   # 学习率
            random_state=42,      # 随机种子
            n_jobs=-1             # 使用所有CPU核心
        )

        # 训练模型
        model.fit(X_train, y_train)

        # 预测整个数据集
        predictions = model.predict(X)

        # 处理异常预测值
        valid_mask = (predictions >= 0) & (predictions <= 500)
        predictions[~valid_mask] = -1

        df['prediction'] = predictions

        # 评估测试集性能
        if any(test_mask):
            y_test = y[test_mask]
            test_pred = predictions[test_mask]
            valid_test_mask = test_pred != -1

            if any(valid_test_mask):
                r2 = r2_score(y_test[valid_test_mask], test_pred[valid_test_mask])
                rmse = np.sqrt(mean_squared_error(y_test[valid_test_mask], test_pred[valid_test_mask]))
                print(f"✅ LightGBM预测完成，测试集R²={r2:.4f}, RMSE={rmse:.4f}")
            else:
                print("✅ LightGBM预测完成，但无有效测试集预测")
        else:
            print("✅ LightGBM预测完成")

        return df
    except Exception as e:
        print(f"❌ LightGBM预测失败: {str(e)}")
        df['prediction'] = -1
        return df

def torch_predictor(X, y, df, features):
    """使用PyTorch神经网络进行辐射量预测"""
    try:
        print("使用PyTorch神经网络进行预测...")
        # 划分训练集和测试集
        train_mask = df.index < pd.Timestamp('2023-01-01')
        test_mask = df.index >= pd.Timestamp('2023-01-01')

        if not any(train_mask) or not any(test_mask):
            print("⚠️ 无法划分训练集和测试集，使用全量数据")
            X_train, y_train = X, y
            X_test = X
            y_test = y
        else:
            X_train, y_train = X[train_mask], y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]

        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)  # 转换为列向量
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

        # 创建数据集和数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # 创建模型
        model = SimpleTorchModel(X.shape[1])  # 输入大小为特征数
        criterion = nn.HuberLoss()  # 使用Huber损失函数（对异常值不敏感）
        optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)  # AdamW优化器

        # 训练模型
        model.train()
        for epoch in range(150):  # 训练150个epoch
            epoch_loss = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()  # 梯度清零
                outputs = model(inputs)  # 前向传播
                loss = criterion(outputs, targets)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
                epoch_loss += loss.item()  # 累加损失

            # 每30个epoch打印一次损失
            if (epoch + 1) % 30 == 0:
                print(f'Epoch {epoch + 1}, Loss: {epoch_loss/len(train_loader):.4f}')

        # 预测
        model.eval()  # 切换到评估模式
        with torch.no_grad():  # 不计算梯度
            # 预测整个数据集
            X_full_scaled = scaler.transform(X)
            X_full_tensor = torch.FloatTensor(X_full_scaled)
            predictions = model(X_full_tensor).numpy().flatten()

        # 处理异常预测值
        valid_mask = (predictions >= 0) & (predictions <= 500)
        predictions[~valid_mask] = -1

        df['prediction'] = predictions

        # 评估测试集性能
        if any(test_mask):
            test_pred = predictions[test_mask]
            valid_test_mask = test_pred != -1

            if any(valid_test_mask):
                r2 = r2_score(y_test[valid_test_mask], test_pred[valid_test_mask])
                rmse = np.sqrt(mean_squared_error(y_test[valid_test_mask], test_pred[valid_test_mask]))
                print(f"✅ PyTorch预测完成，测试集R²={r2:.4f}, RMSE={rmse:.4f}")
            else:
                print("✅ PyTorch预测完成，但无有效测试集预测")
        else:
            print("✅ PyTorch预测完成")

        return df
    except Exception as e:
        print(f"❌ PyTorch预测失败: {str(e)}")
        df['prediction'] = -1
        return df

def perform_clustering(df, features):
    """执行K-means聚类分析"""
    try:
        # 数据标准化
        scaler = StandardScaler()
        cluster_data = scaler.fit_transform(df[features])

        # 确定最佳聚类数 (3-6) 使用轮廓系数
        best_score = -np.inf  # 最佳轮廓系数
        best_k = 3  # 最佳聚类数

        for k in range(3, 7):  # 尝试3-6个聚类
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(cluster_data)
            score = silhouette_score(cluster_data, kmeans.labels_)

            if score > best_score:
                best_score = score
                best_k = k

        print(f"选择聚类数: k={best_k} (轮廓系数={best_score:.4f})")

        # 应用最佳聚类
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        df['cluster_id'] = kmeans.fit_predict(cluster_data)

        # 分析聚类特征
        cluster_stats = df.groupby('cluster_id')[features].mean()
        print("聚类特征统计:")
        print(cluster_stats)

        # 可视化聚类结果（至少需要两个特征）
        if len(features) >= 2:
            plt.figure(figsize=(10, 6))
            plt.scatter(df[features[0]], df[features[1]], c=df['cluster_id'], cmap='viridis', alpha=0.6)
            plt.xlabel(features[0])
            plt.ylabel(features[1])
            plt.title(f'天气模式聚类 (k={best_k})')
            plt.colorbar(label='Cluster ID')
            plt.savefig('visualizations/weather_clusters.png')
            plt.close()
            print("✅ 聚类可视化已保存")

        return df
    except Exception as e:
        print(f"❌ 聚类分析失败: {str(e)}")
        df['cluster_id'] = -1  # 聚类失败时设置默认值
        return df

# ====================== 3. 主流程 ======================
def main():
    """项目主函数"""
    print("=== NASA POWER气象数据开发项目 ===")
    start_time = time.time()  # 记录开始时间

    # 清理旧数据仓库（如果存在）
    if os.path.exists('data_warehouse'):
        print("清理旧数据仓库...")
        try:
            # 递归删除目录
            for root, dirs, files in os.walk('data_warehouse', topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir('data_warehouse')
            print("✅ 旧数据仓库已删除")
        except Exception as e:
            print(f"❌ 删除旧数据仓库失败: {str(e)}")

    # 重新创建目录
    os.makedirs('data_warehouse', exist_ok=True)
    os.makedirs('etl_logs', exist_ok=True)
    os.makedirs('powerbi_output', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)

    # 初始化数据仓库
    print("初始化数据仓库...")
    dw = DataWarehouse()
    dw.create_stored_procedures()

    # 尝试加载NASA POWER数据
    ds = None
    try:
        print("尝试加载NASA POWER数据...")
        # NASA POWER数据URL（Zarr格式）
        url = 'https://nasa-power.s3.amazonaws.com/syn1deg/temporal/power_syn1deg_daily_temporal_lst.zarr'
        ds = xr.open_dataset(url, engine='zarr', consolidated=True, chunks={'time': 100})
        print("✅ NASA POWER数据加载成功")
    except Exception as e:
        print(f"❌ NASA POWER数据加载失败: {str(e)}")
        print("⚠️ 将使用样本数据模式")

    # 执行ETL流程
    if ds is not None:
        # 处理2020-2023年数据
        print("开始ETL抽取...")
        success = dw.etl_extract(ds, '2020-01-01', '2023-12-31')
        if not success:
            print("⚠️ ETL抽取失败，使用样本数据")
            df_cleaned = dw.create_sample_data()
        else:
            print("开始ETL转换...")
            df_cleaned = dw.etl_transform()
    else:
        # 直接创建样本数据
        print("使用样本数据模式...")
        df_cleaned = dw.create_sample_data()

    # 生成每日汇总（DWS层）
    print("生成每日汇总数据...")
    dw.generate_daily_summary()

    # 执行数据分析（聚类和预测）
    print("执行数据分析...")
    # 可选择预测模型: 'random_forest', 'lightgbm', 'pytorch'
    df_analysis = analyze_data(df_cleaned.copy(), model_type='random_forest')

    # 更新聚类和预测结果到数据库
    if not df_analysis.empty:
        print("更新聚类和预测结果...")
        dw.update_cluster_predictions(df_analysis)
    else:
        print("⚠️ 没有生成分析结果，跳过数据库更新")

    # 准备BI数据和文档
    print("准备Power BI数据...")
    dw.prepare_powerbi_data()

    print("生成技术文档...")
    dw.generate_technical_documentation()

    # 数据库内容检查
    print("检查数据库内容...")
    dw.inspect_database()

    # 生成可视化图表
    try:
        print("生成可视化图表...")
        # SQL: 从DWS层加载数据
        df_summary = pd.read_sql("SELECT * FROM dws_daily_summary", dw.conn)

        # 确保必要的列存在
        if 'prediction' in df_summary.columns:
            df_summary.rename(columns={'prediction': 'prediction'}, inplace=True)

        if 'cluster_id' not in df_summary.columns:
            df_summary['cluster_id'] = -1
        if 'prediction' not in df_summary.columns:
            df_summary['prediction'] = -1

        # 生成综合可视化报告
        generate_comprehensive_visualization(df_summary)
    except Exception as e:
        print(f"❌ 可视化数据加载失败: {str(e)}")
        # 尝试使用样本数据生成可视化
        try:
            print("尝试使用样本数据生成可视化...")
            # 创建模拟数据
            sample_data = {
                'date': pd.date_range('2020-01-01', '2023-12-31', freq='D'),
                'avg_radiation': 150 + 100 * np.sin(np.linspace(0, 4 * np.pi, 1461)),
                'avg_cloud_cover': 50 + 40 * np.sin(np.linspace(0, 3 * np.pi, 1461)),
                'avg_precipitable_water': 2.5 + 2 * np.sin(np.linspace(0, 2 * np.pi, 1461)),
                'cluster_id': np.random.choice([0, 1, 2], size=1461),
                'prediction': 150 + 100 * np.sin(np.linspace(0, 4 * np.pi, 1461) + np.random.normal(0, 20, 1461))
            }
            df_sample = pd.DataFrame(sample_data)
            generate_comprehensive_visualization(df_sample)
        except Exception as e2:
            print(f"❌ 样本可视化也失败: {str(e2)}")

    # 项目总结
    elapsed_time = time.time() - start_time
    print(f"\n=== 项目执行完成 ===")
    print(f"总耗时: {elapsed_time:.2f}秒")
    print("输出文件:")
    print(f"- {dw.db_path} (SQLite数据仓库)")

    if os.path.exists('powerbi_output/daily_weather_summary.csv'):
        print("- powerbi_output/daily_weather_summary.csv (Power BI数据)")
    else:
        print("⚠️ Power BI数据文件未生成")

    if os.path.exists('data_warehouse/technical_documentation.md'):
        print("- data_warehouse/technical_documentation.md (技术文档)")
    else:
        print("⚠️ 技术文档未生成")

    # 检查可视化图表
    vis_path = 'visualizations/comprehensive_weather_report.png'
    if os.path.exists(vis_path):
        print(f"- {vis_path} (综合气象报告)")
    else:
        print("⚠️ 综合气象报告未生成，请检查日志")

    cluster_path = 'visualizations/weather_clusters.png'
    if os.path.exists(cluster_path):
        print(f"- {cluster_path} (聚类可视化)")
    else:
        print("⚠️ 聚类可视化未生成")

if __name__ == "__main__":
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 修复可能的拼写错误
    try:
        torch.manual_seed(42)
    except:
        pass

    # 启动主程序
    main()