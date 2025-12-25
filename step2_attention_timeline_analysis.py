"""
小红书文本分析 - 第二步：消费者关注度随时间波动分析
根据《小红书文本分析详细操作指导.md》第2部分实现
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
from pathlib import Path
from datetime import datetime

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ==================== 加载预处理数据 ====================
print("=" * 60)
print("加载预处理数据...")
print("=" * 60)

posts = pd.read_csv('merged_results/posts_processed.csv', encoding='utf-8-sig')
comments = pd.read_csv('merged_results/comments_processed.csv', encoding='utf-8-sig')

# 确保时间字段为 datetime 类型
posts['time'] = pd.to_datetime(posts['time'])
comments['create_time'] = pd.to_datetime(comments['create_time'])

print(f"✓ 帖子数据：{len(posts)} 条")
print(f"✓ 评论数据：{len(comments)} 条\n")

# ==================== 2.1 关键事件时间线整理 ====================
print("=" * 60)
print("2.1 关键事件时间线整理")
print("=" * 60)

# 根据外卖大战的真实事件时间线定义关键事件
# 参考《小红书文本分析详细操作指导.md》第2部分
key_events = pd.DataFrame({
    'date': [
        '2025-04-01',  # 京东推出外卖业务
        '2025-05-13',  # 五部门约谈外卖平台
        '2025-06-01',  # 京东完成达达集团整合
        '2025-06-15',  # 补贴战白热化开始
        '2025-07-18',  # 监管再次约谈并明确新规
        '2025-08-01',  # 三大平台承诺抵制恶性竞争
        '2025-09-01',  # 竞争策略分化
        '2025-10-15',  # 反不正当竞争法施行
        '2025-11-01',  # 三季度财报披露
    ],
    'event_name': [
        '京东外卖上线百亿补贴',
        '五部门约谈外卖平台',
        '京东整合达达集团',
        '补贴战白热化',
        '监管约谈明确新规',
        '三大平台承诺抵制恶性竞争',
        '竞争策略分化',
        '反不正当竞争法施行',
        '三季度财报披露亏损'
    ],
    'description': [
        '京东正式推出外卖业务并上线"百亿补贴"；阿里巴巴升级即时零售业务为"淘宝闪购"',
        '市场监管总局等五部门针对恶性竞争等问题约谈主要外卖平台',
        '京东完成对达达集团的私有化整合，成立本地生活服务事业群',
        '补贴战白热化，京东、美团、淘宝闪购日订单量均创下峰值',
        '市场监管总局因恶性补贴问题再次约谈平台，明确新规将于10月15日生效',
        '美团、饿了么、京东三大平台同日发布声明，承诺抵制恶性竞争',
        '竞争策略分化，京东收缩补贴，阿里和京东从不同场景切入挑战美团',
        '新修订的《反不正当竞争法》正式施行，行业竞争重点转向算法与效率',
        '主要平台三季度财报披露因价格战导致巨额亏损，高管表态将收缩投入'
    ]
})

key_events['date'] = pd.to_datetime(key_events['date'])

print("外卖大战关键事件时间线：")
print("-" * 60)
for idx, row in key_events.iterrows():
    print(f"{row['date'].strftime('%Y年%m月%d日')} - {row['event_name']}")
    print(f"  说明：{row['description']}")
    print()

print()

# ==================== 2.2 关注度统计 ====================
print("=" * 60)
print("2.2 关注度统计与可视化")
print("=" * 60)

# 按月统计
posts['month'] = posts['time'].dt.to_period('M')
comments['month'] = comments['create_time'].dt.to_period('M')

post_counts = posts.groupby('month').size()
comment_counts = comments.groupby('month').size()

print("\n【帖子月度统计】")
print(post_counts)

print("\n【评论月度统计】")
print(comment_counts)

# 获取所有出现过的月份（以帖子月份为主）
all_months = post_counts.index
post_counts_aligned = post_counts.reindex(all_months, fill_value=0)
comment_counts_aligned = comment_counts.reindex(all_months, fill_value=0)

# 转换为字符串格式便于绘图
post_counts_str = post_counts_aligned.index.astype(str)

print(f"\n帖子发布月份范围：{post_counts_str[0]} ~ {post_counts_str[-1]}")
print(f"评论时间月份范围：{comment_counts.index.astype(str)[0]} ~ {comment_counts.index.astype(str)[-1]}")

# ==================== 2.3 生成可视化图表 ====================
print("\n" + "=" * 60)
print("生成可视化图表...")
print("=" * 60)

# 创建图表 - 简洁版设计
fig, ax = plt.subplots(figsize=(18, 8))

# 使用双Y轴
months = range(len(post_counts_aligned))

# 左Y轴 - 帖子数（使用柱状图）
bars = ax.bar(months, post_counts_aligned.values, alpha=0.7, width=0.5,
              label='帖子数', color='#FF6B6B', edgecolor='white', linewidth=2,
              zorder=2)

# 在柱子上标注数值
for i, (bar, val) in enumerate(zip(bars, post_counts_aligned.values)):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.2, f'{int(val)}',
               ha='center', va='bottom', fontsize=11, fontweight='bold',
               color='#C0392B')

# 右Y轴 - 评论数（使用面积图）
ax2 = ax.twinx()
ax2.plot(months, comment_counts_aligned.values, marker='o', linewidth=3.5,
         markersize=12, label='评论数', color='#3498DB',
         markerfacecolor='#3498DB', markeredgewidth=2.5,
         markeredgecolor='white', alpha=0.95, zorder=3)
ax2.fill_between(months, comment_counts_aligned.values, 
                 alpha=0.25, color='#3498DB', zorder=1)

# 在评论数据点上标注数值
for i, val in enumerate(comment_counts_aligned.values):
    if val > 0:
        ax2.text(i, val + 20, f'{int(val)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color='#21618C', bbox=dict(boxstyle='round,pad=0.4',
                facecolor='white', alpha=0.8, edgecolor='#3498DB', linewidth=1.5))

# 标注关键事件 - 简洁版
event_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
               '#E67E22', '#1ABC9C', '#34495E', '#E91E63']

# 创建月份到事件的映射
month_events = {}
for idx, row in key_events.iterrows():
    event_date = row['date']
    month_period = str(event_date.to_period('M'))
    
    if month_period in post_counts_aligned.index.astype(str):
        month_idx = list(post_counts_aligned.index.astype(str)).index(month_period)
        if month_idx not in month_events:
            month_events[month_idx] = []
        month_events[month_idx].append((idx, row['event_name'], event_colors[idx % len(event_colors)]))

# 只在图表顶部用编号标记
for month_idx, events in month_events.items():
    for idx, event_name, color in events:
        # 在顶部用编号标记
        ax2.plot(month_idx, ax2.get_ylim()[1] * 0.97, marker='v', markersize=15,
                color=color, markeredgewidth=2, markeredgecolor='white', zorder=5)

# 设置标签和标题
ax.set_xlabel('月份', fontsize=14, fontweight='bold', color='#2C3E50', labelpad=10)
ax.set_ylabel('帖子数', fontsize=14, fontweight='bold', color='#C0392B', labelpad=10)
ax2.set_ylabel('评论数', fontsize=14, fontweight='bold', color='#21618C', labelpad=10)

# 添加主标题
fig.suptitle('消费者关注度随时间波动分析 - 外卖大战热度追踪', fontsize=16, fontweight='bold',
            color='#2C3E50', y=0.96)

# 设置 x 轴
ax.set_xticks(months)

# 创建x轴标签 - 在有事件的月份下方添加事件名称
x_labels = []
for i, month_str in enumerate(post_counts_str):
    label = month_str
    if i in month_events:
        # 如果这个月有事件，添加事件名称（多个事件换行）
        event_names = [event_name for _, event_name, _ in month_events[i]]
        if event_names:
            label = month_str + '\n' + '\n'.join(event_names)
    x_labels.append(label)

ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=10, fontweight='normal')
ax.tick_params(axis='both', labelsize=11, length=6, width=1.5)
ax.tick_params(axis='y', labelcolor='#C0392B', labelsize=12)
ax2.tick_params(axis='y', labelcolor='#21618C', labelsize=12)

# 设置Y轴范围
ax.set_ylim(0, max(post_counts_aligned.values) * 1.3)
ax2.set_ylim(0, max(comment_counts_aligned.values) * 1.15)

# 美化网格 - 只显示水平网格线
ax.yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=1, color='#BDC3C7')
ax.set_axisbelow(True)

# 图例 - 更美观的样式
legend1 = ax.legend(loc='upper left', fontsize=13, frameon=True,
                   shadow=True, fancybox=True, framealpha=0.95,
                   edgecolor='#C0392B', facecolor='white')
legend2 = ax2.legend(loc='upper right', fontsize=13, frameon=True,
                    shadow=True, fancybox=True, framealpha=0.95,
                    edgecolor='#21618C', facecolor='white')

# 设置背景色 - 渐变效果
ax.set_facecolor('#FDFEFE')
fig.patch.set_facecolor('#FFFFFF')

# 设置边框
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
    ax2.spines[spine].set_visible(False)
    
ax.spines['left'].set_color('#C0392B')
ax.spines['left'].set_linewidth(2.5)
ax.spines['bottom'].set_color('#34495E')
ax.spines['bottom'].set_linewidth(2.5)
ax2.spines['right'].set_color('#21618C')
ax2.spines['right'].set_linewidth(2.5)

# 调整布局
plt.subplots_adjust(bottom=0.18, top=0.92, left=0.08, right=0.92)

# 保存图表
output_dir = Path('step2')
output_dir.mkdir(parents=True, exist_ok=True)

chart_path = output_dir / '关注度随时间变化.png'
plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ 图表已保存：{chart_path}\n")

plt.close()
print("[已完成] 月度可视化")

# ==================== 2.4 分周期分析 ====================
print("=" * 60)
print("2.4 时间周期分析")
print("=" * 60)

# 按周统计
posts['week'] = posts['time'].dt.to_period('W')
comments['week'] = comments['create_time'].dt.to_period('W')

posts_weekly = posts.groupby('week').size()
comments_weekly = comments.groupby('week').size()

print(f"\n【周度统计】")
print(f"帖子周度统计（前5周）：")
print(posts_weekly.head())
print(f"\n评论周度统计（前5周）：")
print(comments_weekly.head())

# 生成周度可视化 - 优化版
fig, ax = plt.subplots(figsize=(18, 7))

# 对齐周度数据
all_weeks = posts_weekly.index
posts_weekly_aligned = posts_weekly.reindex(all_weeks, fill_value=0)
comments_weekly_aligned = comments_weekly.reindex(all_weeks, fill_value=0)

weeks = range(len(posts_weekly_aligned))

# 使用渐变色柱状图
colors_posts = plt.cm.Reds(0.5 + posts_weekly_aligned.values / posts_weekly_aligned.max() * 0.4)
ax.bar(weeks, posts_weekly_aligned.values, alpha=0.8, width=0.6, 
       label='帖子数', color=colors_posts, edgecolor='white', linewidth=1.5)

# 第二个Y轴 - 评论数折线图
ax2 = ax.twinx()
ax2.plot(weeks, comments_weekly_aligned.values, marker='o', linewidth=3, 
         markersize=8, label='评论数', color='#3498DB', 
         markeredgewidth=2, markeredgecolor='white', alpha=0.9, zorder=3)
ax2.fill_between(weeks, comments_weekly_aligned.values, alpha=0.1, color='#3498DB')

# 标注高峰周
max_comments_week_idx = comments_weekly_aligned.values.argmax()
max_comments_value = comments_weekly_aligned.values[max_comments_week_idx]
ax2.annotate(f'高峰周\n{max_comments_value}条评论', 
            xy=(max_comments_week_idx, max_comments_value),
            xytext=(max_comments_week_idx, max_comments_value + 50),
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#3498DB', 
                     alpha=0.7, edgecolor='white', linewidth=2),
            arrowprops=dict(arrowstyle='->', color='#3498DB', lw=2),
            color='white')

# 设置标签
ax.set_xlabel('周份', fontsize=13, fontweight='bold', color='#2C3E50')
ax.set_ylabel('帖子数', fontsize=12, color='#E74C3C', fontweight='bold')
ax2.set_ylabel('评论数', fontsize=12, color='#3498DB', fontweight='bold')
ax.set_title('消费者关注度周度分析\n帖子发布与评论互动趋势', 
            fontsize=15, fontweight='bold', pad=15, color='#2C3E50')

# 设置 x 轴标签 - 每隔3周显示
display_step = max(1, len(weeks) // 10)  # 动态调整显示密度
ax.set_xticks(weeks[::display_step])
ax.set_xticklabels(posts_weekly_aligned.index.astype(str)[::display_step], 
                   rotation=45, ha='right', fontsize=9)

# Y轴刻度颜色
ax.tick_params(axis='y', labelcolor='#E74C3C', labelsize=10)
ax2.tick_params(axis='y', labelcolor='#3498DB', labelsize=10)

# 网格美化
ax.grid(True, alpha=0.2, linestyle='--', linewidth=1, axis='y', color='#E74C3C')
ax2.grid(True, alpha=0.2, linestyle='--', linewidth=1, axis='y', color='#3498DB')
ax.set_axisbelow(True)

# 图例
legend1 = ax.legend(loc='upper left', fontsize=11, frameon=True, 
                   shadow=True, fancybox=True, framealpha=0.9)
legend2 = ax2.legend(loc='upper right', fontsize=11, frameon=True,
                    shadow=True, fancybox=True, framealpha=0.9)
legend1.get_frame().set_facecolor('white')
legend1.get_frame().set_edgecolor('#CCCCCC')
legend2.get_frame().set_facecolor('white')
legend2.get_frame().set_edgecolor('#CCCCCC')

# 背景色
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

# 边框
for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)
for spine in ax2.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.5)

plt.tight_layout()

weekly_chart_path = output_dir / '关注度周度分析.png'
plt.savefig(weekly_chart_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ 周度图表已保存：{weekly_chart_path}\n")

plt.close()
print("[已完成] 周度可视化")

# ==================== 2.5 数据统计总结 ====================
print("=" * 60)
print("2.5 数据统计总结")
print("=" * 60)

print(f"\n【关注度统计】")
print(f"帖子总数：{len(posts)} 条")
print(f"评论总数：{len(comments)} 条")
print(f"平均每篇帖子评论数：{len(comments) / len(posts):.2f} 条")

print(f"\n【月度高峰期】")
peak_month_posts = post_counts.idxmax()
peak_month_comments = comment_counts.idxmax()
print(f"帖子发布最多月份：{peak_month_posts}（{post_counts.max()} 条）")
print(f"评论最多月份：{peak_month_comments}（{comment_counts.max()} 条）")

print(f"\n【周度高峰期】")
peak_week_posts = posts_weekly.idxmax()
peak_week_comments = comments_weekly.idxmax()
print(f"帖子发布最多周份：{peak_week_posts}（{posts_weekly.max()} 条）")
print(f"评论最多周份：{peak_week_comments}（{comments_weekly.max()} 条）")

# 保存统计数据为 CSV
summary_stats = pd.DataFrame({
    '指标': ['帖子总数', '评论总数', '平均评论数/帖子', '帖子最多月份', '帖子最多月份数量',
             '评论最多月份', '评论最多月份数量', '帖子最多周份', '帖子最多周份数量',
             '评论最多周份', '评论最多周份数量'],
    '数值': [
        len(posts), len(comments), f"{len(comments) / len(posts):.2f}",
        str(peak_month_posts), post_counts.max(),
        str(peak_month_comments), comment_counts.max(),
        str(peak_week_posts), posts_weekly.max(),
        str(peak_week_comments), comments_weekly.max()
    ]
})

stats_path = output_dir / '关注度统计汇总.csv'
summary_stats.to_csv(stats_path, index=False, encoding='utf-8-sig')
print(f"\n✓ 统计汇总已保存：{stats_path}")

# 保存月度数据为 CSV
monthly_data = pd.DataFrame({
    '月份': post_counts_aligned.index.astype(str),
    '帖子数': post_counts_aligned.values,
    '评论数': comment_counts_aligned.values
})
monthly_path = output_dir / '月度统计数据.csv'
monthly_data.to_csv(monthly_path, index=False, encoding='utf-8-sig')
print(f"✓ 月度数据已保存：{monthly_path}")

# 保存周度数据为 CSV
weekly_data = pd.DataFrame({
    '周份': posts_weekly_aligned.index.astype(str),
    '帖子数': posts_weekly_aligned.values,
    '评论数': comments_weekly_aligned.values
})
weekly_path = output_dir / '周度统计数据.csv'
weekly_data.to_csv(weekly_path, index=False, encoding='utf-8-sig')
print(f"✓ 周度数据已保存：{weekly_path}")

# 保存关键事件为 CSV
events_path = output_dir / '关键事件时间线.csv'
key_events.to_csv(events_path, index=False, encoding='utf-8-sig')
print(f"✓ 关键事件已保存：{events_path}")

print("\n" + "=" * 60)
print("第二步消费者关注度随时间波动分析完成！")
print("=" * 60)

print("\n生成的输出文件：")
print(f"  1. {chart_path}")
print(f"  2. {weekly_chart_path}")
print(f"  3. {stats_path}")
print(f"  4. {monthly_path}")
print(f"  5. {weekly_path}")
print(f"  6. {events_path}")
