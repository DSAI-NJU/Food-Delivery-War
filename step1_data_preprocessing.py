"""
小红书文本分析 - 第一步：数据准备与预处理
根据《小红书文本分析详细操作指导.md》第1部分实现
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# ==================== 1.1 数据加载 ====================
print("=" * 50)
print("1.1 数据加载...")
print("=" * 50)

# 根据实际编码读取数据
posts = pd.read_csv('data/data/xhs/csv/final_contents.csv', encoding='gb2312')
comments = pd.read_csv('data/data/xhs/csv/final_comments.csv', encoding='utf-8')

print(f"帖子数据加载完成：{len(posts)} 条")
print(f"帖子字段：{list(posts.columns)}\n")
print(f"评论数据加载完成：{len(comments)} 条")
print(f"评论字段：{list(comments.columns)}\n")

# ==================== 1.2 数据清洗 ====================
print("=" * 50)
print("1.2 数据清洗...")
print("=" * 50)

# 备份原始数据
posts_raw = posts.copy()
comments_raw = comments.copy()

# --- 步骤1：合并帖子的 title 和 desc 为 text 字段 ---
print("\n【步骤1】合并帖子 title 和 desc 为 text 字段...")
posts['title'] = posts['title'].fillna('')
posts['desc'] = posts['desc'].fillna('')
posts['text'] = posts['title'] + ' ' + posts['desc']
print(f"帖子 text 字段生成完成")
print(f"示例：{posts['text'].iloc[0][:100]}...\n")

# 评论直接用 content 字段
if 'content' in comments.columns:
    comments['text'] = comments['content']
    print("评论使用 content 字段作为 text\n")

# --- 步骤2：处理空值和重复 ---
print("【步骤2】去除空值和重复...")
posts_before = len(posts)
posts = posts.dropna(subset=['text'])
posts = posts.drop_duplicates(subset=['text'], keep='first')
posts_after = len(posts)
print(f"帖子清洗：{posts_before} -> {posts_after}（删除 {posts_before - posts_after} 条）\n")

comments_before = len(comments)
comments = comments.dropna(subset=['text'])
comments = comments.drop_duplicates(subset=['text'], keep='first')
comments_after = len(comments)
print(f"评论清洗：{comments_before} -> {comments_after}（删除 {comments_before - comments_after} 条）\n")

# --- 步骤3：乱码检测与过滤 ---
print("【步骤3】乱码检测与过滤...")

def has_garbage(text):
    """检测文本是否包含乱码或异常字符"""
    if pd.isna(text):
        return True
    text = str(text)
    # 检测连续的控制字符、特殊符号过多
    garbage_count = len(re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', text))
    if garbage_count > 0:
        return True
    return False

posts['has_garbage'] = posts['text'].apply(has_garbage)
comments['has_garbage'] = comments['text'].apply(has_garbage)

posts_before = len(posts)
posts = posts[~posts['has_garbage']]
posts.drop('has_garbage', axis=1, inplace=True)
posts_after = len(posts)
print(f"帖子去乱码：{posts_before} -> {posts_after}（删除 {posts_before - posts_after} 条）\n")

comments_before = len(comments)
comments = comments[~comments['has_garbage']]
comments.drop('has_garbage', axis=1, inplace=True)
comments_after = len(comments)
print(f"评论去乱码：{comments_before} -> {comments_after}（删除 {comments_before - comments_after} 条）\n")

# --- 步骤4：时间字段转为 datetime 类型 ---
print("【步骤4】时间字段转为 datetime 类型...")

# 帖子时间
if 'time' in posts.columns:
    posts['time'] = pd.to_datetime(posts['time'], errors='coerce')
    null_time = posts['time'].isna().sum()
    print(f"帖子时间转换完成（{null_time} 个无效时间值）")
else:
    print("警告：帖子数据中未找到 'time' 字段")

# 评论时间
if 'create_time' in comments.columns:
    comments['create_time'] = pd.to_datetime(comments['create_time'], errors='coerce')
    null_time = comments['create_time'].isna().sum()
    print(f"评论时间转换完成（{null_time} 个无效时间值）\n")
else:
    print("警告：评论数据中未找到 'create_time' 字段\n")

# --- 步骤5：地域字段标准化（暂不处理）---
print("【步骤5】地域字段标准化...")
print("根据需求，暂不处理地域字段，后续分析时再进行标准化\n")

# ==================== 数据保存 ====================
print("=" * 50)
print("保存预处理后的数据...")
print("=" * 50)

# 创建处理后的数据目录
output_dir = Path('data/data/xhs/csv/处理')
output_dir.mkdir(parents=True, exist_ok=True)

# 保存处理后的数据
posts.to_csv(output_dir / 'posts_processed.csv', index=False, encoding='utf-8-sig')
comments.to_csv(output_dir / 'comments_processed.csv', index=False, encoding='utf-8-sig')

print(f"✓ 帖子数据已保存：{output_dir / 'posts_processed.csv'}")
print(f"✓ 评论数据已保存：{output_dir / 'comments_processed.csv'}\n")

# ==================== 数据统计与探索 ====================
print("=" * 50)
print("数据预处理完成统计")
print("=" * 50)

print(f"\n【帖子数据统计】")
print(f"  原始数据：{len(posts_raw)} 条")
print(f"  处理后：{len(posts)} 条")
print(f"  清洗比例：{(1 - len(posts)/len(posts_raw))*100:.2f}%")
print(f"  时间范围：{posts['time'].min()} ~ {posts['time'].max()}")

print(f"\n【评论数据统计】")
print(f"  原始数据：{len(comments_raw)} 条")
print(f"  处理后：{len(comments)} 条")
print(f"  清洗比例：{(1 - len(comments)/len(comments_raw))*100:.2f}%")
if 'create_time' in comments.columns:
    print(f"  时间范围：{comments['create_time'].min()} ~ {comments['create_time'].max()}")

print(f"\n【字段信息】")
print(f"帖子字段：{list(posts.columns)}")
print(f"评论字段：{list(comments.columns)}")

# ==================== 数据详细探索 ====================
print("\n" + "=" * 50)
print("数据详细探索")
print("=" * 50)

print("\n" + "="*50)
print("【帖子数据各列详细信息】")
print("="*50)
for col in posts.columns:
    print(f"\n列名: {col}")
    print(f"  数据类型: {posts[col].dtype}")
    print(f"  非空值数: {posts[col].notna().sum()} / {len(posts)} ({posts[col].notna().sum()/len(posts)*100:.1f}%)")
    print(f"  空值数: {posts[col].isna().sum()} ({posts[col].isna().sum()/len(posts)*100:.1f}%)")
    print(f"  唯一值数: {posts[col].nunique()}")
    
    # 根据数据类型显示不同的统计信息
    if posts[col].dtype in ['int64', 'float64']:
        print(f"  数值统计:")
        print(f"    最小值: {posts[col].min()}")
        print(f"    最大值: {posts[col].max()}")
        print(f"    平均值: {posts[col].mean():.2f}")
        print(f"    中位数: {posts[col].median()}")
    elif posts[col].dtype == 'object':
        print(f"  文本长度统计:")
        text_lengths = posts[col].astype(str).str.len()
        print(f"    最短: {text_lengths.min()} 字符")
        print(f"    最长: {text_lengths.max()} 字符")
        print(f"    平均: {text_lengths.mean():.1f} 字符")
        print(f"  前5个高频值:")
        value_counts = posts[col].value_counts().head(5)
        for val, count in value_counts.items():
            val_str = str(val)[:50] + '...' if len(str(val)) > 50 else str(val)
            print(f"    {val_str}: {count} 次")
    elif 'datetime' in str(posts[col].dtype):
        print(f"  时间范围: {posts[col].min()} 至 {posts[col].max()}")
    
    print(f"  示例值: {str(posts[col].iloc[0])[:100]}..." if len(str(posts[col].iloc[0])) > 100 else f"  示例值: {posts[col].iloc[0]}")

print("\n" + "="*50)
print("【评论数据各列详细信息】")
print("="*50)
for col in comments.columns:
    print(f"\n列名: {col}")
    print(f"  数据类型: {comments[col].dtype}")
    print(f"  非空值数: {comments[col].notna().sum()} / {len(comments)} ({comments[col].notna().sum()/len(comments)*100:.1f}%)")
    print(f"  空值数: {comments[col].isna().sum()} ({comments[col].isna().sum()/len(comments)*100:.1f}%)")
    print(f"  唯一值数: {comments[col].nunique()}")
    
    # 根据数据类型显示不同的统计信息
    if comments[col].dtype in ['int64', 'float64']:
        print(f"  数值统计:")
        print(f"    最小值: {comments[col].min()}")
        print(f"    最大值: {comments[col].max()}")
        print(f"    平均值: {comments[col].mean():.2f}")
        print(f"    中位数: {comments[col].median()}")
    elif comments[col].dtype == 'object':
        print(f"  文本长度统计:")
        text_lengths = comments[col].astype(str).str.len()
        print(f"    最短: {text_lengths.min()} 字符")
        print(f"    最长: {text_lengths.max()} 字符")
        print(f"    平均: {text_lengths.mean():.1f} 字符")
        print(f"  前5个高频值:")
        value_counts = comments[col].value_counts().head(5)
        for val, count in value_counts.items():
            val_str = str(val)[:50] + '...' if len(str(val)) > 50 else str(val)
            print(f"    {val_str}: {count} 次")
    elif 'datetime' in str(comments[col].dtype):
        print(f"  时间范围: {comments[col].min()} 至 {comments[col].max()}")
    
    print(f"  示例值: {str(comments[col].iloc[0])[:100]}..." if len(str(comments[col].iloc[0])) > 100 else f"  示例值: {comments[col].iloc[0]}")

print("\n" + "=" * 50)
print("第一步数据准备与预处理完成！")
print("=" * 50)
