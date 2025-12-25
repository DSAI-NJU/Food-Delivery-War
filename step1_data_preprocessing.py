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

# 根据实际编码读取合并后的数据
posts = pd.read_csv('merged_results/final_contents.csv', encoding='utf-8-sig')
comments = pd.read_csv('merged_results/final_comments_2025.csv', encoding='utf-8-sig')

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
posts = posts.drop_duplicates(keep='first')  # 完全一样的数据样本才去重
posts_after = len(posts)
print(f"帖子清洗：{posts_before} -> {posts_after}（删除 {posts_before - posts_after} 条）\n")

comments_before = len(comments)
comments = comments.dropna(subset=['text'])
comments = comments.drop_duplicates(keep='first')  # 完全一样的数据样本才去重
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



# ==================== 数据保存 ====================
print("=" * 50)
print("保存预处理后的数据...")
print("=" * 50)

# 创建处理后的数据目录
output_dir = Path('step1')
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



# ==================== 生成数据结构表格 ====================
print("\n" + "=" * 50)
print("生成数据结构表格...")
print("=" * 50)

def create_structure_table(df, data_name):
    """创建数据结构信息表格"""
    structure_info = []
    
    for col in df.columns:
        info = {
            '列名': col,
            '数据类型': str(df[col].dtype),
            '总行数': len(df),
            '非空值数': df[col].notna().sum(),
            '空值数': df[col].isna().sum(),
            '空值率(%)': f"{df[col].isna().sum()/len(df)*100:.2f}",
            '唯一值数': df[col].nunique()
        }
        
        # 根据数据类型添加额外信息
        if df[col].dtype in ['int64', 'float64']:
            info['最小值'] = df[col].min()
            info['最大值'] = df[col].max()
            info['平均值'] = f"{df[col].mean():.2f}"
            info['中位数'] = df[col].median()
            info['示例值'] = df[col].iloc[0]
        elif 'datetime' in str(df[col].dtype):
            info['最小值'] = df[col].min()
            info['最大值'] = df[col].max()
            info['平均值'] = '-'
            info['中位数'] = '-'
            info['示例值'] = df[col].iloc[0]
        else:  # object类型
            text_lengths = df[col].astype(str).str.len()
            info['最小值'] = f"{text_lengths.min()}字符"
            info['最大值'] = f"{text_lengths.max()}字符"
            info['平均值'] = f"{text_lengths.mean():.1f}字符"
            info['中位数'] = '-'
            sample_val = str(df[col].iloc[0])
            info['示例值'] = sample_val[:50] + '...' if len(sample_val) > 50 else sample_val
        
        structure_info.append(info)
    
    structure_df = pd.DataFrame(structure_info)
    return structure_df

# 生成帖子数据结构表格
posts_structure = create_structure_table(posts, '帖子')
print("\n【帖子数据结构表格】")
print(posts_structure.to_string(index=False))

# 生成评论数据结构表格
comments_structure = create_structure_table(comments, '评论')
print("\n【评论数据结构表格】")
print(comments_structure.to_string(index=False))

# 保存结构表格到CSV
posts_structure.to_csv(output_dir / '帖子数据结构表.csv', index=False, encoding='utf-8-sig')
comments_structure.to_csv(output_dir / '评论数据结构表.csv', index=False, encoding='utf-8-sig')

print(f"\n✓ 帖子数据结构表已保存：{output_dir / '帖子数据结构表.csv'}")
print(f"✓ 评论数据结构表已保存：{output_dir / '评论数据结构表.csv'}")

print("\n" + "=" * 50)
print("数据结构表格生成完成！")
print("=" * 50)
