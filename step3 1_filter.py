"""
Step 3.2 - 基于数据驱动关键词过滤无关评论和帖子
处理思路：
  1. 使用 extract_topic_keywords_from_data.py 提取的关键词（数据驱动）
  2. 删除那些与关键词完全不相关的评论和帖子文本
  3. 输出清洗后的评论和帖子数据文件
"""

import pandas as pd
from pathlib import Path


# ==================== 配置 ====================
INPUT_DIR = Path('step1')
OUTPUT_DIR = Path('step3')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 从数据中自动提取的议题关键词（来自 extract_topic_keywords_from_data.py）
TOPICS_KEYWORDS = {
    '价格补贴': ['补贴', '便宜', '价格', '优惠', '活动', '免费', '价格战', '红包', '赚钱', '亿元', '几块钱', '块钱', '元购', '优惠券', '花钱', '一分钱'],
    '平台竞争': ['美团', '平台', '淘宝', '阿里', '京东', '竞争', '不用', '多多', '对比', '选择', '用美团', '没用'],
    '用户体验': ['用户', '问题', '客服', '服务', '投诉', '喜欢'],
    '配送服务': ['外卖', '骑手', '时间', '配送费', '配送', '晚点', '效率', '速度'],
    '监管政策': ['执法'],
    '商家权益': ['商家', '利润', '成本', '餐饮', '亏损', '收入', '权益', '费用', '商户', '合作', '净利润'],
    '市场格局': ['行业', '市场', '市场份额', '格局'],
    '产品品类': ['奶茶', '电商', '品牌', '产品', '品质', '分类'],
    '消费习惯': ['消费', '消费者', '习惯', '经常', '购买', '下单', '需求'],
}

# 无关内容的关键词（表情、笑声、简短回应等）
IRRELEVANT_KEYWORDS = {
    '哈哈', '哈哈哈', 'doge', '偷笑', '微笑', '害羞', '汗颜', '捂脸', '石化',
    '哭惹', '哭', '笑哭', '笑死', '绝了', '绝笑', '哈哈哈哈', '哈哈哈哈哈', '笑cry',
    '没错', '对对对', '对呀', '赞', '点赞', '同意', '同感', '我也是', '我也',
    '说得好', '说的对', '说得对', '有道理', '同样', '在在', '666', '牛', '厉害',
    '确实', '对', '是', '嗯', '哦', '啊', '呃', '额', '嘿', '呵呵',
    '[', ']', '【', '】', '表情', 'emoji',
    'r', 'R', 'in', 'by', '111', '15', '10',
    '1', '2', '3', '我', '你', '他', '她',
}

# 构建所有议题关键词的集合（用于快速查找）
ALL_TOPIC_KEYWORDS = set()
for keywords in TOPICS_KEYWORDS.values():
    ALL_TOPIC_KEYWORDS.update(keywords)


# ==================== 过滤函数 ====================
def is_irrelevant_text(text: str, min_length: int = 3) -> bool:
    """
    判断文本是否完全无关（@、表情、笑声等）
    
    判断逻辑（简化版）：
    1. 空白或极短 → 无关
    2. 大部分是@符号 → 无关
    3. 大部分是表情/笑声词汇 → 无关
    
    Args:
        text: 文本内容
        min_length: 最小文本长度
    
    Returns:
        True: 文本完全无关，应该被过滤
        False: 文本可能有关，应该被保留
    """
    text_str = str(text).strip()
    text_lower = text_str.lower()
    
    # ========== 判断1：完全空白或极短 ==========
    if len(text_lower) < min_length:
        return True
    
    # ========== 判断2：检查是否大部分是@符号 ==========
    at_count = text_str.count('@')
    if at_count > 0:
        # 统计@相关内容的字符数
        at_related_chars = at_count  # @符号本身
        # 简单估计：每个@后面可能跟5-15个字符的用户名
        at_related_chars += at_count * 8  # 平均估计
        
        # 如果@相关内容占比超过40%，视为无关
        if at_related_chars / len(text_str) > 0.4:
            return True
    
    # ========== 判断3：如果大部分是表情/笑声词汇 ==========
    # 计算无关词汇的总字符数
    matched_chars = 0
    for keyword in IRRELEVANT_KEYWORDS:
        if keyword.lower() in text_lower:
            # 计算该关键词出现的次数
            count = text_lower.count(keyword.lower())
            matched_chars += len(keyword) * count
    
    # 如果60%以上的内容是无关关键词，判断为完全无关
    if len(text_lower) > 0 and matched_chars / len(text_lower) > 0.6:
        return True
    
    # 其他情况视为可能有关，保留
    return False


# ==================== 主处理流程 ====================
def main():
    print("=" * 90)
    print("Step 3.2 - 基于数据驱动关键词过滤无关评论和帖子")
    print("=" * 90)
    
    print("\n【使用的关键词体系】")
    print(f"  • 议题数量：{len(TOPICS_KEYWORDS)}")
    print(f"  • 总关键词数：{len(ALL_TOPIC_KEYWORDS)}")
    for topic, keywords in TOPICS_KEYWORDS.items():
        print(f"  • {topic}: {len(keywords)} 个关键词")
    
    # ========== 1. 加载原始数据 ==========
    print("\n【步骤1】加载原始数据...")
    posts_file = INPUT_DIR / 'posts_processed.csv'
    comments_file = INPUT_DIR / 'comments_processed.csv'
    
    if not posts_file.exists() or not comments_file.exists():
        print(f"❌ 错误：找不到数据文件")
        return
    
    posts = pd.read_csv(posts_file, encoding='utf-8-sig')
    comments = pd.read_csv(comments_file, encoding='utf-8-sig')
    
    original_posts_count = len(posts)
    original_comments_count = len(comments)
    
    print(f"✓ 已加载帖子数据：{original_posts_count} 条")
    print(f"✓ 已加载评论数据：{original_comments_count} 条")
    
    # ========== 2. 过滤无关评论 ==========
    print("\n【步骤2】过滤无关评论...")
    comments['is_irrelevant'] = comments['text'].fillna('').apply(lambda x: is_irrelevant_text(x, min_length=3))
    irrelevant_comments_count = comments['is_irrelevant'].sum()
    
    # 移除无关评论
    comments_filtered = comments[~comments['is_irrelevant']].drop('is_irrelevant', axis=1).reset_index(drop=True)
    filtered_comments_count = len(comments_filtered)
    
    print(f"✓ 评论过滤完成！")
    print(f"  • 原始评论数：{original_comments_count} 条")
    print(f"  • 过滤的无关评论：{irrelevant_comments_count} 条 （占比 {irrelevant_comments_count/original_comments_count*100:.2f}%）")
    print(f"  • 保留的有效评论：{filtered_comments_count} 条 （占比 {filtered_comments_count/original_comments_count*100:.2f}%）")
    
    # ========== 3. 过滤无关帖子 ==========
    print("\n【步骤3】过滤无关帖子...")
    posts['is_irrelevant'] = posts['text'].fillna('').apply(lambda x: is_irrelevant_text(x, min_length=10))
    irrelevant_posts_count = posts['is_irrelevant'].sum()
    
    # 移除无关帖子
    posts_filtered = posts[~posts['is_irrelevant']].drop('is_irrelevant', axis=1).reset_index(drop=True)
    filtered_posts_count = len(posts_filtered)
    
    print(f"✓ 帖子过滤完成！")
    print(f"  • 原始帖子数：{original_posts_count} 条")
    print(f"  • 过滤的无关帖子：{irrelevant_posts_count} 条 （占比 {irrelevant_posts_count/original_posts_count*100:.2f}%）")
    print(f"  • 保留的有效帖子：{filtered_posts_count} 条 （占比 {filtered_posts_count/original_posts_count*100:.2f}%）")
    
    # ========== 4. 保存清洗后的数据 ==========
    print("\n【步骤4】保存清洗后的数据...")
    
    # 保存评论
    comments_output_file = OUTPUT_DIR / 'comments_filtered.csv'
    comments_filtered.to_csv(comments_output_file, index=False, encoding='utf-8-sig')
    print(f"✓ 已保存：{comments_output_file}")
    print(f"  • 文件大小：{filtered_comments_count} 行 × {len(comments_filtered.columns)} 列")
    
    # 保存帖子
    posts_output_file = OUTPUT_DIR / 'posts_filtered.csv'
    posts_filtered.to_csv(posts_output_file, index=False, encoding='utf-8-sig')
    print(f"✓ 已保存：{posts_output_file}")
    print(f"  • 文件大小：{filtered_posts_count} 行 × {len(posts_filtered.columns)} 列")
    
    # ========== 5. 生成过滤统计报告 ==========
    print("\n【步骤5】生成过滤统计报告...")
    report_rows = [
        {'数据类型': '帖子', '原始数量': original_posts_count, '过滤数量': irrelevant_posts_count, 
         '保留数量': filtered_posts_count, '过滤率': f"{irrelevant_posts_count/original_posts_count*100:.2f}%"},
        {'数据类型': '评论', '原始数量': original_comments_count, '过滤数量': irrelevant_comments_count,
         '保留数量': filtered_comments_count, '过滤率': f"{irrelevant_comments_count/original_comments_count*100:.2f}%"},
        {'数据类型': '总计', '原始数量': original_posts_count + original_comments_count, 
         '过滤数量': irrelevant_posts_count + irrelevant_comments_count,
         '保留数量': filtered_posts_count + filtered_comments_count,
         '过滤率': f"{(irrelevant_posts_count + irrelevant_comments_count)/(original_posts_count + original_comments_count)*100:.2f}%"},
    ]
    report_df = pd.DataFrame(report_rows)
    
    report_file = OUTPUT_DIR / 'filter_report.csv'
    report_df.to_csv(report_file, index=False, encoding='utf-8-sig')
    print(f"✓ 已保存：{report_file}")
    
    # 打印统计信息
    print("\n【过滤统计摘要】")
    print(report_df.to_string(index=False))
    
    # ========== 6. 样本展示 ==========
    print("\n【被过滤的评论样本（无关内容）】（显示前5条）")
    if irrelevant_comments_count > 0:
        irrelevant_samples = comments[comments['is_irrelevant']]['text'].head(5)
        for i, sample in enumerate(irrelevant_samples, 1):
            sample_text = str(sample)[:60] + ('...' if len(str(sample)) > 60 else '')
            print(f"  {i}. {sample_text}")
    else:
        print("  （无）")
    
    print("\n【保留的评论样本（有关内容）】（显示前5条）")
    relevant_samples = comments_filtered['text'].head(5)
    for i, sample in enumerate(relevant_samples, 1):
        sample_text = str(sample)[:60] + ('...' if len(str(sample)) > 60 else '')
        print(f"  {i}. {sample_text}")
    
    if irrelevant_posts_count > 0:
        print("\n【被过滤的帖子样本（无关内容）】（显示前3条）")
        irrelevant_posts_samples = posts[posts['is_irrelevant']]['text'].head(3)
        for i, sample in enumerate(irrelevant_posts_samples, 1):
            sample_text = str(sample)[:80] + ('...' if len(str(sample)) > 80 else '')
            print(f"  {i}. {sample_text}")
    
    print("\n【保留的帖子样本（有关内容）】（显示前3条）")
    relevant_posts_samples = posts_filtered['text'].head(3)
    for i, sample in enumerate(relevant_posts_samples, 1):
        sample_text = str(sample)[:80] + ('...' if len(str(sample)) > 80 else '')
        print(f"  {i}. {sample_text}")
    
    # ========== 7. 完成提示 ==========
    print("\n" + "=" * 90)
    print("✓ 处理完成！")
    print("=" * 90)
    print("\n【输出文件】")
    print(f"  1. {comments_output_file}")
    print(f"     └─ 清洗后的评论数据（{filtered_comments_count}条有效评论）")
    print(f"  2. {posts_output_file}")
    print(f"     └─ 清洗后的帖子数据（{filtered_posts_count}条有效帖子）")
    print(f"  3. {report_file}")
    print(f"     └─ 过滤统计报告")
    
    print("\n【数据质量提升】")
    print(f"  • 评论：从 {original_comments_count} 条精简到 {filtered_comments_count} 条")
    print(f"  • 帖子：从 {original_posts_count} 条精简到 {filtered_posts_count} 条")
    print(f"  • 总数据：从 {original_posts_count + original_comments_count} 条精简到 {filtered_posts_count + filtered_comments_count} 条")
    print(f"  • 数据纯净度提升：过滤了 {(irrelevant_posts_count + irrelevant_comments_count)/(original_posts_count + original_comments_count)*100:.1f}% 的无关内容")
    
    print("\n【后续步骤】")
    print("  1. 使用清洗后的数据进行议题分类和分析")
    print("  2. 数据文件路径：")
    print(f"     - 评论：step3/comments_filtered.csv")
    print(f"     - 帖子：step3/posts_filtered.csv")
    print("=" * 90)
    
    return posts_filtered, comments_filtered


if __name__ == '__main__':
    posts_filtered, comments_filtered = main()
