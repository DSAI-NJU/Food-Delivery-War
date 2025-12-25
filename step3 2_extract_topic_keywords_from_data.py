"""
从数据中提取各议题的关键词
处理思路：
  1. 预定义9个议题（仅名称）
  2. 对posts和comments数据进行分词和清洗
  3. 使用TF-IDF等方法提取高频关键词
  4. 根据议题名称和语义相关性，将关键词智能分配到各议题
  5. 输出各议题的关键词列表
"""

import pandas as pd
import numpy as np
import re
import jieba
import jieba.analyse
from collections import Counter
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


# ==================== 配置 ====================
INPUT_DIR = Path('step1')
OUTPUT_DIR = Path('step3')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 预定义议题（仅名称，不预先给定关键词）
TOPICS_NAMES = [
    '价格补贴',
    '平台竞争',
    '用户体验',
    '配送服务',
    '监管政策',
    '商家权益',
    '市场格局',
    '产品品类',
    '消费习惯',
]

# 噪音词和通用词（需要过滤）
NOISE_WORDS = {
    '我', '你', '他', '她', '它', '我们', '你们', '他们', '什么', '那么', '这么', '那个', '这个',
    '的', '了', '和', '是', '在', '就', '都', '而', '及', '与', '着', '或', '不', '也', '有', '这', '那',
    '吧', '呢', '啦', '啊', '哦', '呃', '额', '嘿', '哈', '嗯', '呀', '呗',
    '能', '用', '做', '会', '对', '比', '说', '点', '买', '么', '好', '多', '有', '看', '去', '来', '给',
    'r', 'R', 'doge', '哈哈', '哭', '笑', '哈哈哈', '偷笑', '微笑', '捂脸', '笑哭', '流汗', '石化',
    '个', '次', '种', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
    '10', '100', '111', '15', '20', '30',
}

# 议题相关的领域词（用于辅助关键词分配）
TOPIC_RELATED_TERMS = {
    '价格补贴': ['价格', '补贴', '优惠', '便宜', '贵', '钱', '折扣', '红包', '券', '活动', '促销', '省', '实惠', '免费', '元', '费'],
    '平台竞争': ['平台', '竞争', '美团', '京东', '饿了么', '淘宝', '拼多多', '阿里', '对比', '选择', '换', '用', '市场', '份额'],
    '用户体验': ['体验', '用户', '方便', '快捷', '好用', '界面', '操作', '服务', '满意', '推荐', '喜欢', '问题', '投诉', '客服', '功能'],
    '配送服务': ['配送', '外卖', '送餐', '骑手', '快递', '时间', '速度', '准时', '延迟', '送达', '晚', '慢', '快', '效率'],
    '监管政策': ['监管', '政策', '法律', '规定', '约谈', '整改', '处罚', '合规', '执法', '政府', '部门', '违规', '调查'],
    '商家权益': ['商家', '店铺', '商户', '餐饮', '抽成', '佣金', '费用', '成本', '利润', '收入', '亏损', '入驻', '合作', '品牌'],
    '市场格局': ['市场', '格局', '份额', '地位', '龙头', '霸主', '行业', '变化', '洗牌', '垄断', '独大', '领先', '第一'],
    '产品品类': ['产品', '品类', '商品', '奶茶', '咖啡', '饮品', '餐饮', '美食', '品牌', '电商', '物品', '日用品'],
    '消费习惯': ['消费', '习惯', '购买', '下单', '点餐', '选择', '偏好', '频率', '需求', '经常', '总是', '频繁', '养成'],
}


# ==================== 工具函数 ====================
def load_stopwords():
    """加载停用词"""
    candidate = Path('stopwords.txt')
    if candidate.exists():
        return set(line.strip() for line in candidate.open('r', encoding='utf-8') if line.strip())
    return set([
        '的', '了', '和', '是', '在', '就', '都', '而', '及', '与', '着', '或', '一个', '没有', '我们', '你们',
        '他们', '她们', '是否', '所以', '如果', '因为', '但是', '并且', '然后', '而且', '这个', '那个', '这里',
        '什么', '怎么', '为什么', '可以', '不是', '已经', '还是', '只是', '非常', '很多', '一些', '一点',
        '话题', '分享', '笔记', '薯条', '啊', '哦', '哈哈', '呢', '吧', '吗', '啦', '上', '下', '里', '中',
        '再', '又', '还', '更', '最', '很', '太', '挺', '真', '实', '全', '整', '每', '各', '另', '其',
    ])


def clean_text(text: str) -> str:
    """移除特殊字符，只保留中文、英文和数字"""
    return re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9]+', '', str(text))


def segment(text: str, stopwords: set) -> list:
    """分词并过滤停用词和噪音词"""
    tokens = [w for w in jieba.lcut(text) 
              if w and len(w) >= 2 and w not in stopwords and w not in NOISE_WORDS]
    return tokens


def extract_keywords_tfidf(texts: list, stopwords: set, topn: int = 100):
    """使用TF-IDF提取关键词"""
    if len(texts) == 0:
        return []
    
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: segment(x, stopwords),
        max_features=5000,
        min_df=2,  # 至少出现2次（降低门槛）
        max_df=0.9,  # 最多出现在90%的文档中（提高容忍度）
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
        
        # 按得分排序
        top_indices = mean_scores.argsort()[::-1][:topn]
        keywords_with_scores = [(feature_names[i], mean_scores[i]) for i in top_indices]
        
        return keywords_with_scores
    except Exception as e:
        print(f"TF-IDF提取失败: {e}")
        return []


def extract_keywords_frequency(texts: list, stopwords: set, topn: int = 100):
    """使用词频统计提取关键词"""
    all_words = []
    for text in texts:
        words = segment(text, stopwords)
        all_words.extend(words)
    
    counter = Counter(all_words)
    top_words = counter.most_common(topn)
    
    # 标准化得分（0-1）
    max_count = top_words[0][1] if top_words else 1
    keywords_with_scores = [(word, count / max_count) for word, count in top_words]
    
    return keywords_with_scores


def extract_keywords_textrank(texts: list, topn: int = 100):
    """使用TextRank提取关键词"""
    all_text = ' '.join(texts)
    try:
        keywords = jieba.analyse.textrank(all_text, topK=topn, withWeight=True)
        return list(keywords)
    except Exception as e:
        print(f"TextRank提取失败: {e}")
        return []


def calculate_relevance_to_topic(keyword: str, topic_name: str, related_terms: list) -> float:
    """
    计算关键词与议题的相关度
    
    计算逻辑：
    1. 字符重叠度（关键词与议题名的字符重叠）
    2. 领域词匹配度（关键词是否在议题的领域词列表中）
    3. 字符包含度（关键词是否包含议题名中的字）
    
    返回：0-1之间的相关度得分
    """
    relevance = 0.0
    
    # 1. 字符重叠度（权重0.3）
    shared_chars = len(set(keyword) & set(topic_name))
    if len(keyword) + len(topic_name) > 0:
        char_overlap = shared_chars / (len(keyword) + len(topic_name))
        relevance += char_overlap * 0.3
    
    # 2. 领域词匹配度（权重0.5）
    if keyword in related_terms:
        relevance += 0.5
    else:
        # 部分匹配
        for term in related_terms:
            if term in keyword or keyword in term:
                relevance += 0.3
                break
    
    # 3. 字符包含度（权重0.2）
    topic_chars = set(topic_name)
    keyword_chars = set(keyword)
    if topic_chars & keyword_chars:
        inclusion = len(topic_chars & keyword_chars) / len(topic_chars)
        relevance += inclusion * 0.2
    
    return min(relevance, 1.0)


def assign_keywords_to_topics(keywords_with_scores: list, topics_names: list, 
                               topic_related_terms: dict, max_per_topic: int = 30) -> dict:
    """
    将提取的关键词智能分配到各个议题
    
    分配策略：
    1. 计算每个关键词与各议题的相关度
    2. 将关键词分配给相关度最高的议题（如果相关度>阈值）
    3. 每个议题最多保留max_per_topic个关键词
    """
    print("\n【智能分配关键词到各议题】")
    
    topic_keywords = {topic: [] for topic in topics_names}
    assigned_keywords = set()
    
    for keyword, score in keywords_with_scores:
        if keyword in assigned_keywords:
            continue
        
        # 计算与各议题的相关度
        relevances = {}
        for topic in topics_names:
            related_terms = topic_related_terms.get(topic, [])
            relevance = calculate_relevance_to_topic(keyword, topic, related_terms)
            relevances[topic] = relevance
        
        # 找到相关度最高的议题
        best_topic = max(relevances, key=relevances.get)
        best_relevance = relevances[best_topic]
        
        # 如果相关度超过阈值，分配给该议题（降低阈值以提取更多关键词）
        if best_relevance > 0.1 and len(topic_keywords[best_topic]) < max_per_topic:
            topic_keywords[best_topic].append({
                'keyword': keyword,
                'score': score,
                'relevance': best_relevance
            })
            assigned_keywords.add(keyword)
    
    # 按得分排序
    for topic in topics_names:
        topic_keywords[topic] = sorted(
            topic_keywords[topic], 
            key=lambda x: x['score'] * x['relevance'], 
            reverse=True
        )
    
    return topic_keywords


def _build_topic_keyword_map(topic_keywords: dict, topics_names: list) -> dict:
    """将 {topic: [{keyword, score, relevance}, ...]} 转为 {topic: [keyword, ...]}"""
    topic_kw_map = {}
    for topic in topics_names:
        items = topic_keywords.get(topic, [])
        topic_kw_map[topic] = [str(it.get('keyword', '')).strip() for it in items if str(it.get('keyword', '')).strip()]
    return topic_kw_map


def _count_topic_coverage(df: pd.DataFrame, text_col: str, topic_kw_map: dict) -> dict:
    """统计每个议题下命中文本的数量（至少命中一个该议题关键词即计数）"""
    counts = {topic: 0 for topic in topic_kw_map.keys()}
    for _, row in df.iterrows():
        s = str(row.get(text_col, '')).lower()
        if not s:
            continue
        for topic, kws in topic_kw_map.items():
            hit = False
            for kw in kws:
                k = str(kw).lower()
                if k and k in s:
                    hit = True
                    break
            if hit:
                counts[topic] += 1
    return counts


# ==================== 主处理流程 ====================
def main():
    print("=" * 90)
    print("从数据中提取各议题的关键词")
    print("=" * 90)
    
    # ========== 1. 加载数据 ==========
    print("\n【步骤1】加载数据...")
    # 使用步骤3清洗后的数据作为关键词提取的输入
    posts = pd.read_csv(OUTPUT_DIR / 'posts_filtered.csv', encoding='utf-8-sig')
    comments = pd.read_csv(OUTPUT_DIR / 'comments_filtered.csv', encoding='utf-8-sig')
    
    print(f"✓ 帖子数据：{len(posts)} 条")
    print(f"✓ 评论数据：{len(comments)} 条")
    
    # ========== 2. 文本清洗和分词 ==========
    print("\n【步骤2】文本清洗和分词...")
    stopwords = load_stopwords()
    print(f"✓ 停用词数量：{len(stopwords)}")
    
    # 清洗文本
    posts['text_clean'] = posts['text'].fillna('').apply(clean_text)
    comments['text_clean'] = comments['text'].fillna('').apply(clean_text)
    
    # 合并所有文本
    all_texts = list(posts['text_clean']) + list(comments['text_clean'])
    all_texts = [t for t in all_texts if len(t) > 0]
    
    print(f"✓ 合并后文本数量：{len(all_texts)} 条")
    
    # ========== 3. 提取关键词（多种方法） ==========
    print("\n【步骤3】提取关键词...")
    
    print("\n  方法1: TF-IDF")
    tfidf_keywords = extract_keywords_tfidf(all_texts, stopwords, topn=250)
    print(f"  ✓ 提取关键词：{len(tfidf_keywords)} 个")
    if tfidf_keywords:
        print(f"  示例：{', '.join([kw for kw, _ in tfidf_keywords[:10]])}")
    
    print("\n  方法2: 词频统计")
    freq_keywords = extract_keywords_frequency(all_texts, stopwords, topn=250)
    print(f"  ✓ 提取关键词：{len(freq_keywords)} 个")
    if freq_keywords:
        print(f"  示例：{', '.join([kw for kw, _ in freq_keywords[:10]])}")
    
    print("\n  方法3: TextRank")
    textrank_keywords = extract_keywords_textrank(all_texts, topn=250)
    print(f"  ✓ 提取关键词：{len(textrank_keywords)} 个")
    if textrank_keywords:
        print(f"  示例：{', '.join([kw for kw, _ in textrank_keywords[:10]])}")
    
    # ========== 4. 合并多种方法的结果 ==========
    print("\n【步骤4】合并多种方法的关键词...")
    
    # 将关键词及其得分收集到字典中
    keyword_scores = {}
    
    for kw, score in tfidf_keywords:
        keyword_scores[kw] = keyword_scores.get(kw, 0) + score * 0.4
    
    for kw, score in freq_keywords:
        keyword_scores[kw] = keyword_scores.get(kw, 0) + score * 0.4
    
    for kw, score in textrank_keywords:
        keyword_scores[kw] = keyword_scores.get(kw, 0) + score * 0.2
    
    # 按得分排序
    combined_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"✓ 合并后关键词：{len(combined_keywords)} 个")
    print(f"  Top20: {', '.join([kw for kw, _ in combined_keywords[:20]])}")
    
    # ========== 5. 将关键词分配到各议题 ==========
    print("\n【步骤5】将关键词分配到各议题...")
    topic_keywords = assign_keywords_to_topics(
        combined_keywords, 
        TOPICS_NAMES, 
        TOPIC_RELATED_TERMS,
        max_per_topic=50  # 增加每个议题的关键词上限，提高覆盖度
    )
    
    # 打印每个议题的关键词数量
    print("\n✓ 各议题关键词数量：")
    for topic in TOPICS_NAMES:
        count = len(topic_keywords[topic])
        print(f"  • {topic}: {count} 个关键词")

    # ========== 5.1 统计各议题对应的帖子/评论数量 ==========
    print("\n【各议题的帖子/评论数量】")
    topic_kw_map = _build_topic_keyword_map(topic_keywords, TOPICS_NAMES)
    post_counts = _count_topic_coverage(posts, 'text', topic_kw_map)
    comment_counts = _count_topic_coverage(comments, 'text', topic_kw_map)
    for topic in TOPICS_NAMES:
        print(f"  • {topic}: 帖子 {post_counts.get(topic, 0)} 条，评论 {comment_counts.get(topic, 0)} 条")
    
    # ========== 6. 保存结果 ==========
    print("\n【步骤6】保存结果...")
    
    # 格式1: CSV格式（详细信息）
    rows = []
    for topic in TOPICS_NAMES:
        keywords = topic_keywords[topic]
        for item in keywords:
            rows.append({
                '议题': topic,
                '关键词': item['keyword'],
                'TF-IDF得分': f"{item['score']:.4f}",
                '相关度': f"{item['relevance']:.4f}",
                '综合得分': f"{item['score'] * item['relevance']:.4f}"
            })
    
    df_detailed = pd.DataFrame(rows)
    detailed_file = OUTPUT_DIR / 'topic_keywords_detailed.csv'
    df_detailed.to_csv(detailed_file, index=False, encoding='utf-8-sig')
    print(f"✓ 详细信息已保存：{detailed_file}")
    
    # 格式2: Python代码格式（直接可用）
    code_lines = ["# 各议题的关键词（从数据中自动提取）\n", "TOPICS = {\n"]
    for topic in TOPICS_NAMES:
        keywords = [item['keyword'] for item in topic_keywords[topic]]
        keywords_str = "', '".join(keywords)
        code_lines.append(f"    '{topic}': ['{keywords_str}'],\n")
    code_lines.append("}\n")
    
    code_file = OUTPUT_DIR / 'topic_keywords_generated.py'
    with open(code_file, 'w', encoding='utf-8') as f:
        f.writelines(code_lines)
    print(f"✓ Python代码已保存：{code_file}")
    
    # 格式3: 简洁列表格式（每个议题一行）
    simple_rows = []
    for topic in TOPICS_NAMES:
        keywords = [item['keyword'] for item in topic_keywords[topic]]
        simple_rows.append({
            '议题': topic,
            '关键词数量': len(keywords),
            '关键词列表': ', '.join(keywords)
        })
    
    df_simple = pd.DataFrame(simple_rows)
    simple_file = OUTPUT_DIR / 'topic_keywords_list.csv'
    df_simple.to_csv(simple_file, index=False, encoding='utf-8-sig')
    print(f"✓ 简洁列表已保存：{simple_file}")

    # 额外：保存各议题帖子/评论数量统计
    counts_rows = []
    for topic in TOPICS_NAMES:
        counts_rows.append({
            '议题': topic,
            '帖子数量': post_counts.get(topic, 0),
            '评论数量': comment_counts.get(topic, 0)
        })
    df_counts = pd.DataFrame(counts_rows)
    counts_file = OUTPUT_DIR / 'topic_post_comment_counts.csv'
    df_counts.to_csv(counts_file, index=False, encoding='utf-8-sig')
    print(f"✓ 议题帖子/评论数量统计已保存：{counts_file}")
    
    # ========== 7. 打印结果预览 ==========
    print("\n【步骤7】结果预览...")
    print("\n" + "=" * 90)
    for topic in TOPICS_NAMES:
        keywords = [item['keyword'] for item in topic_keywords[topic]]
        print(f"\n【{topic}】({len(keywords)}个关键词)")
        print(f"  {', '.join(keywords[:15])}{'...' if len(keywords) > 15 else ''}")
    
    # ========== 8. 生成统计报告 ==========
    print("\n" + "=" * 90)
    print("✓ 处理完成！")
    print("=" * 90)
    
    total_keywords = sum(len(topic_keywords[topic]) for topic in TOPICS_NAMES)
    print(f"\n【统计摘要】")
    print(f"  • 议题数量：{len(TOPICS_NAMES)}")
    print(f"  • 总关键词数：{total_keywords}")
    print(f"  • 平均每个议题：{total_keywords / len(TOPICS_NAMES):.1f} 个关键词")
    
    print(f"\n【输出文件】")
    print(f"  1. {detailed_file}")
    print(f"     └─ 包含详细得分信息的CSV文件")
    print(f"  2. {code_file}")
    print(f"     └─ Python代码格式，可直接复制使用")
    print(f"  3. {simple_file}")
    print(f"     └─ 简洁列表格式，便于查看")
    print(f"  4. {counts_file}")
    print(f"     └─ 各议题帖子/评论数量统计")
    
    print("\n【后续步骤】")
    print("  1. 查看 topic_keywords_list.csv 了解各议题的关键词")
    print("  2. 将 topic_keywords_generated.py 中的TOPICS定义复制到分析脚本中使用")
    print("  3. 根据需要手动调整关键词（添加或删除）")
    print("=" * 90)
    
    return topic_keywords


if __name__ == '__main__':
    topic_keywords = main()
