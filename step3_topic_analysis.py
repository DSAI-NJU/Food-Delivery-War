"""
小红书文本分析 - 第三步：核心议题分析
根据《小红书文本分析详细操作指导.md》第3部分实现
基于多种方法提取关键词，并将文本分类到预定义议题
"""

import re
from collections import Counter, defaultdict
from pathlib import Path

import jieba
import jieba.analyse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 尝试导入词云
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


# ==================== 配置与工具函数 ====================
OUTPUT_DIR = Path('data/data/xhs/csv/处理')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 预定义议题及其关键词
TOPICS = {
    '价格补贴': ['补贴', '优惠', '便宜', '红包', '满减', '折扣', '价格', '省钱', '实惠', '便宜了'],
    '平台竞争': ['美团', '京东', '饿了么', '淘宝', '闪购', '竞争', '打仗', '大战', '对抗', '挑战'],
    '用户体验': ['体验', '方便', '快捷', '好用', '界面', '操作', '服务', '满意', '推荐', '喜欢'],
    '配送服务': ['配送', '外卖', '送餐', '骑手', '快递', '时间', '速度', '准时', '延迟', '送达'],
    '监管政策': ['监管', '政策', '法律', '约谈', '规定', '合规', '整改', '处罚', '反垄断', '竞争法'],
    '商家权益': ['商家', '店铺', '商户', '餐饮', '抽成', '佣金', '费用', '成本', '利润', '入驻'],
    '市场格局': ['市场', '份额', '占有率', '地位', '霸主', '龙头', '格局', '变化', '洗牌', '行业'],
    '消费习惯': ['习惯', '消费', '购买', '下单', '点餐', '选择', '偏好', '频率', '需求', '用户']
}

PALETTE = {
    '正向': '#8EC5FF',  # soft blue
    '中性': '#FFD88E',  # soft yellow
    '负向': '#99A8B2',  # soft gray-blue
}


def load_stopwords():
    """加载停用词"""
    candidate = Path('stopwords.txt')
    if candidate.exists():
        return set(line.strip() for line in candidate.open('r', encoding='utf-8') if line.strip())
    return set([
        '的', '了', '和', '是', '在', '就', '都', '而', '及', '与', '着', '或', '一个', '没有', '我们', '你们',
        '他们', '她们', '是否', '所以', '如果', '因为', '但是', '并且', '然后', '而且', '这个', '那个', '这里',
        '什么', '怎么', '为什么', '可以', '不是', '已经', '还是', '只是', '非常', '很多', '一些', '一点',
        '话题', '分享', '笔记', '薯条', '啊', '哦', '哈哈', '呢', '吧', '吗', '啦'
    ])


def clean_text(text: str) -> str:
    """移除特殊字符"""
    return re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9]+', '', str(text))


def segment(text: str, stopwords: set) -> list:
    """分词并过滤"""
    tokens = [w for w in jieba.lcut(text) if w and len(w) > 1 and w not in stopwords]
    return tokens


def extract_keywords_multi_method(texts: list, stopwords: set, topn: int = 50):
    """
    使用多种方法提取关键词：
    1. TF-IDF
    2. TextRank (jieba)
    3. 词频统计
    """
    print("\n【方法1】TF-IDF 关键词提取...")
    tfidf_keywords = extract_tfidf_keywords(texts, stopwords, topn)
    
    print("【方法2】TextRank 关键词提取...")
    textrank_keywords = extract_textrank_keywords(texts, topn)
    
    print("【方法3】词频统计关键词提取...")
    freq_keywords = extract_freq_keywords(texts, stopwords, topn)
    
    # 合并多种方法的结果，取并集
    all_keywords = set(tfidf_keywords) | set(textrank_keywords) | set(freq_keywords)
    
    return {
        'tfidf': tfidf_keywords,
        'textrank': textrank_keywords,
        'freq': freq_keywords,
        'combined': list(all_keywords)
    }


def extract_tfidf_keywords(texts: list, stopwords: set, topn: int = 50):
    """TF-IDF 提取关键词"""
    if len(texts) == 0:
        return []
    vectorizer = TfidfVectorizer(
        tokenizer=jieba.lcut,
        stop_words=list(stopwords),
        max_features=5000,
        min_df=2,
    )
    tfidf = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names_out()
    mean_scores = np.asarray(tfidf.mean(axis=0)).ravel()
    top_idx = mean_scores.argsort()[::-1][:topn]
    return words[top_idx].tolist()


def extract_textrank_keywords(texts: list, topn: int = 50):
    """TextRank 提取关键词"""
    all_text = ' '.join(texts)
    keywords = jieba.analyse.textrank(all_text, topK=topn, withWeight=False)
    return keywords


def extract_freq_keywords(texts: list, stopwords: set, topn: int = 50):
    """词频统计提取关键词"""
    all_words = []
    for text in texts:
        words = segment(text, stopwords)
        all_words.extend(words)
    counter = Counter(all_words)
    return [word for word, _ in counter.most_common(topn)]


def classify_to_topics(text: str, topics_dict: dict) -> list:
    """
    根据关键词将文本分类到议题
    返回匹配的议题列表（可能匹配多个）
    """
    matched_topics = []
    for topic, keywords in topics_dict.items():
        for keyword in keywords:
            if keyword in text:
                matched_topics.append(topic)
                break
    return matched_topics if matched_topics else ['其他']


def attach_sentiment(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """为帖子/评论附加情感标签，优先使用 step4 的情感打分输出。"""
    sentiment_file = Path('处理了') / ('帖子情感打分.csv' if kind == 'post' else '评论情感打分.csv')
    if not sentiment_file.exists():
        print(f"[WARN] 未找到情感结果文件：{sentiment_file}，将尝试使用现有列 sentiment")
        if 'sentiment' in df.columns:
            return df
        return df.assign(sentiment=np.nan)

    sent_df = pd.read_csv(sentiment_file, encoding='utf-8')
    key_cols = []
    if kind == 'post' and 'note_id' in df.columns and 'note_id' in sent_df.columns:
        key_cols = ['note_id']
    elif kind == 'comment' and 'comment_id' in df.columns and 'comment_id' in sent_df.columns:
        key_cols = ['comment_id']
    elif 'text' in df.columns and 'text' in sent_df.columns:
        key_cols = ['text']

    if key_cols:
        merged = df.merge(sent_df[key_cols + ['sentiment']], on=key_cols, how='left')
    else:
        # 回退：按行顺序合并，风险是对齐不完美
        merged = df.copy()
        merged['sentiment'] = sent_df['sentiment'].reindex(range(len(df))).values

    return merged


def plot_topic_sentiment_stack(df: pd.DataFrame, title: str, outfile: Path) -> None:
    if df.empty:
        print(f"[WARN] 无数据绘制：{title}")
        return
    # df: columns [topic, sentiment, count, ratio]
    pivot = df.pivot_table(index='topic', columns='sentiment', values='ratio', fill_value=0.0)
    pivot = pivot[[c for c in ['负向', '中性', '正向'] if c in pivot.columns]]
    ax = pivot.plot(kind='bar', stacked=True, figsize=(12, 7), color=[PALETTE.get(c) for c in pivot.columns])
    ax.set_title(title)
    ax.set_xlabel('议题')
    ax.set_ylabel('占比')
    plt.xticks(rotation=30, ha='right')
    ax.legend(title='情感')
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] 议题情感堆叠图已保存：{outfile}")


def build_wordcloud_improved(counter: Counter, outfile: Path, title: str, max_words: int = 100):
    """生成改进版词云"""
    if not WORDCLOUD_AVAILABLE:
        print(f"[WordCloud] 模块不可用，跳过生成：{title}")
        return
    if not counter or len(counter) == 0:
        print(f"[WordCloud] 无足够词频，跳过生成：{title}")
        return
    
    # 尝试多个中文字体路径
    font_paths = [
        'C:/Windows/Fonts/simhei.ttf',  # 黑体
        'C:/Windows/Fonts/msyh.ttc',     # 微软雅黑
        'C:/Windows/Fonts/msyhbd.ttc',   # 微软雅黑粗体
        'C:/Windows/Fonts/simsun.ttc',   # 宋体
    ]
    
    font_path = None
    for path in font_paths:
        if Path(path).exists():
            font_path = path
            break
    
    if font_path is None:
        print(f"[WARNING] 未找到中文字体，词云可能无法正确显示中文")
    
    # 创建词云
    wc = WordCloud(
        font_path=font_path,
        width=1200,
        height=800,
        background_color='white',
        max_words=max_words,
        collocations=False,
        relative_scaling=0.5,
        colormap='viridis'
    )
    wc.generate_from_frequencies(dict(counter))
    
    # 保存并显示
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] 词云已保存：{outfile}")


# ==================== 主流程 ====================
if __name__ == '__main__':
    print("=" * 70)
    print("第三步：核心议题分析")
    print("=" * 70)

    stopwords = load_stopwords()
    print(f"停用词数量：{len(stopwords)}")

    # 加载预处理数据
    posts = pd.read_csv(OUTPUT_DIR / 'posts_processed.csv', encoding='utf-8')
    comments = pd.read_csv(OUTPUT_DIR / 'comments_processed.csv', encoding='utf-8')
    print(f"\n帖子数据：{len(posts)} 条，评论数据：{len(comments)} 条")

    # 文本清洗
    posts['text'] = posts['text'].fillna('').apply(clean_text)
    comments['text'] = comments['text'].fillna('').apply(clean_text)

    # 分词
    print("\n正在进行分词处理...")
    posts['words'] = posts['text'].apply(lambda x: segment(x, stopwords))
    comments['words'] = comments['text'].apply(lambda x: segment(x, stopwords))

    # ========== 1. 使用多种方法提取关键词 ==========
    print("\n" + "=" * 70)
    print("【步骤1】使用多种方法提取关键词")
    print("=" * 70)
    
    # 提取帖子关键词
    print("\n>>> 帖子关键词提取：")
    post_keywords = extract_keywords_multi_method(posts['text'].tolist(), stopwords, topn=50)
    print(f"TF-IDF 方法提取关键词数：{len(post_keywords['tfidf'])}")
    print(f"TextRank 方法提取关键词数：{len(post_keywords['textrank'])}")
    print(f"词频统计方法提取关键词数：{len(post_keywords['freq'])}")
    print(f"合并后关键词总数：{len(post_keywords['combined'])}")
    
    # 提取评论关键词
    print("\n>>> 评论关键词提取：")
    comment_keywords = extract_keywords_multi_method(comments['text'].tolist(), stopwords, topn=50)
    print(f"TF-IDF 方法提取关键词数：{len(comment_keywords['tfidf'])}")
    print(f"TextRank 方法提取关键词数：{len(comment_keywords['textrank'])}")
    print(f"词频统计方法提取关键词数：{len(comment_keywords['freq'])}")
    print(f"合并后关键词总数：{len(comment_keywords['combined'])}")

    # 保存关键词对比结果
    keywords_comparison = pd.DataFrame({
        '方法': ['TF-IDF', 'TextRank', '词频统计', '合并结果'],
        '帖子关键词数': [
            len(post_keywords['tfidf']),
            len(post_keywords['textrank']),
            len(post_keywords['freq']),
            len(post_keywords['combined'])
        ],
        '评论关键词数': [
            len(comment_keywords['tfidf']),
            len(comment_keywords['textrank']),
            len(comment_keywords['freq']),
            len(comment_keywords['combined'])
        ]
    })
    keywords_comparison.to_csv(OUTPUT_DIR / '关键词提取方法对比.csv', index=False, encoding='utf-8-sig')
    print(f"\n[OK] 关键词提取方法对比已保存")

    # ========== 2. 议题分类 ==========
    print("\n" + "=" * 70)
    print("【步骤2】基于关键词的议题分类")
    print("=" * 70)
    print(f"\n预定义议题数量：{len(TOPICS)}")
    for topic, keywords in TOPICS.items():
        print(f"  • {topic}: {len(keywords)} 个关键词")

    # 对帖子进行议题分类
    print("\n>>> 帖子议题分类中...")
    posts['topics'] = posts['text'].apply(lambda x: classify_to_topics(x, TOPICS))
    posts['topic_count'] = posts['topics'].apply(len)
    
    # 对评论进行议题分类
    print(">>> 评论议题分类中...")
    comments['topics'] = comments['text'].apply(lambda x: classify_to_topics(x, TOPICS))
    comments['topic_count'] = comments['topics'].apply(len)

    # 统计议题分布
    post_topic_dist = Counter([t for topics in posts['topics'] for t in topics])
    comment_topic_dist = Counter([t for topics in comments['topics'] for t in topics])

    topic_dist_df = pd.DataFrame({
        '议题': list(set(post_topic_dist.keys()) | set(comment_topic_dist.keys())),
    })
    topic_dist_df['帖子数量'] = topic_dist_df['议题'].apply(lambda x: post_topic_dist.get(x, 0))
    topic_dist_df['评论数量'] = topic_dist_df['议题'].apply(lambda x: comment_topic_dist.get(x, 0))
    topic_dist_df = topic_dist_df.sort_values('评论数量', ascending=False)
    
    topic_dist_path = OUTPUT_DIR / '议题分布统计.csv'
    topic_dist_df.to_csv(topic_dist_path, index=False, encoding='utf-8-sig')
    print(f"\n[OK] 议题分布统计已保存：{topic_dist_path}")
    print("\n议题分布概况：")
    print(topic_dist_df.to_string(index=False))

    # 保存分类后的数据
    posts[['text', 'topics', 'topic_count']].to_csv(
        OUTPUT_DIR / '帖子议题分类结果.csv', index=False, encoding='utf-8-sig'
    )
    comments[['text', 'topics', 'topic_count']].to_csv(
        OUTPUT_DIR / '评论议题分类结果.csv', index=False, encoding='utf-8-sig'
    )
    print("[OK] 议题分类结果已保存")

    # ========== 2.1 议题 × 情感占比堆叠图（帖子/评论/合并） ==========
    print("\n>>> 议题 × 情感占比堆叠图 ...")

    def build_topic_sentiment(df_topics: pd.DataFrame, kind: str, label: str):
        df_sent = attach_sentiment(df_topics, kind)
        if 'sentiment' not in df_sent.columns:
            print(f"[WARN] {label} 缺少 sentiment 列，跳过")
            return None
        exploded = df_sent.explode('topics')
        exploded = exploded.dropna(subset=['topics', 'sentiment'])
        if exploded.empty:
            print(f"[WARN] {label} 无可用数据，跳过")
            return None
        counts = exploded.groupby(['topics', 'sentiment']).size().rename('count').reset_index()
        topic_totals = counts.groupby('topics')['count'].sum().rename('total').reset_index()
        merged = counts.merge(topic_totals, on='topics')
        merged['ratio'] = merged['count'] / merged['total']
        merged = merged.rename(columns={'topics': 'topic'})
        return merged

    post_ts = build_topic_sentiment(posts[['text', 'topics']].copy(), 'post', '帖子')
    comment_ts = build_topic_sentiment(comments[['text', 'topics']].copy(), 'comment', '评论')

    if post_ts is not None:
        plot_topic_sentiment_stack(post_ts, '帖子：各议题情感占比', OUTPUT_DIR / '议题情感堆叠_帖子.png')
    if comment_ts is not None:
        plot_topic_sentiment_stack(comment_ts, '评论：各议题情感占比', OUTPUT_DIR / '议题情感堆叠_评论.png')
    if post_ts is not None and comment_ts is not None:
        combined_ts = pd.concat([post_ts, comment_ts], ignore_index=True)
        combined_ts = combined_ts.groupby(['topic', 'sentiment'])['count'].sum().reset_index()
        totals = combined_ts.groupby('topic')['count'].sum().rename('total').reset_index()
        combined_ts = combined_ts.merge(totals, on='topic')
        combined_ts['ratio'] = combined_ts['count'] / combined_ts['total']
        plot_topic_sentiment_stack(combined_ts, '合并：各议题情感占比', OUTPUT_DIR / '议题情感堆叠_合并.png')

    # ========== 3. 生成词云图 ==========
    print("\n" + "=" * 70)
    print("【步骤3】生成词云图")
    print("=" * 70)
    
    # 整体词云
    print("\n>>> 生成整体词云...")
    post_counter = Counter([w for ws in posts['words'] for w in ws])
    comment_counter = Counter([w for ws in comments['words'] for w in ws])
    
    build_wordcloud_improved(post_counter, OUTPUT_DIR / '词云_帖子_整体.png', '帖子整体词云', max_words=100)
    build_wordcloud_improved(comment_counter, OUTPUT_DIR / '词云_评论_整体.png', '评论整体词云', max_words=100)
    
    # 为每个主要议题生成词云
    print("\n>>> 为各议题生成词云...")
    topic_wordclouds_dir = OUTPUT_DIR / '议题词云'
    topic_wordclouds_dir.mkdir(exist_ok=True)
    
    for topic in TOPICS.keys():
        # 筛选该议题的评论
        topic_comments = comments[comments['topics'].apply(lambda x: topic in x)]
        if len(topic_comments) == 0:
            print(f"  • {topic}: 无数据，跳过")
            continue
        
        # 统计该议题的词频
        topic_words = [w for ws in topic_comments['words'] for w in ws]
        topic_counter = Counter(topic_words)
        
        if len(topic_counter) < 10:
            print(f"  • {topic}: 词汇量不足，跳过")
            continue
        
        # 生成词云
        output_path = topic_wordclouds_dir / f'词云_{topic}.png'
        build_wordcloud_improved(topic_counter, output_path, f'议题词云：{topic}', max_words=80)

    print("\n" + "=" * 70)
    print("核心议题分析完成！")
    print("=" * 70)
    print(f"\n输出文件位于：{OUTPUT_DIR}")
    print("\n生成的文件包括：")
    print("  1. 关键词提取方法对比.csv - 多种方法的关键词提取结果对比")
    print("  2. 议题分布统计.csv - 各议题的帖子和评论分布")
    print("  3. 帖子议题分类结果.csv - 帖子的议题分类详情")
    print("  4. 评论议题分类结果.csv - 评论的议题分类详情")
    print("  5. 词云_帖子_整体.png - 帖子整体词云图")
    print("  6. 词云_评论_整体.png - 评论整体词云图")
    print(f"  7. 议题词云/ - 各议题的词云图（{len(TOPICS)}个议题）")
    print("=" * 70)
