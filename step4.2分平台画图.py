
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams, font_manager

from snownlp import SnowNLP
import jieba
from collections import Counter

# =============== 可配置：输入与输出路径 ===============
# 用户指定的两个输入 CSV 绝对路径
COMMENTS_CSV = "C:/Users/Lenovo/Desktop/经济学中的人工智能与数据科学/小组作业/final_comments.csv"
POSTS_CSV = "C:/Users/Lenovo/Desktop/经济学中的人工智能与数据科学/小组作业/final_contents.csv"

# 输出目录（与仓库现有结构保持一致）
OUTPUT_DIR = Path("处理了")
FIG_DIR = Path("分平台图")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CH_FONT_PROP = None


def _set_chinese_font():
    # 优先用具体字体文件，确保不出现方框；如没有则抛出异常提醒用户放置字体
    global CH_FONT_PROP
    font_files = [
        'C:/Windows/Fonts/msyh.ttc',       # 微软雅黑
        'C:/Windows/Fonts/msyhbd.ttc',     # 微软雅黑粗体
        'C:/Windows/Fonts/simhei.ttf',     # 黑体
        'C:/Windows/Fonts/simsun.ttc',     # 宋体
    ]
    for ff in font_files:
        if Path(ff).exists():
            try:
                font_manager.fontManager.addfont
                fprop = font_manager.FontProperties(fname=ff)
                fname = fprop.get_name()
                rcParams['font.family'] = [fname]
                rcParams['font.sans-serif'] = [fname]
                rcParams['axes.unicode_minus'] = False
                CH_FONT_PROP = fprop
                print(f"[OK] 使用中文字体：{fname} ({ff})")
                return
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] 字体加载失败 {ff}: {e}")
    raise RuntimeError("未找到中文字体。请将 msyh.ttc 或 simhei.ttf 放到 C:/Windows/Fonts 后重跑 step4.py")


_set_chinese_font()
sns.set_style("whitegrid")
PALETTE = {
    "正向": "#8EC5FF",  # soft blue
    "中性": "#FFD88E",  # soft yellow-orange
    "负向": "#99A8B2"   # soft gray-blue
}

# 议题词典（借鉴 step3 预定义议题）
TOPICS = {
    '价格补贴': ['补贴', '优惠', '便宜', '红包', '满减', '折扣', '价格', '省钱', '实惠', '便宜了', '活动价', '补贴券', '返现', '立减', '补差价', '折扣券'],
    '平台竞争': ['美团', '京东', '饿了么', '淘宝', '闪购', '竞争', '打仗', '大战', '对抗', '挑战', '对标', '竞品', 'pk', 'PK', '双平台', '跨平台'],
    '用户体验': ['体验', '方便', '快捷', '好用', '界面', '操作', '服务', '满意', '推荐', '喜欢', '卡顿', '崩溃', '闪退', 'bug', 'BUG', '客服', '售后', '投诉', '反馈', '界面设计'],
    '配送服务': ['配送', '外卖', '送餐', '骑手', '快递', '时间', '速度', '准时', '延迟', '送达', '配送费', '外卖费', '运费', '起送', '起送价', '送费', '运力', '达达', '蜂鸟', '闪送', '小费', '调度'],
    '监管政策': ['监管', '政策', '法律', '约谈', '规定', '合规', '整改', '处罚', '反垄断', '竞争法', '罚款', '监管部门', '通报'],
    '商家权益': ['商家', '店铺', '商户', '餐饮', '抽成', '佣金', '费用', '成本', '利润', '入驻', '抽佣', '扣点', '返点', '结算', '账期', '流量倾斜'],
    '市场格局': ['市场', '份额', '占有率', '地位', '霸主', '龙头', '格局', '变化', '洗牌', '行业', '行业格局', '市场份额', '市占', '双寡头'],
    '消费习惯': ['习惯', '消费', '购买', '下单', '点餐', '选择', '偏好', '频率', '需求', '用户', '复购', '囤货', '下单频次', '口味', '口碑']
}

TOPIC_LIST = list(TOPICS.keys()) + ["其他"]

PLATFORM_COLORS = {
    "美团": "#F6D878",   # 淡黄色
    "饿了么": "#8EC5FF",  # 浅蓝色
    "淘宝": "#C7CBD1",   # 浅灰色
}


def read_csv_auto(path: str | Path, encodings: Tuple[str, ...] = ("utf-8", "gb18030", "gbk", "gb2312")) -> pd.DataFrame:
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"读取 CSV 失败：{path}\n最后错误：{last_err}")


def ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def snownlp_score(text: str) -> float:
    try:
        s = SnowNLP(str(text))
        return float(s.sentiments)
    except Exception:
        return np.nan


def get_sentiment_label(score: float, pos_th: float = 0.6, neg_th: float = 0.4) -> str:
    if pd.isna(score):
        return "中性"
    if score >= pos_th:
        return "正向"
    if score <= neg_th:
        return "负向"
    return "中性"


def tokenize(text: str) -> list[str]:
    return [w for w in jieba.lcut(str(text)) if w and len(w) > 1]


def classify_topic(text: str) -> str:
    for topic, kws in TOPICS.items():
        if any(kw in str(text) for kw in kws):
            return topic
    return "其他"


def to_datetime_if_exists(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = ensure_datetime(df[col])
    return df


def filter_platforms(df: pd.DataFrame) -> dict:
    platforms = {
        "美团": ["美团"],
        "饿了么": ["饿了么"],
        "淘宝": ["淘宝"],
    }
    result = {}
    if "text" not in df.columns:
        return result
    for plat, kws in platforms.items():
        mask = df["text"].apply(lambda t: any(kw in str(t) for kw in kws))
        sub = df[mask].copy()
        if not sub.empty:
            result[plat] = sub
    return result


def sentiment_stack_data(platform_dfs: dict) -> pd.DataFrame:
    rows = []
    for plat, pdf in platform_dfs.items():
        cnt = pdf.groupby("sentiment").size().rename("count").reset_index()
        total = cnt["count"].sum()
        cnt["platform"] = plat
        cnt["ratio"] = cnt["count"] / total
        rows.append(cnt)
    if not rows:
        return pd.DataFrame(columns=["platform", "sentiment", "ratio", "count"])
    return pd.concat(rows, ignore_index=True)


def plot_platform_sentiment_stack(df: pd.DataFrame, outfile: Path) -> None:
    if df.empty:
        print("[WARN] 平台情感数据为空，跳过绘图")
        return
    pivot = df.pivot_table(index="platform", columns="sentiment", values="ratio", fill_value=0.0)
    cols = [c for c in ["负向", "中性", "正向"] if c in pivot.columns]
    pivot = pivot[cols]
    ax = pivot.plot(kind="bar", stacked=True, figsize=(9, 6), color=[PALETTE.get(c) for c in cols])
    ax.set_title("三平台评论情感占比", fontproperties=CH_FONT_PROP)
    ax.set_xlabel("平台", fontproperties=CH_FONT_PROP)
    ax.set_ylabel("占比", fontproperties=CH_FONT_PROP)
    ax.legend(title="情感", prop=CH_FONT_PROP, title_fontproperties=CH_FONT_PROP)
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=CH_FONT_PROP)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=CH_FONT_PROP)
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] 已保存：{outfile}")


def topic_ratio_platform(platform_dfs: dict) -> pd.DataFrame:
    rows = []
    for plat, pdf in platform_dfs.items():
        if "topic" not in pdf.columns:
            continue
        cnt = pdf.groupby("topic").size().rename("count").reset_index()
        total = cnt["count"].sum()
        cnt["platform"] = plat
        cnt["ratio"] = cnt["count"] / total if total > 0 else 0
        # 补齐缺失的议题，确保横轴完整
        existing_topics = set(cnt["topic"].tolist())
        missing = [t for t in TOPIC_LIST if t not in existing_topics]
        if missing:
            pad = pd.DataFrame({
                "topic": missing,
                "count": [0] * len(missing),
                "platform": plat,
                "ratio": [0] * len(missing),
            })
            cnt = pd.concat([cnt, pad], ignore_index=True)
        rows.append(cnt)
    if not rows:
        return pd.DataFrame(columns=["platform", "topic", "ratio", "count"])
    return pd.concat(rows, ignore_index=True)


def plot_topic_platform(df: pd.DataFrame, outfile: Path) -> None:
    if df.empty:
        print("[WARN] 平台议题数据为空，跳过绘图")
        return
    # 展示全部议题，按预定义词典顺序（附带“其他”）
    topic_order = [t for t in TOPIC_LIST if t in df["topic"].unique()]
    pivot = df.pivot_table(index="topic", columns="platform", values="ratio", fill_value=0.0)
    # 保证列顺序
    cols = [p for p in ["美团", "饿了么", "淘宝"] if p in pivot.columns]
    pivot = pivot.reindex(topic_order)
    ax = pivot[cols].plot(kind="bar", figsize=(12, 6), color=[PLATFORM_COLORS[c] for c in cols])
    ax.set_title("各议题在三平台的讨论占比", fontproperties=CH_FONT_PROP)
    ax.set_xlabel("议题", fontproperties=CH_FONT_PROP)
    ax.set_ylabel("占比", fontproperties=CH_FONT_PROP)
    plt.xticks(rotation=25, ha="right")
    ax.legend(title="平台", prop=CH_FONT_PROP, title_fontproperties=CH_FONT_PROP)
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=CH_FONT_PROP)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=CH_FONT_PROP)
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] 已保存：{outfile}")


def wordfreq_platform(platform_dfs: dict, topn: int = 15) -> tuple[dict, pd.DataFrame]:
    counters = {}
    rows = []
    for plat, pdf in platform_dfs.items():
        tokens = pdf["text"].apply(tokenize)
        counter = Counter([w for ws in tokens for w in ws])
        counters[plat] = counter
        for w, f in counter.most_common(topn):
            rows.append({"platform": plat, "word": w, "freq": f})
    return counters, pd.DataFrame(rows)


def plot_platform_rolling_mean(platform_dfs: dict, outfile: Path, time_col: str = "create_time", score_col: str = "sentiment_score", window: int = 30) -> None:
    plt.figure(figsize=(12, 6))
    has_data = False
    for plat in ["美团", "饿了么", "淘宝"]:
        if plat not in platform_dfs:
            continue
        pdf = platform_dfs[plat]
        if time_col not in pdf.columns or score_col not in pdf.columns:
            continue
        tmp = pdf.dropna(subset=[time_col, score_col]).copy()
        if tmp.empty:
            continue
        tmp[time_col] = pd.to_datetime(tmp[time_col])
        daily = tmp.set_index(time_col)[score_col].resample('D').mean()
        if daily.empty:
            continue
        rolling = daily.rolling(window, min_periods=1).mean()
        plt.plot(rolling.index, rolling.values, label=plat, color=PLATFORM_COLORS.get(plat, "#888"), linewidth=2)
        has_data = True
    if not has_data:
        print("[WARN] 分平台滚动均值无数据，跳过绘图")
        plt.close()
        return
    plt.title(f"三平台情感得分{window}天滚动均值", fontproperties=CH_FONT_PROP)
    plt.xlabel("日期", fontproperties=CH_FONT_PROP)
    plt.ylabel("情感得分", fontproperties=CH_FONT_PROP)
    plt.xticks(fontproperties=CH_FONT_PROP, rotation=30, ha="right")
    plt.yticks(fontproperties=CH_FONT_PROP)
    plt.legend(prop=CH_FONT_PROP)
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] 已保存：{outfile}")


def plot_wordfreq_platform(counters: dict, outfile: Path, topn: int = 15) -> None:
    if not counters:
        print("[WARN] 词频为空，跳过绘图")
        return
    plats = [p for p in ["美团", "饿了么", "淘宝"] if p in counters]
    fig, axes = plt.subplots(1, len(plats), figsize=(5 * len(plats), 6), sharey=False)
    if len(plats) == 1:
        axes = [axes]
    for ax, plat in zip(axes, plats):
        counter = counters[plat]
        items = counter.most_common(topn)
        if not items:
            ax.axis('off')
            continue
        words, freqs = zip(*items)
        ax.barh(range(len(words)), freqs, color=PLATFORM_COLORS.get(plat, "#888"))
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontproperties=CH_FONT_PROP)
        ax.invert_yaxis()
        ax.set_title(f"{plat} 词频Top{topn}", fontproperties=CH_FONT_PROP)
        ax.set_xlabel("频次", fontproperties=CH_FONT_PROP)
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] 已保存：{outfile}")


def main(pos_th: float = 0.6, neg_th: float = 0.4, topn: int = 15) -> None:
    print("=" * 70)
    print("分平台情感与议题分析")
    print("=" * 70)

    comments = read_csv_auto(COMMENTS_CSV)
    if "content" in comments.columns:
        comments["text"] = comments["content"].fillna("").astype(str)
    else:
        comments["text"] = comments.astype(str).agg(" ".join, axis=1)

    comments = to_datetime_if_exists(comments, "create_time")

    comments["sentiment_score"] = comments["text"].apply(snownlp_score)
    comments["sentiment"] = comments["sentiment_score"].apply(lambda s: get_sentiment_label(s, pos_th=pos_th, neg_th=neg_th))
    comments["topic"] = comments["text"].apply(classify_topic)

    platform_dfs = filter_platforms(comments)
    if not platform_dfs:
        print("[WARN] 未匹配到平台关键词，美团/饿了么/淘宝 均为空。")
        return

    sentiment_df = sentiment_stack_data(platform_dfs)
    sentiment_df.to_csv(OUTPUT_DIR / "分平台情感占比.csv", index=False, encoding="utf-8-sig")
    plot_platform_sentiment_stack(sentiment_df, FIG_DIR / "平台情感占比_堆叠.png")

    topic_df = topic_ratio_platform(platform_dfs)
    topic_df.to_csv(OUTPUT_DIR / "分平台议题占比.csv", index=False, encoding="utf-8-sig")
    plot_topic_platform(topic_df, FIG_DIR / "平台议题占比_柱状.png")

    counters, top_words_df = wordfreq_platform(platform_dfs, topn=topn)
    top_words_df.to_csv(OUTPUT_DIR / "分平台词频Top.csv", index=False, encoding="utf-8-sig")
    plot_wordfreq_platform(counters, FIG_DIR / "平台词频Top.png", topn=topn)

    plot_platform_rolling_mean(platform_dfs, FIG_DIR / "平台情感滚动均值30天.png", time_col="create_time", score_col="sentiment_score", window=30)

    print("[OK] 完成，图表已输出到 分平台图/，CSV 在 处理了/ 内。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分平台情感与议题分析")
    parser.add_argument("--pos-th", type=float, default=0.6, help="正向阈值（>=此值为正向）")
    parser.add_argument("--neg-th", type=float, default=0.4, help="负向阈值（<=此值为负向）")
    parser.add_argument("--topn", type=int, default=15, help="词频TopN")
    args = parser.parse_args()
    try:
        main(pos_th=args.pos_th, neg_th=args.neg_th, topn=args.topn)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] {e}")
        sys.exit(1)
