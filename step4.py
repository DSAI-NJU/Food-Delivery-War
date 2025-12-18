"""
小红书文本分析 - 第四步：情感分析
根据《小红书文本分析详细操作指导(1).md》第4部分实现

功能概览：
- 读取原始帖子与评论 CSV（用户提供的绝对路径）
- 使用 SnowNLP 进行情感打分（0~1）与极性分类（正/中/负）
- 生成月度情感比例曲线图（帖子、评论）
- 导出情感结果与月度统计到 处理了/ 与 情感分析图/
"""

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
FIG_DIR = Path("情感分析图")
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
                font_manager.fontManager.addfont(ff)
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
    '价格补贴': ['补贴', '优惠', '便宜', '红包', '满减', '折扣', '价格', '省钱', '实惠', '便宜了'],
    '平台竞争': ['美团', '京东', '饿了么', '淘宝', '闪购', '竞争', '打仗', '大战', '对抗', '挑战'],
    '用户体验': ['体验', '方便', '快捷', '好用', '界面', '操作', '服务', '满意', '推荐', '喜欢'],
    '配送服务': ['配送', '外卖', '送餐', '骑手', '快递', '时间', '速度', '准时', '延迟', '送达'],
    '监管政策': ['监管', '政策', '法律', '约谈', '规定', '合规', '整改', '处罚', '反垄断', '竞争法'],
    '商家权益': ['商家', '店铺', '商户', '餐饮', '抽成', '佣金', '费用', '成本', '利润', '入驻'],
    '市场格局': ['市场', '份额', '占有率', '地位', '霸主', '龙头', '格局', '变化', '洗牌', '行业'],
    '消费习惯': ['习惯', '消费', '购买', '下单', '点餐', '选择', '偏好', '频率', '需求', '用户']
}


# =============== 工具函数 ===============
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


def get_sentiment_label(score: float, pos_th: float = 0.7, neg_th: float = 0.3) -> str:
    if pd.isna(score):
        return "中性"
    if score >= pos_th:
        return "正向"
    if score <= neg_th:
        return "负向"
    return "中性"


def snownlp_score(text: str) -> float:
    try:
        s = SnowNLP(str(text))
        return float(s.sentiments)
    except Exception:
        return np.nan


def plot_sentiment_area(df_long: pd.DataFrame, title: str, outfile: Path) -> None:
    if df_long.empty:
        print(f"[WARN] 无数据可绘制：{title}")
        return
    # 透视为宽表（每列为一个情感类别，值为比例）
    pivot = df_long.pivot_table(index="month", columns="sentiment", values="ratio", fill_value=0.0)
    # 按时间排序
    pivot = pivot.sort_index()
    # 保证列顺序
    cols = [c for c in ["负向", "中性", "正向"] if c in pivot.columns]
    pivot = pivot[cols]

    colors = [PALETTE.get(c) for c in cols]
    ax = pivot.plot.area(stacked=True, figsize=(10, 6), alpha=0.9, color=colors)
    ax.set_title(title, fontproperties=CH_FONT_PROP)
    ax.set_xlabel("月份", fontproperties=CH_FONT_PROP)
    ax.set_ylabel("比例", fontproperties=CH_FONT_PROP)
    ax.legend(title="情感", prop=CH_FONT_PROP, title_fontproperties=CH_FONT_PROP)
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=CH_FONT_PROP)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=CH_FONT_PROP)
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] 已保存图表：{outfile}")


def plot_monthly_stacked(df_long: pd.DataFrame, title: str, outfile: Path) -> None:
    if df_long.empty:
        print(f"[WARN] 无数据可绘制：{title}")
        return
    pivot = df_long.pivot_table(index="month", columns="sentiment", values="ratio", fill_value=0.0)
    cols = [c for c in ["负向", "中性", "正向"] if c in pivot.columns]
    pivot = pivot[cols]
    base_col = cols[0] if cols else None
    if base_col and base_col in pivot.columns:
        pivot = pivot.sort_values(by=base_col, ascending=False)
    else:
        pivot = pivot.sort_index()
    ax = pivot.plot(kind="bar", stacked=True, figsize=(12, 6), color=[PALETTE.get(c) for c in cols])
    ax.set_title(title, fontproperties=CH_FONT_PROP)
    ax.set_xlabel("月份", fontproperties=CH_FONT_PROP)
    ax.set_ylabel("比例", fontproperties=CH_FONT_PROP)
    ax.legend(title="情感", prop=CH_FONT_PROP, title_fontproperties=CH_FONT_PROP)
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=CH_FONT_PROP)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=CH_FONT_PROP)
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] 已保存图表：{outfile}")


def build_wordcloud(counter: Counter, title: str, outfile: Path, theme: str = "blue") -> None:
    try:
        from wordcloud import WordCloud
    except Exception:
        print("[WARN] wordcloud 未安装，跳过生成：", title)
        return
    if not counter:
        print(f"[WARN] 词频为空，跳过生成：{title}")
        return
    font_paths = [
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simsun.ttc',
    ]
    font_path = None
    for p in font_paths:
        if Path(p).exists():
            font_path = p
            break
    wc = WordCloud(font_path=font_path, width=1200, height=800, background_color='white', max_words=200,
                   collocations=False)
    wc.generate_from_frequencies(dict(counter))
    fig, ax = plt.subplots(figsize=(12, 8))
    # Apply soft color theme via colormap
    if theme == "blue":
        cmap = plt.cm.Blues
    elif theme == "yellow":
        cmap = plt.cm.YlOrBr
    else:
        cmap = plt.cm.Blues
    ax.imshow(wc.recolor(colormap=cmap), interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontproperties=CH_FONT_PROP)
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] 已保存词云：{outfile}")


def plot_rolling_mean(df: pd.DataFrame, time_col: str, score_col: str, title: str, outfile: Path, window: int = 15) -> None:
    if time_col not in df.columns:
        return
    tmp = df.dropna(subset=[time_col, score_col]).copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col])
    if tmp.empty:
        print(f"[WARN] 无数据可绘制滚动均值：{title}")
        return
    ts = tmp.set_index(time_col)[score_col].resample('D').mean().rolling(window, min_periods=1).mean()
    plt.figure(figsize=(12, 5))
    plt.plot(ts.index, ts.values, color="#8EC5FF")
    plt.title(title, fontproperties=CH_FONT_PROP)
    plt.xlabel("日期", fontproperties=CH_FONT_PROP)
    plt.ylabel(f"{window}天滚动均值", fontproperties=CH_FONT_PROP)
    plt.xticks(fontproperties=CH_FONT_PROP)
    plt.yticks(fontproperties=CH_FONT_PROP)
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] 已保存滚动均值图：{outfile}")


def plot_violin(df: pd.DataFrame, title: str, outfile: Path) -> None:
    if df.empty:
        print(f"[WARN] 无数据可绘制小提琴图：{title}")
        return
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df, x="dataset", y="sentiment_score", palette=["#8EC5FF", "#FFD88E", "#99A8B2"], cut=0)
    plt.title(title, fontproperties=CH_FONT_PROP)
    plt.xlabel("数据集", fontproperties=CH_FONT_PROP)
    plt.ylabel("情感得分", fontproperties=CH_FONT_PROP)
    plt.xticks(fontproperties=CH_FONT_PROP)
    plt.yticks(fontproperties=CH_FONT_PROP)
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] 已保存小提琴图：{outfile}")


def classify_to_topics(text: str) -> list[str]:
    matched = []
    for topic, keywords in TOPICS.items():
        for kw in keywords:
            if kw in str(text):
                matched.append(topic)
                break
    return matched if matched else ["其他"]


def build_topic_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "text" not in df.columns or "sentiment" not in df.columns:
        return pd.DataFrame()
    tmp = df[["text", "sentiment"]].copy()
    tmp["topics"] = tmp["text"].apply(classify_to_topics)
    exploded = tmp.explode("topics")
    exploded = exploded.dropna(subset=["topics", "sentiment"])
    if exploded.empty:
        return pd.DataFrame()
    counts = exploded.groupby(["topics", "sentiment"]).size().rename("count").reset_index()
    totals = counts.groupby("topics")["count"].sum().rename("total").reset_index()
    merged = counts.merge(totals, on="topics")
    merged["ratio"] = merged["count"] / merged["total"]
    merged = merged.rename(columns={"topics": "topic"})
    return merged


def time_bucket(dt: pd.Series, col: str) -> pd.Series:
    s = pd.to_datetime(dt, errors="coerce")
    hour = s.dt.hour
    buckets = pd.cut(
        hour,
        bins=[-1, 11, 17, 23],
        labels=["早上", "中午", "晚上"],
    )
    return buckets.astype(str)


def plot_timebucket_mean(df: pd.DataFrame, time_col: str, score_col: str, title: str, outfile: Path) -> None:
    if time_col not in df.columns or score_col not in df.columns:
        return
    tmp = df.dropna(subset=[time_col, score_col]).copy()
    if tmp.empty:
        print(f"[WARN] 无数据绘制时间段均值：{title}")
        return
    tmp["time_bucket"] = time_bucket(tmp[time_col], time_col)
    grouped = tmp.groupby("time_bucket")[score_col].mean().reindex(["早上", "中午", "晚上"])
    plt.figure(figsize=(8, 5))
    grouped.plot(kind="bar", color="#8EC5FF")
    plt.title(title, fontproperties=CH_FONT_PROP)
    plt.xlabel("时间段", fontproperties=CH_FONT_PROP)
    plt.ylabel("平均情感得分", fontproperties=CH_FONT_PROP)
    plt.xticks(rotation=0, fontproperties=CH_FONT_PROP)
    plt.yticks(fontproperties=CH_FONT_PROP)
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] 已保存时间段均值图：{outfile}")


def plot_timebucket_stack(df: pd.DataFrame, time_col: str, title: str, outfile: Path) -> None:
    if time_col not in df.columns or "sentiment" not in df.columns:
        return
    tmp = df.dropna(subset=[time_col, "sentiment"]).copy()
    if tmp.empty:
        print(f"[WARN] 无数据绘制时间段堆叠：{title}")
        return
    tmp["time_bucket"] = time_bucket(tmp[time_col], time_col)
    cnt = tmp.groupby(["time_bucket", "sentiment"]).size().rename("count").reset_index()
    total = cnt.groupby("time_bucket")["count"].sum().rename("total").reset_index()
    merged = cnt.merge(total, on="time_bucket")
    merged["ratio"] = merged["count"] / merged["total"]
    pivot = merged.pivot_table(index="time_bucket", columns="sentiment", values="ratio", fill_value=0.0)
    cols = [c for c in ["负向", "中性", "正向"] if c in pivot.columns]
    pivot = pivot.reindex(["早上", "中午", "晚上"]).fillna(0.0)
    pivot = pivot[cols]
    ax = pivot.plot(kind="bar", stacked=True, figsize=(10, 6), color=[PALETTE.get(c) for c in cols])
    ax.set_title(title, fontproperties=CH_FONT_PROP)
    ax.set_xlabel("时间段", fontproperties=CH_FONT_PROP)
    ax.set_ylabel("占比", fontproperties=CH_FONT_PROP)
    ax.legend(title="情感", prop=CH_FONT_PROP, title_fontproperties=CH_FONT_PROP)
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=CH_FONT_PROP)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=CH_FONT_PROP)
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] 已保存时间段堆叠图：{outfile}")


def plot_topic_sentiment_stack(df: pd.DataFrame, title: str, outfile: Path) -> None:
    if df.empty:
        print(f"[WARN] 无数据绘制：{title}")
        return
    pivot = df.pivot_table(index="topic", columns="sentiment", values="ratio", fill_value=0.0)
    cols = [c for c in ["负向", "中性", "正向"] if c in pivot.columns]
    pivot = pivot[cols]
    base_col = cols[0] if cols else None
    if base_col and base_col in pivot.columns:
        pivot = pivot.sort_values(by=base_col, ascending=False)
    ax = pivot.plot(kind="bar", stacked=True, figsize=(12, 7), color=[PALETTE.get(c) for c in cols])
    ax.set_title(title, fontproperties=CH_FONT_PROP)
    ax.set_xlabel("议题", fontproperties=CH_FONT_PROP)
    ax.set_ylabel("占比", fontproperties=CH_FONT_PROP)
    plt.xticks(rotation=30, ha="right")
    ax.legend(title="情感", prop=CH_FONT_PROP, title_fontproperties=CH_FONT_PROP)
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=CH_FONT_PROP)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=CH_FONT_PROP)
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] 议题情感堆叠图已保存：{outfile}")


def extract_platform_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    platforms = {
        "美团": ["美团"],
        "饿了么": ["饿了么"],
        "淘宝": ["淘宝"],
    }
    if "text" not in df.columns or "sentiment" not in df.columns:
        return pd.DataFrame()
    base = df.dropna(subset=["text", "sentiment"]).copy()
    rows = []
    for plat, kws in platforms.items():
        mask = base["text"].apply(lambda t: any(kw in str(t) for kw in kws))
        sub = base[mask]
        if sub.empty:
            continue
        cnt = sub.groupby("sentiment").size().rename("count").reset_index()
        total = cnt["count"].sum()
        cnt["platform"] = plat
        cnt["ratio"] = cnt["count"] / total
        rows.append(cnt)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_platform_stack(df: pd.DataFrame, title: str, outfile: Path) -> None:
    if df.empty:
        print(f"[WARN] 无数据可绘制平台情感堆叠：{title}")
        return
    pivot = df.pivot_table(index="platform", columns="sentiment", values="ratio", fill_value=0.0)
    cols = [c for c in ["负向", "中性", "正向"] if c in pivot.columns]
    pivot = pivot[cols]
    base_col = cols[0] if cols else None
    if base_col:
        pivot = pivot.sort_values(by=base_col, ascending=False)
    ax = pivot.plot(kind="bar", stacked=True, figsize=(10, 6), color=[PALETTE.get(c) for c in cols])
    ax.set_title(title, fontproperties=CH_FONT_PROP)
    ax.set_xlabel("平台", fontproperties=CH_FONT_PROP)
    ax.set_ylabel("情感占比", fontproperties=CH_FONT_PROP)
    ax.legend(title="情感", prop=CH_FONT_PROP, title_fontproperties=CH_FONT_PROP)
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=CH_FONT_PROP)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=CH_FONT_PROP)
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] 平台情感堆叠图已保存：{outfile}")


def load_key_events() -> pd.DataFrame:
    events_path = OUTPUT_DIR / "关键事件时间线.csv"
    if events_path.exists():
        df = pd.read_csv(events_path, encoding="utf-8")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    return pd.DataFrame(columns=["date", "event_name", "description"])


def month_labels_with_events(month_index: pd.Index, events: pd.DataFrame) -> list[str]:
    labels = []
    events = events.copy()
    if "date" in events.columns:
        events["month_str"] = events["date"].dt.to_period("M").astype(str)
    else:
        events["month_str"] = ""
    for m in month_index.astype(str):
        ev_names = events.loc[events["month_str"] == m, "event_name"].dropna().tolist()
        if ev_names:
            labels.append(m + "\n" + "\n".join(ev_names))
        else:
            labels.append(m)
    return labels


def plot_monthly_mean_sentiment(df: pd.DataFrame, time_col: str, score_col: str, title: str, outfile: Path, events: pd.DataFrame) -> None:
    if time_col not in df.columns or score_col not in df.columns:
        return
    tmp = df.dropna(subset=[time_col, score_col]).copy()
    if tmp.empty:
        print(f"[WARN] 无数据绘制月度均值：{title}")
        return
    tmp[time_col] = pd.to_datetime(tmp[time_col])
    monthly = tmp.set_index(time_col)[score_col].resample('M').mean()
    if monthly.empty:
        print(f"[WARN] 月度均值为空：{title}")
        return
    x = range(len(monthly))
    labels = month_labels_with_events(monthly.index, events)
    plt.figure(figsize=(14, 6))
    plt.bar(x, monthly.values, color="#8EC5FF", edgecolor="white")
    plt.xticks(x, labels, rotation=30, ha="right", fontproperties=CH_FONT_PROP)
    plt.yticks(fontproperties=CH_FONT_PROP)
    plt.title(title, fontproperties=CH_FONT_PROP)
    plt.xlabel("月份", fontproperties=CH_FONT_PROP)
    plt.ylabel("平均情感得分", fontproperties=CH_FONT_PROP)
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] 已保存月度均值图：{outfile}")


def plot_rolling_mean_generic(daily_series: pd.Series, window: int, title: str, outfile: Path) -> None:
    if daily_series.empty:
        print(f"[WARN] 无数据绘制滚动均值：{title}")
        return
    rolling = daily_series.rolling(window, min_periods=1).mean()
    plt.figure(figsize=(14, 6))
    plt.plot(daily_series.index, daily_series.values, color="#99A8B2", alpha=0.4, label="日均情感")
    plt.plot(rolling.index, rolling.values, color="#8EC5FF", linewidth=2.2, label=f"{window}天滚动均值")
    plt.title(title, fontproperties=CH_FONT_PROP)
    plt.xlabel("日期", fontproperties=CH_FONT_PROP)
    plt.ylabel("情感得分", fontproperties=CH_FONT_PROP)
    plt.xticks(fontproperties=CH_FONT_PROP, rotation=45, ha="right")
    plt.yticks(fontproperties=CH_FONT_PROP)
    plt.legend(prop=CH_FONT_PROP)
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] 已保存滚动均值图：{outfile}")


# =============== 主流程 ===============
def main(pos_th: float = 0.6, neg_th: float = 0.4, extreme_top: int = 500) -> None:
    print("=" * 70)
    print("第四步：情感分析")
    print("=" * 70)

    # 1) 读取数据
    print("\n[1/5] 加载原始 CSV ...")
    posts = read_csv_auto(POSTS_CSV)
    comments = read_csv_auto(COMMENTS_CSV)
    print(f"帖子：{len(posts)} 条，评论：{len(comments)} 条")

    # 读取关键事件时间线（用于月度标签）
    events_df = load_key_events()
    if events_df.empty:
        print("[WARN] 未找到关键事件时间线，将不显示事件标签。")

    # 2) 组装文本字段
    print("\n[2/5] 组装文本字段 ...")
    if "title" in posts.columns and "desc" in posts.columns:
        posts["title"] = posts["title"].fillna("")
        posts["desc"] = posts["desc"].fillna("")
        posts["text"] = (posts["title"].astype(str) + " " + posts["desc"].astype(str)).str.strip()
    else:
        # 兜底：若列名不符，尝试已有 text
        if "text" not in posts.columns:
            posts["text"] = posts.astype(str).agg(" ".join, axis=1)

    if "content" in comments.columns:
        comments["text"] = comments["content"].fillna("").astype(str)
    else:
        if "text" not in comments.columns:
            comments["text"] = comments.astype(str).agg(" ".join, axis=1)

    # 3) 时间字段与月份
    print("\n[3/5] 解析时间并生成月份 ...")
    if "time" in posts.columns:
        posts["time"] = ensure_datetime(posts["time"])  # 帖子时间
    if "create_time" in comments.columns:
        comments["create_time"] = ensure_datetime(comments["create_time"])  # 评论时间

    posts["month"] = posts.get("time", pd.to_datetime(pd.NaT)).dt.to_period("M").astype(str)
    comments["month"] = comments.get("create_time", pd.to_datetime(pd.NaT)).dt.to_period("M").astype(str)

    # 4) 情感打分
    print("\n[4/5] 使用 SnowNLP 进行情感打分 ...")
    # 对较长数据集，逐行计算
    posts["sentiment_score"] = posts["text"].apply(snownlp_score)
    comments["sentiment_score"] = comments["text"].apply(snownlp_score)

    posts["sentiment"] = posts["sentiment_score"].apply(lambda s: get_sentiment_label(s, pos_th=pos_th, neg_th=neg_th))
    comments["sentiment"] = comments["sentiment_score"].apply(lambda s: get_sentiment_label(s, pos_th=pos_th, neg_th=neg_th))

    # 导出明细
    post_detail_cols = [c for c in [
        "note_id", "time", "month", "title", "desc", "text", "sentiment_score", "sentiment"
    ] if c in posts.columns]
    comment_detail_cols = [c for c in [
        "note_id", "comment_id", "create_time", "month", "content", "text", "sentiment_score", "sentiment"
    ] if c in comments.columns]

    posts[post_detail_cols].to_csv(OUTPUT_DIR / "帖子情感打分.csv", index=False, encoding="utf-8-sig")
    comments[comment_detail_cols].to_csv(OUTPUT_DIR / "评论情感打分.csv", index=False, encoding="utf-8-sig")
    print("[OK] 已保存：处理了/帖子情感打分.csv、处理了/评论情感打分.csv")

    # 5) 月度情感比例（帖子 & 评论）
    print("\n[5/5] 统计并绘制月度情感比例 ...")
    def monthly_ratio(df: pd.DataFrame, time_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        tmp = df.dropna(subset=["month", "sentiment"]).copy()
        cnt = tmp.groupby(["month", "sentiment"]).size().rename("count").reset_index()
        total = cnt.groupby("month")["count"].sum().rename("total").reset_index()
        merged = cnt.merge(total, on="month", how="left")
        merged["ratio"] = (merged["count"] / merged["total"]).astype(float)
        # 长表与宽表
        wide = merged.pivot_table(index="month", columns="sentiment", values="count", fill_value=0)
        wide = wide.sort_index()
        return merged, wide

    posts_long, posts_wide = monthly_ratio(posts, "time")
    comments_long, comments_wide = monthly_ratio(comments, "create_time")

    posts_long.to_csv(OUTPUT_DIR / "月度情感_帖子_长表.csv", index=False, encoding="utf-8-sig")
    posts_wide.to_csv(OUTPUT_DIR / "月度情感_帖子_宽表.csv", encoding="utf-8-sig")
    comments_long.to_csv(OUTPUT_DIR / "月度情感_评论_长表.csv", index=False, encoding="utf-8-sig")
    comments_wide.to_csv(OUTPUT_DIR / "月度情感_评论_宽表.csv", encoding="utf-8-sig")
    print("[OK] 已保存月度情感统计 CSV（长表/宽表）")

    # 绘图
    plot_sentiment_area(posts_long, "帖子：情感比例随时间变化", FIG_DIR / "情感比例_帖子_面积图.png")
    plot_sentiment_area(comments_long, "评论：情感比例随时间变化", FIG_DIR / "情感比例_评论_面积图.png")
    # 追加：堆叠柱状图
    plot_monthly_stacked(posts_long, "帖子：每月情感占比（堆叠柱）", FIG_DIR / "情感占比_帖子_堆叠柱.png")
    plot_monthly_stacked(comments_long, "评论：每月情感占比（堆叠柱）", FIG_DIR / "情感占比_评论_堆叠柱.png")

    # ========== 新增：平台情感占比（评论，平台：美团/饿了么/淘宝） ==========
    print("\n[附加] 平台情感占比（评论，美团/饿了么/淘宝） ...")
    platform_sent = extract_platform_sentiment(comments)
    if not platform_sent.empty:
        platform_sent.to_csv(OUTPUT_DIR / "平台情感占比_评论.csv", index=False, encoding="utf-8-sig")
        plot_platform_stack(platform_sent, "三平台评论情感占比", FIG_DIR / "情感占比_平台堆叠_评论.png")
    else:
        print("[WARN] 未找到包含关键词的评论，跳过平台情感堆叠图。")

    # ========== 新增：月度均值 & 30天滚动均值（合并，带事件标签） ==========
    print("\n[附加] 月度均值与30天滚动均值（合并） ...")
    combined_time_full = pd.concat([
        posts[["time", "sentiment_score"]].rename(columns={"time": "ts"}),
        comments[["create_time", "sentiment_score"]].rename(columns={"create_time": "ts"})
    ], ignore_index=True)
    combined_time_full = combined_time_full.dropna(subset=["ts", "sentiment_score"]).copy()
    if not combined_time_full.empty:
        combined_time_full["ts"] = pd.to_datetime(combined_time_full["ts"])
        plot_monthly_mean_sentiment(
            combined_time_full.rename(columns={"ts": "time"}),
            "time",
            "sentiment_score",
            "情感得分月度均值（合并）",
            FIG_DIR / "月度均分_合并.png",
            events_df,
        )
        daily_mean = combined_time_full.set_index("ts")["sentiment_score"].resample('D').mean()
        plot_rolling_mean_generic(daily_mean, 30, "情感得分30天滚动均值（合并）", FIG_DIR / "滚动均值30天_合并.png")
    else:
        print("[WARN] 合并后无时间或情感得分数据，无法绘制月度/滚动均值图。")

    # ========== 词频与词云（正/中/负），帖子/评论/合并 ==========
    print("\n[附加] 统计分词词频并生成词云 ...")
    def tokenize(text: str) -> list[str]:
        return [w for w in jieba.lcut(str(text)) if w and len(w) > 1]

    # 为帖子与评论计算分词
    posts_words = posts["text"].apply(tokenize)
    comments_words = comments["text"].apply(tokenize)

    def sentiment_word_counter(df_words: pd.Series, sentiments: pd.Series) -> dict:
        buckets = {"正向": [], "中性": [], "负向": []}
        for ws, s in zip(df_words, sentiments):
            if s in buckets:
                buckets[s].extend(ws)
        return {k: Counter(v) for k, v in buckets.items()}

    post_counters = sentiment_word_counter(posts_words, posts["sentiment"])
    comment_counters = sentiment_word_counter(comments_words, comments["sentiment"])
    combined_counters = {
        k: post_counters.get(k, Counter()) + comment_counters.get(k, Counter()) for k in ["正向", "中性", "负向"]
    }

    def save_overall_freq(counters: dict, prefix: str, top_n: int | None = None):
        for label, cnt in counters.items():
            pairs = cnt.most_common(top_n) if top_n else cnt.most_common()
            df = pd.DataFrame(pairs, columns=["word", "freq"])
            df.to_csv(OUTPUT_DIR / f"整体词频_{prefix}_{label}.csv", index=False, encoding="utf-8-sig")

    # 保存 Top 50（可按需调整）
    save_overall_freq(post_counters, "帖子", top_n=50)
    save_overall_freq(comment_counters, "评论", top_n=50)
    save_overall_freq(combined_counters, "合并", top_n=50)

    # 月度词频：按月聚合后再分情感词频
    def monthly_sentiment_wordfreq(df: pd.DataFrame, words_series: pd.Series, prefix: str, time_col: str):
        tmp = df[["month", "sentiment"]].copy()
        tmp["words"] = words_series.values
        grouped = tmp.dropna(subset=["month"]).groupby(["month", "sentiment"])
        rows = []
        for (m, s), g in grouped:
            cnt = Counter([w for ws in g["words"] for w in ws])
            for w, f in cnt.most_common():
                rows.append({"month": m, "sentiment": s, "word": w, "freq": f})
        out = pd.DataFrame(rows)
        out.to_csv(OUTPUT_DIR / f"月度词频_{prefix}.csv", index=False, encoding="utf-8-sig")
        return out

    monthly_sentiment_wordfreq(posts, posts_words, "帖子", "time")
    monthly_sentiment_wordfreq(comments, comments_words, "评论", "create_time")
    # 合并
    combined_df = pd.DataFrame({
        "month": pd.concat([posts["month"], comments["month"]], ignore_index=True),
        "sentiment": pd.concat([posts["sentiment"], comments["sentiment"]], ignore_index=True),
        "words": pd.concat([posts_words, comments_words], ignore_index=True),
    })
    monthly_sentiment_wordfreq(combined_df, combined_df["words"], "合并", "month")

    # 词云（整体，正/中/负）：三份
    for label, cnt in post_counters.items():
        theme = "blue" if label != "负向" else "yellow"
        build_wordcloud(cnt, f"帖子整体{label}词云", FIG_DIR / f"词云_帖子_{label}.png", theme=theme)
    for label, cnt in comment_counters.items():
        theme = "blue" if label != "负向" else "yellow"
        build_wordcloud(cnt, f"评论整体{label}词云", FIG_DIR / f"词云_评论_{label}.png", theme=theme)
    for label, cnt in combined_counters.items():
        theme = "blue" if label != "负向" else "yellow"
        build_wordcloud(cnt, f"合并整体{label}词云", FIG_DIR / f"词云_合并_{label}.png", theme=theme)

    # ========== 极端情感评论词云（正向 TopN / 负向 TopN） ==========
    print("\n[附加] 极端情感评论 TopN 词云 ...")
    if "sentiment_score" in comments.columns and "text" in comments.columns:
        dfc = comments.dropna(subset=["sentiment_score", "text"]).copy()
        # 负向程度最高（得分最低）的 TopN
        neg_top = dfc.nsmallest(extreme_top, "sentiment_score")
        pos_top = dfc.nlargest(extreme_top, "sentiment_score")

        neg_tokens = neg_top["text"].apply(lambda t: [w for w in jieba.lcut(str(t)) if w and len(w) > 1])
        pos_tokens = pos_top["text"].apply(lambda t: [w for w in jieba.lcut(str(t)) if w and len(w) > 1])

        neg_counter = Counter([w for ws in neg_tokens for w in ws])
        pos_counter = Counter([w for ws in pos_tokens for w in ws])

        build_wordcloud(neg_counter, f"评论 负向Top{extreme_top} 词云", FIG_DIR / f"词云_评论_负向Top{extreme_top}.png", theme="yellow")
        build_wordcloud(pos_counter, f"评论 正向Top{extreme_top} 词云", FIG_DIR / f"词云_评论_正向Top{extreme_top}.png", theme="blue")

    # ========== 周度情感得分均分折线图（帖子/评论/合并） ==========
    print("\n[附加] 计算周度平均情感得分并绘制折线 ...")
    def weekly_mean(df: pd.DataFrame, score_col: str, time_col: str, title: str, outfile: Path):
        if time_col not in df.columns:
            return
        tmp = df.dropna(subset=[time_col, score_col]).copy()
        tmp["week"] = pd.to_datetime(tmp[time_col]).dt.to_period("W").astype(str)
        s = tmp.groupby("week")[score_col].mean().sort_index()
        s.to_csv(OUTPUT_DIR / f"周度情感均分_{title}.csv", encoding="utf-8-sig")
        plt.figure(figsize=(12, 5))
        s.plot(color="#8EC5FF")
        plt.title(f"周度平均情感得分：{title}", fontproperties=CH_FONT_PROP)
        plt.xlabel("周", fontproperties=CH_FONT_PROP)
        plt.ylabel("平均得分", fontproperties=CH_FONT_PROP)
        plt.xticks(fontproperties=CH_FONT_PROP)
        plt.yticks(fontproperties=CH_FONT_PROP)
        plt.tight_layout()
        plt.savefig(str(outfile), dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[OK] 已保存周度折线：{outfile}")

    weekly_mean(posts, "sentiment_score", "time", "帖子", FIG_DIR / "周度均分_帖子.png")
    weekly_mean(comments, "sentiment_score", "create_time", "评论", FIG_DIR / "周度均分_评论.png")
    # 合并周度
    combined_time = pd.concat([
        posts[["time", "sentiment_score"]].rename(columns={"time": "ts"}),
        comments[["create_time", "sentiment_score"]].rename(columns={"create_time": "ts"})
    ], ignore_index=True)
    combined_time = combined_time.dropna(subset=["ts", "sentiment_score"])  # type: ignore[index]
    combined_time["week"] = pd.to_datetime(combined_time["ts"]).dt.to_period("W").astype(str)
    s = combined_time.groupby("week")["sentiment_score"].mean().sort_index()
    s.to_csv(OUTPUT_DIR / "周度情感均分_合并.csv", encoding="utf-8-sig")
    plt.figure(figsize=(12, 5))
    s.plot(color="#8EC5FF")
    plt.title("周度平均情感得分：合并", fontproperties=CH_FONT_PROP)
    plt.xlabel("周", fontproperties=CH_FONT_PROP)
    plt.ylabel("平均得分", fontproperties=CH_FONT_PROP)
    plt.xticks(fontproperties=CH_FONT_PROP)
    plt.yticks(fontproperties=CH_FONT_PROP)
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "周度均分_合并.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK] 已保存周度折线：", FIG_DIR / "周度均分_合并.png")

    # ========== 15天滚动均值（帖子/评论/合并） ==========
    print("\n[附加] 15天滚动均值 ...")
    plot_rolling_mean(posts, "time", "sentiment_score", "帖子情感得分15天滚动均值", FIG_DIR / "滚动均值15天_帖子.png", window=15)
    plot_rolling_mean(comments, "create_time", "sentiment_score", "评论情感得分15天滚动均值", FIG_DIR / "滚动均值15天_评论.png", window=15)
    if not combined_time_full.empty:
        tmp = combined_time_full.set_index("ts")["sentiment_score"].resample('D').mean().rolling(15, min_periods=1).mean()
        plt.figure(figsize=(12, 5))
        plt.plot(tmp.index, tmp.values, color="#8EC5FF")
        plt.title("合并情感得分15天滚动均值", fontproperties=CH_FONT_PROP)
        plt.xlabel("日期", fontproperties=CH_FONT_PROP)
        plt.ylabel("15天滚动均值", fontproperties=CH_FONT_PROP)
        plt.xticks(fontproperties=CH_FONT_PROP)
        plt.yticks(fontproperties=CH_FONT_PROP)
        plt.tight_layout()
        plt.savefig(str(FIG_DIR / "滚动均值15天_合并.png"), dpi=200, bbox_inches="tight")
        plt.close()
        print("[OK] 已保存滚动均值：", FIG_DIR / "滚动均值15天_合并.png")
    else:
        print("[WARN] 合并后无数据，跳过15天滚动均值图。")

    # ========== 议题 × 情感占比堆叠图（帖子/评论/合并） ==========
    print("\n[附加] 议题 × 情感占比堆叠图 ...")
    post_topic_sent = build_topic_sentiment(posts)
    comment_topic_sent = build_topic_sentiment(comments)
    if not post_topic_sent.empty:
        plot_topic_sentiment_stack(post_topic_sent, "帖子：各议题情感占比", FIG_DIR / "议题情感堆叠_帖子.png")
    if not comment_topic_sent.empty:
        plot_topic_sentiment_stack(comment_topic_sent, "评论：各议题情感占比", FIG_DIR / "议题情感堆叠_评论.png")
    if not post_topic_sent.empty and not comment_topic_sent.empty:
        combined_ts = pd.concat([post_topic_sent, comment_topic_sent], ignore_index=True)
        combined_ts = combined_ts.groupby(["topic", "sentiment"])["count"].sum().reset_index()
        totals = combined_ts.groupby("topic")["count"].sum().rename("total").reset_index()
        combined_ts = combined_ts.merge(totals, on="topic")
        combined_ts["ratio"] = combined_ts["count"] / combined_ts["total"]
        plot_topic_sentiment_stack(combined_ts, "合并：各议题情感占比", FIG_DIR / "议题情感堆叠_合并.png")

    # ========== 情感得分小提琴图（帖子/评论/合并） ==========
    print("\n[附加] 情感得分小提琴图 ...")
    violin_df = []
    if "sentiment_score" in posts.columns:
        violin_df.append(posts[["sentiment_score"]].assign(dataset="帖子"))
    if "sentiment_score" in comments.columns:
        violin_df.append(comments[["sentiment_score"]].assign(dataset="评论"))
    if violin_df:
        vdf = pd.concat(violin_df, ignore_index=True).dropna(subset=["sentiment_score"])
        plot_violin(vdf, "情感得分分布（小提琴图）", FIG_DIR / "情感得分_小提琴图.png")

    # ========== 时间段（凌晨/早上/中午/下午/晚上）均值与堆叠 ==========
    print("\n[附加] 时间段情感均值与堆叠 ...")
    plot_timebucket_mean(posts, "time", "sentiment_score", "帖子情感得分（按时间段均值）", FIG_DIR / "时间段均值_帖子.png")
    plot_timebucket_mean(comments, "create_time", "sentiment_score", "评论情感得分（按时间段均值）", FIG_DIR / "时间段均值_评论.png")
    # 合并均值
    combined_tb = pd.concat([
        posts[["time", "sentiment_score"]].rename(columns={"time": "ts"}),
        comments[["create_time", "sentiment_score"]].rename(columns={"create_time": "ts"})
    ], ignore_index=True)
    combined_tb = combined_tb.dropna(subset=["ts", "sentiment_score"])
    if not combined_tb.empty:
        combined_tb_plot = combined_tb.copy()
        combined_tb_plot.rename(columns={"ts": "time"}, inplace=True)
        plot_timebucket_mean(combined_tb_plot, "time", "sentiment_score", "合并情感得分（按时间段均值）", FIG_DIR / "时间段均值_合并.png")

    # 堆叠占比
    plot_timebucket_stack(posts, "time", "帖子：时间段情感占比", FIG_DIR / "时间段占比_帖子.png")
    plot_timebucket_stack(comments, "create_time", "评论：时间段情感占比", FIG_DIR / "时间段占比_评论.png")
    if not combined_tb.empty:
        combined_tb_plot = combined_tb.copy()
        combined_tb_plot.rename(columns={"ts": "time"}, inplace=True)
        # 需有 sentiment 列；此处重用原 posts/comments 的 sentiment 合并
        comb_sent = pd.concat([
            posts[["time", "sentiment"]].rename(columns={"time": "ts"}),
            comments[["create_time", "sentiment"]].rename(columns={"create_time": "ts"})
        ], ignore_index=True)
        comb_sent = comb_sent.dropna(subset=["ts", "sentiment"]).rename(columns={"ts": "time"})
        plot_timebucket_stack(comb_sent, "time", "合并：时间段情感占比", FIG_DIR / "时间段占比_合并.png")

    # 概览汇总
    summary = pd.DataFrame({
        "数据集": ["帖子", "评论"],
        "样本数": [len(posts), len(comments)],
        "正向": [int((posts["sentiment"] == "正向").sum()), int((comments["sentiment"] == "正向").sum())],
        "中性": [int((posts["sentiment"] == "中性").sum()), int((comments["sentiment"] == "中性").sum())],
        "负向": [int((posts["sentiment"] == "负向").sum()), int((comments["sentiment"] == "负向").sum())],
    })
    summary.to_csv(OUTPUT_DIR / "情感统计汇总.csv", index=False, encoding="utf-8-sig")
    print("[OK] 已保存：处理了/情感统计汇总.csv")

    print("\n" + "=" * 70)
    print("情感分析完成！")
    print("输出：")
    print("  • 处理了/帖子情感打分.csv")
    print("  • 处理了/评论情感打分.csv")
    print("  • 处理了/月度情感_帖子_长表.csv、宽表.csv")
    print("  • 处理了/月度情感_评论_长表.csv、宽表.csv")
    print("  • 处理了/情感统计汇总.csv")
    print("  • 处理了/平台情感占比_评论.csv")
    print("  • 情感分析图/情感比例_帖子_面积图.png、情感比例_评论_面积图.png")
    print("  • 情感分析图/情感占比_帖子_堆叠柱.png、情感占比_评论_堆叠柱.png")
    print("  • 情感分析图/情感占比_平台堆叠_评论.png")
    print("  • 情感分析图/词云_帖子_*.png、词云_评论_*.png、词云_合并_*.png")
    print("  • 情感分析图/周度均分_帖子.png、周度均分_评论.png、周度均分_合并.png")
    print("  • 情感分析图/月度均分_合并.png、滚动均值30天_合并.png")
    print("  • 情感分析图/滚动均值15天_*.png、情感得分_小提琴图.png")
    print("  • 情感分析图/议题情感堆叠_帖子.png、议题情感堆叠_评论.png、议题情感堆叠_合并.png")
    print("  • 处理了/整体词频_*_*.csv 与 处理了/月度词频_*_.csv、周度情感均分_*.csv")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="小红书文本分析 - 情感分析")
    parser.add_argument("--pos-th", type=float, default=0.6, help="正向阈值（>=此值为正向）")
    parser.add_argument("--neg-th", type=float, default=0.4, help="负向阈值（<=此值为负向）")
    parser.add_argument("--extreme-top", type=int, default=500, help="极端情感评论词云取 TopN")
    args = parser.parse_args()
    try:
        main(pos_th=args.pos_th, neg_th=args.neg_th, extreme_top=args.extreme_top)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] {e}")
        sys.exit(1)
