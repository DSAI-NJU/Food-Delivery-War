

import sys
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import shutil
from matplotlib.ticker import PercentFormatter
import matplotlib.font_manager as fm
import seaborn as sns


# ==================== 配置 ====================
STEP_DIR = Path('step3')
FIG_DIR = STEP_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

COMMENTS_FILE = STEP_DIR / 'comments_filtered.csv'
POSTS_FILE = STEP_DIR / 'posts_filtered.csv'

# 全局中文字体属性（若成功设置则用于所有文本元素）
CN_FONTPROP = None


def setup_chinese_fonts():
    """设置可用的中文字体（优先注册 Windows 字体文件），避免中文缺字；返回选用的字体名称"""
    import seaborn as sns
    from pathlib import Path

    # 优先尝试在 Windows 上直接注册常见中文字体文件
    font_file_candidates = [
        ('Microsoft YaHei', r'C:\\Windows\\Fonts\\msyh.ttc'),
        ('Microsoft YaHei', r'C:\\Windows\\Fonts\\msyh.ttf'),
        ('SimHei', r'C:\\Windows\\Fonts\\simhei.ttf'),
        ('SimSun', r'C:\\Windows\\Fonts\\simsun.ttc'),
        ('NSimSun', r'C:\\Windows\\Fonts\\nsimsun.ttc'),
        ('KaiTi', r'C:\\Windows\\Fonts\\kaiti.ttf'),
        ('Microsoft JhengHei', r'C:\\Windows\\Fonts\\msjh.ttc'),
    ]

    global CN_FONTPROP
    chosen = None
    fontprop = None
    try:
        for name, fpath in font_file_candidates:
            try:
                if Path(fpath).exists():
                    fm.fontManager.addfont(fpath)
                    # 重新构建字体缓存，确保新注册字体可被 findfont 使用
                    try:
                        fm._rebuild()
                    except Exception:
                        pass
                    chosen = name
                    # 指定 FontProperties 以确保所有文本明确使用该字体文件
                    try:
                        fontprop = fm.FontProperties(fname=fpath)
                    except Exception:
                        fontprop = None
                    break
            except Exception:
                continue

        # 若未通过文件注册成功，则根据已安装字体名称回退
        if not chosen:
            available_names = [f.name for f in fm.fontManager.ttflist]
            for name in ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'KaiTi', 'Noto Sans CJK SC', 'Arial Unicode MS']:
                if name in available_names:
                    chosen = name
                    try:
                        fontprop = fm.FontProperties(family=name)
                    except Exception:
                        fontprop = None
                    break

        # 应用到 rcParams
        # 优先直接指定所选字体为 family，确保各图形元素统一使用
        if chosen:
            plt.rcParams['font.family'] = chosen
        else:
            plt.rcParams['font.family'] = 'sans-serif'
        if chosen:
            plt.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
        else:
            # 兜底：尝试使用 SimHei
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_theme(style='whitegrid', context='talk')
        CN_FONTPROP = fontprop
    except Exception:
        sns.set_theme(style='whitegrid', context='talk')
    return chosen


def load_topics() -> Dict[str, List[str]]:
    """从 step3/topic_keywords_generated.py 加载 TOPICS（使用 importlib 兼容加载）"""
    import importlib.util
    file_path = STEP_DIR / 'topic_keywords_generated.py'
    if not file_path.exists():
        print(f"❌ 未找到文件：{file_path}")
        return {}
    try:
        spec = importlib.util.spec_from_file_location('topic_keywords_generated', str(file_path))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        if hasattr(module, 'TOPICS'):
            return module.TOPICS
        print('❌ 文件中未定义 TOPICS 变量')
    except Exception as e:
        print(f"❌ 无法加载主题关键词：{e}")
    return {}


def standardize_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """将时间列标准化为 datetime；尽量兼容不同列名"""
    time_cols = ['time', 'created_at', 'publish_time', 'datetime']
    col = None
    for c in time_cols:
        if c in df.columns:
            col = c
            break
    if col is None:
        # 如果没有时间列，直接丢弃（无法做时间序列）
        df = df.copy()
        df['time'] = pd.NaT
        return df
    df = df.copy()
    df['time'] = pd.to_datetime(df[col], errors='coerce')
    return df


def get_text_column(df: pd.DataFrame) -> str:
    """返回最可能的文本列名"""
    for c in ['text', 'content', 'desc', 'title']:
        if c in df.columns:
            return c
    # 兜底：首列
    return df.columns[0]


def detect_topics(text: str, topics: Dict[str, List[str]]) -> Dict[str, int]:
    """根据关键词匹配检测文本涉及的议题，返回 {topic: 命中关键词数量}"""
    if not isinstance(text, str) or not text:
        return {}
    s = text.lower()
    # 简单清洗，去除控制符
    s = re.sub(r"\s+", " ", s)
    matched = {}
    for topic, kws in topics.items():
        count = 0
        for kw in kws:
            k = str(kw).lower()
            if not k:
                continue
            if k in s:
                count += 1
        if count > 0:
            matched[topic] = count
    return matched


def build_time_series(df: pd.DataFrame, topics: Dict[str, List[str]], bucket: str = 'W') -> pd.DataFrame:
    """
    生成按时间桶（日D/周W/月M）的议题关注度时间序列。
    关注度 = 文本中命中关键词的数量（去除重复按关键词计数）。
    """
    df = df.copy()
    text_col = get_text_column(df)

    # 过滤掉没有时间的数据
    df = df[~df['time'].isna()]
    if df.empty:
        return pd.DataFrame(columns=['date', 'topic', 'attention'])

    # 时间桶
    df['date'] = df['time'].dt.to_period(bucket).dt.to_timestamp()

    # 计算每行的议题匹配
    rows = []
    for _, row in df.iterrows():
        topics_hits = detect_topics(str(row[text_col]), topics)
        for topic, hit in topics_hits.items():
            rows.append({'date': row['date'], 'topic': topic, 'attention': hit})

    ts = pd.DataFrame(rows)
    if ts.empty:
        return pd.DataFrame(columns=['date', 'topic', 'attention'])

    # 聚合
    ts_agg = ts.groupby(['date', 'topic'], as_index=False)['attention'].sum()
    ts_agg = ts_agg.sort_values(['date', 'topic'])
    return ts_agg


def _prepare_monthly_pivot(ts_monthly: pd.DataFrame) -> pd.DataFrame:
    """将月度数据透视为 (date x topic) 的矩阵"""
    if ts_monthly.empty:
        return pd.DataFrame()
    months = sorted(ts_monthly['date'].unique())
    topics = sorted(ts_monthly['topic'].unique())
    pivot = ts_monthly.pivot_table(index='date', columns='topic', values='attention', aggfunc='sum').fillna(0)
    pivot = pivot.reindex(index=months).reindex(columns=topics, fill_value=0)
    return pivot


def _get_topic_palette(topics: list):
    """为议题生成稳定且易读的配色，取自 YlGnBu 色系以与热力图一致"""
    cmap = plt.get_cmap('YlGnBu')
    # 采样区间避开过浅/过深，以保持可读性
    samples = np.linspace(0.25, 0.9, len(topics)) if len(topics) > 1 else [0.6]
    base = [cmap(x) for x in samples]
    return {t: base[i] for i, t in enumerate(topics)}


def _plot_monthly_line(pivot: pd.DataFrame):
    if pivot.empty:
        return None
    setup_chinese_fonts()
    df_long = pivot.reset_index().melt(id_vars='date', var_name='topic', value_name='attention')
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_long, x='date', y='attention', hue='topic', marker='o')
    plt.title('议题关注度 - 月度变化（折线）')
    plt.xlabel('月')
    plt.ylabel('关注度（关键词命中计数）')
    plt.legend(title='议题', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    out_path = FIG_DIR / 'topic_timeline_monthly.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ 已保存图表：{out_path}")
    return out_path


def _plot_monthly_area(pivot: pd.DataFrame):
    if pivot.empty:
        return None
    setup_chinese_fonts()
    plt.figure(figsize=(12, 6))
    x = pd.to_datetime(pivot.index)
    y = pivot.values.T  # topics x months
    plt.stackplot(x, *y, labels=pivot.columns)
    plt.title('议题关注度 - 月度堆叠面积图')
    plt.xlabel('月')
    plt.ylabel('关注度')
    plt.legend(title='议题', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    out_path = FIG_DIR / 'topic_area_monthly.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ 已保存图表：{out_path}")
    return out_path


def _plot_monthly_heatmap(pivot: pd.DataFrame):
    if pivot.empty:
        return None
    setup_chinese_fonts()
    # 格式化横轴为 YYYY-MM（去掉 00:00）
    month_labels = pd.to_datetime(pivot.index).strftime('%Y-%m')
    plt.figure(figsize=(max(12, len(pivot.index) * 0.7), 7))
    ax = sns.heatmap(
        pivot.T,
        cmap='YlGnBu',
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': '关注度'}
    )
    if CN_FONTPROP:
        ax.set_title('议题关注度 - 月度热力图', pad=10, fontproperties=CN_FONTPROP)
        ax.set_xlabel('月', fontproperties=CN_FONTPROP)
        ax.set_ylabel('议题', fontproperties=CN_FONTPROP)
    else:
        ax.set_title('议题关注度 - 月度热力图', pad=10)
        ax.set_xlabel('月')
        ax.set_ylabel('议题')
    # 设置格式化后的刻度标签
    ax.set_xticklabels(month_labels, rotation=45, ha='right')
    if CN_FONTPROP:
        for lbl in ax.get_xticklabels():
            lbl.set_fontproperties(CN_FONTPROP)
        for lbl in ax.get_yticklabels():
            lbl.set_fontproperties(CN_FONTPROP)
    # 清理边框
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    out_path = FIG_DIR / 'topic_heatmap_monthly.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ 已保存图表：{out_path}")
    return out_path


def _animate_monthly_line(pivot: pd.DataFrame):
    """生成月度折线动画（GIF）；若动画保存失败则输出帧 PNG"""
    if pivot.empty:
        return None
    setup_chinese_fonts()
    months = pd.to_datetime(pivot.index)
    topics = list(pivot.columns)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title('议题关注度 - 月度变化（动画）')
    ax.set_xlabel('月')
    ax.set_ylabel('关注度')
    lines = {}
    for topic in topics:
        (line,) = ax.plot([], [], label=topic)
        lines[topic] = line
    ax.legend(title='议题', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    def init():
        for line in lines.values():
            line.set_data([], [])
        return list(lines.values())

    def update(frame):
        ax.set_xlim(months.min(), months.max())
        # 自动设置 y 轴上限
        max_y = float(pivot.iloc[: frame + 1].values.max()) if frame >= 0 else 1.0
        ax.set_ylim(0, max(1.0, max_y * 1.1))
        for topic in topics:
            x = months[: frame + 1]
            y = pivot[topic].values[: frame + 1]
            lines[topic].set_data(x, y)
        return list(lines.values())

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(months), interval=400, blit=True)
    gif_path = FIG_DIR / 'topic_timeline_monthly_anim.gif'
    try:
        from matplotlib.animation import PillowWriter
        anim.save(str(gif_path), writer=PillowWriter(fps=3))
        plt.close(fig)
        print(f"✓ 已保存动画：{gif_path}")
        return gif_path
    except Exception as e:
        # 回退：保存帧序列
        frame_dir = FIG_DIR / 'anim_monthly_frames'
        frame_dir.mkdir(exist_ok=True)
        for i in range(len(months)):
            update(i)
            fig.savefig(frame_dir / f'frame_{i:03d}.png', dpi=150)
        plt.close(fig)
        print(f"⚠️ 动画保存失败（{e}），已输出帧到：{frame_dir}")
        return frame_dir


def _plot_monthly_stacked_bar(pivot: pd.DataFrame, normalize: bool = False):
    if pivot.empty:
        return None
    setup_chinese_fonts()
    data = pivot.copy()
    if normalize:
        # 百分比堆叠（每月总和归一化为 100%）
        col_sum = data.sum(axis=1)
        col_sum = col_sum.replace(0, np.nan)
        data = data.div(col_sum, axis=0).fillna(0) * 100
    # 绘制堆叠柱状图
    # 使用类别索引保证柱宽清晰
    month_labels = pd.to_datetime(data.index).strftime('%Y-%m')
    x_idx = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = np.zeros(len(data))
    topics = list(data.columns)
    palette = _get_topic_palette(topics)
    width = 0.7
    for topic in data.columns:
        ax.bar(x_idx, data[topic].values, bottom=bottom, label=topic, color=palette[topic], width=width)
        bottom += data[topic].values
    title_text = '议题关注度 - 月度堆叠柱状图' + ('（百分比）' if normalize else '')
    ylabel_text = '关注度' + ('占比（%）' if normalize else '')
    if CN_FONTPROP:
        ax.set_title(title_text, pad=10, fontproperties=CN_FONTPROP)
        ax.set_xlabel('月', fontproperties=CN_FONTPROP)
        ax.set_ylabel(ylabel_text, fontproperties=CN_FONTPROP)
    else:
        ax.set_title(title_text, pad=10)
        ax.set_xlabel('月')
        ax.set_ylabel(ylabel_text)
    if normalize:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    if CN_FONTPROP:
        ax.legend(title='议题', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, prop=CN_FONTPROP)
    else:
        ax.legend(title='议题', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    # 美化刻度
    ax.set_xticks(x_idx)
    ax.set_xticklabels(month_labels, rotation=30, ha='right')
    if CN_FONTPROP:
        for lbl in ax.get_xticklabels():
            lbl.set_fontproperties(CN_FONTPROP)
        for lbl in ax.get_yticklabels():
            lbl.set_fontproperties(CN_FONTPROP)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    # 数值标签（仅绝对值图）
    if not normalize:
        for i, total in enumerate(bottom):
            if CN_FONTPROP:
                ax.text(i, total + max(bottom) * 0.01, f"{int(total)}", ha='center', va='bottom', fontsize=10, fontproperties=CN_FONTPROP)
            else:
                ax.text(i, total + max(bottom) * 0.01, f"{int(total)}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    out_name = 'topic_stacked_bar_monthly_pct.png' if normalize else 'topic_stacked_bar_monthly.png'
    out_path = FIG_DIR / out_name
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ 已保存图表：{out_path}")
    return out_path


def _plot_topic_total_bar(pivot: pd.DataFrame):
    if pivot.empty:
        return None
    setup_chinese_fonts()
    totals = pivot.sum(axis=0).sort_values(ascending=False)
    topics_sorted = list(totals.index)
    palette = _get_topic_palette(list(pivot.columns))
    colors = [palette[t] for t in topics_sorted]
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x=totals.values, y=totals.index, orient='h', palette=colors)
    if CN_FONTPROP:
        ax.set_title('议题总关注度（全期）', pad=10, fontproperties=CN_FONTPROP)
        ax.set_xlabel('总关注度', fontproperties=CN_FONTPROP)
        ax.set_ylabel('议题', fontproperties=CN_FONTPROP)
    else:
        ax.set_title('议题总关注度（全期）', pad=10)
        ax.set_xlabel('总关注度')
        ax.set_ylabel('议题')
    # 数值标签
    for i, v in enumerate(totals.values):
        if CN_FONTPROP:
            ax.text(v + max(totals.values) * 0.01, i, f"{int(v)}", va='center', fontsize=11, fontproperties=CN_FONTPROP)
        else:
            ax.text(v + max(totals.values) * 0.01, i, f"{int(v)}", va='center', fontsize=11)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    # 应用中文字体到刻度
    if CN_FONTPROP:
        for lbl in ax.get_xticklabels():
            lbl.set_fontproperties(CN_FONTPROP)
        for lbl in ax.get_yticklabels():
            lbl.set_fontproperties(CN_FONTPROP)
    plt.tight_layout()
    out_path = FIG_DIR / 'topic_totals_bar.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ 已保存图表：{out_path}")
    return out_path


def _remove_old_figures():
    # 删除不需要的图表（线图、面积图、动画）
    paths = [
        FIG_DIR / 'topic_area_monthly.png',
        FIG_DIR / 'topic_timeline_monthly.png',
        FIG_DIR / 'topic_timeline_monthly_anim.gif',
    ]
    for p in paths:
        if p.exists():
            try:
                p.unlink()
                print(f"✓ 已删除：{p}")
            except Exception as e:
                print(f"⚠️ 删除失败 {p}：{e}")
    # 删除动画帧文件夹
    frame_dir = FIG_DIR / 'anim_monthly_frames'
    if frame_dir.exists():
        try:
            shutil.rmtree(frame_dir)
            print(f"✓ 已删除：{frame_dir}")
        except Exception as e:
            print(f"⚠️ 删除失败 {frame_dir}：{e}")


def save_and_plot(ts_daily: pd.DataFrame, ts_weekly: pd.DataFrame, ts_monthly: pd.DataFrame):
    # 保存 CSV（保留日/周/月）
    daily_file = STEP_DIR / 'topic_time_series_daily.csv'
    weekly_file = STEP_DIR / 'topic_time_series_weekly.csv'
    monthly_file = STEP_DIR / 'topic_time_series_monthly.csv'
    ts_daily.to_csv(daily_file, index=False, encoding='utf-8-sig')
    ts_weekly.to_csv(weekly_file, index=False, encoding='utf-8-sig')
    ts_monthly.to_csv(monthly_file, index=False, encoding='utf-8-sig')

    print(f"✓ 已保存：{daily_file}")
    print(f"✓ 已保存：{weekly_file}")
    print(f"✓ 已保存：{monthly_file}")

    # 删除旧图表
    _remove_old_figures()

    # 使用月度数据进行可视化（热力图 + 堆叠柱状图 + 总量条形图）
    pivot = _prepare_monthly_pivot(ts_monthly)
    _plot_monthly_heatmap(pivot)
    _plot_monthly_stacked_bar(pivot, normalize=False)
    _plot_monthly_stacked_bar(pivot, normalize=True)
    _plot_topic_total_bar(pivot)


def main():
    print('=' * 90)
    print('Step 3.3 - 议题动态时间变化可视化')
    print('=' * 90)

    # 1) 加载数据
    if not COMMENTS_FILE.exists() or not POSTS_FILE.exists():
        print('❌ 缺少输入文件：comments_filtered.csv 或 posts_filtered.csv')
        return
    comments = pd.read_csv(COMMENTS_FILE, encoding='utf-8-sig')
    posts = pd.read_csv(POSTS_FILE, encoding='utf-8-sig')
    print(f"✓ 评论：{len(comments)} 条，帖子：{len(posts)} 条")

    # 2) 标准化时间列
    comments = standardize_datetime(comments)
    posts = standardize_datetime(posts)

    # 3) 加载议题关键词
    topics = load_topics()
    if not topics:
        print('❌ 未找到 TOPICS，请先运行关键词提取脚本。')
        return
    print(f"✓ 议题数：{len(topics)}")

    # 4) 构建时间序列（分别对评论和帖子，然后合并）
    print('\n【构建时间序列】')
    ts_daily_c = build_time_series(comments, topics, bucket='D')
    ts_weekly_c = build_time_series(comments, topics, bucket='W')
    ts_monthly_c = build_time_series(comments, topics, bucket='M')

    ts_daily_p = build_time_series(posts, topics, bucket='D')
    ts_weekly_p = build_time_series(posts, topics, bucket='W')
    ts_monthly_p = build_time_series(posts, topics, bucket='M')

    # 合并（同维度拼接后再聚合）
    ts_daily = pd.concat([ts_daily_c, ts_daily_p], ignore_index=True)
    ts_weekly = pd.concat([ts_weekly_c, ts_weekly_p], ignore_index=True)
    ts_monthly = pd.concat([ts_monthly_c, ts_monthly_p], ignore_index=True)

    def _agg(ts: pd.DataFrame) -> pd.DataFrame:
        if ts.empty:
            return pd.DataFrame(columns=['date', 'topic', 'attention'])
        g = ts.groupby(['date', 'topic'], as_index=False)['attention'].sum()
        return g.sort_values(['date', 'topic'])

    ts_daily = _agg(ts_daily)
    ts_weekly = _agg(ts_weekly)
    ts_monthly = _agg(ts_monthly)

    print(f"✓ 日度点：{len(ts_daily)}，周度点：{len(ts_weekly)}，月度点：{len(ts_monthly)}")

    # 5) 保存与绘图
    save_and_plot(ts_daily, ts_weekly, ts_monthly)

    print('\n' + '=' * 90)
    print('✓ 完成！')
    print('=' * 90)
    print('\n【输出文件】')
    print(f"  • {STEP_DIR / 'topic_time_series_daily.csv'}")
    print(f"  • {STEP_DIR / 'topic_time_series_weekly.csv'}")
    print(f"  • {STEP_DIR / 'topic_time_series_monthly.csv'}")
    print(f"  • {FIG_DIR / 'topic_heatmap_monthly.png'}（月度热力图）")
    print(f"  • {FIG_DIR / 'topic_stacked_bar_monthly.png'}（月度堆叠柱状图）")
    print(f"  • {FIG_DIR / 'topic_stacked_bar_monthly_pct.png'}（月度百分比堆叠柱状图）")
    print(f"  • {FIG_DIR / 'topic_totals_bar.png'}（全期议题总量条形图）")


if __name__ == '__main__':
    main()
