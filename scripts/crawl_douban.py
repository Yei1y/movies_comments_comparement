# -*- coding: utf-8 -*-
"""豆瓣短评爬虫：按"好评 / 差评 / 一般"三类各抓取 20 页（每页 20 条）。

注意
----
1. 豆瓣对未登录访问有频次限制，实际抓取需要自带有效 Cookie。
   本脚本从项目根目录的 ``.douban_cookie`` 文件（不入版本库）或环境变量
   ``DOUBAN_COOKIE`` 读取，两者都没有时以匿名方式尝试（易被限流）。
2. 抓取结果写入 ``data/``，文件名与后续分析脚本约定的输入一致。
3. 豆瓣的 ``percent_type`` 参数（h/l/m 对应好评/差评/一般）并非始终生效：
   当某部电影的某一评价档位样本很少时，接口可能返回与相邻档位相同的评论，
   从而产生"标签错误"的数据。本项目在 ``data/`` 中的雷霆特工队数据即存在
   该问题（"一般"与"差评"完全重合），详见 README 的"已知问题"一节。
   因此重新抓取后务必做一次标签--文本一致性检查（见 ``check_label_integrity``）。
"""
from pathlib import Path
import os
import random
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
COOKIE_FILE = ROOT / '.douban_cookie'

# 影片与豆瓣条目 ID
MOVIES = {
    'av': {'id': '26100958', 'title': '复仇者联盟4：终局之战',
           'file': 'Avengers-Endgame_comments.csv'},
    'th': {'id': '35927475', 'title': '雷霆特工队*',
           'file': 'Thunderbolts_comments.csv'},
}

# percent_type 取值 -> 中文标签
PERCENT_TYPES = [('h', '好评'), ('l', '差评'), ('m', '一般')]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko",
]


def timeit(func):
    """打印函数执行耗时的装饰器。"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"执行时间: {time.time() - start:.2f}秒")
        return result
    return wrapper


def get_cookie():
    """按 环境变量 -> 本地文件 的顺序读取 Cookie，缺省返回 None。"""
    cookie = os.environ.get('DOUBAN_COOKIE')
    if cookie:
        return cookie.strip()
    if COOKIE_FILE.exists():
        return COOKIE_FILE.read_text(encoding='utf-8').strip()
    print("未找到 Cookie（.douban_cookie 或 $DOUBAN_COOKIE），将以匿名方式尝试抓取")
    return None


def get_headers(cookie=None):
    """构造带随机 UA 的请求头。"""
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': "https://movie.douban.com/",
        'Accept-Language': "zh-CN,zh;q=0.9,en;q=0.8",
    }
    if cookie:
        headers['Cookie'] = cookie
    return headers


def fetch_comments(movie_id, percent_type, num_pages=20, cookie=None):
    """抓取指定影片在某一评价档位下的短评文本。"""
    comments = []
    for page in range(num_pages):
        url = (f"https://movie.douban.com/subject/{movie_id}/comments"
               f"?percent_type={percent_type}&start={page * 20}&limit=20"
               f"&status=P&sort=new_score")
        print(f"正在获取: {url}")
        try:
            response = requests.get(url, headers=get_headers(cookie), timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            comments.extend(s.get_text(strip=True) for s in soup.select(".comment-content p"))
            time.sleep(random.uniform(0.5, 1))  # 随机延迟，降低被封风险
        except Exception as exc:
            print(f"获取评论出错: {exc}")
            continue
    return comments


def check_label_integrity(df):
    """检查不同标签之间是否存在完全重合的文本（抓取档位失效的信号）。"""
    sets = {t: set(df.loc[df['Type'] == t, 'Comment'].dropna()) for t in df['Type'].unique()}
    labels = list(sets)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = sets[labels[i]], sets[labels[j]]
            union = a | b
            if union and len(a & b) / len(union) > 0.9:
                print(f"  [警告] {labels[i]} 与 {labels[j]} 的评论几乎完全重合 "
                      f"(Jaccard={len(a & b) / len(union):.3f})，该档位抓取可能失效")


@timeit
def spider():
    """抓取全部影片的三档短评并写入 data/。"""
    cookie = get_cookie()

    for key, movie in MOVIES.items():
        print(f"\n开始爬取《{movie['title']}》评论...")
        frames = []
        for percent_type, label in PERCENT_TYPES:
            texts = fetch_comments(movie['id'], percent_type, 20, cookie)
            frames.append(pd.DataFrame({'Comment': texts, 'Type': label}))
            print(f"  {label}: 抓取 {len(texts)} 条")

        df = pd.concat(frames, ignore_index=True)
        out_path = DATA_DIR / movie['file']
        df.to_csv(out_path, index=False, encoding='utf-8')
        print(f"已保存到: {out_path}")
        check_label_integrity(df)


if __name__ == "__main__":
    spider()