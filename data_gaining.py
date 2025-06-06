import random
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

# 计时装饰器
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"执行时间: {end - start:.2f}秒")
        return result
    return wrapper

# 随机User-Agent列表
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 5.1; U; en; rv:1.8.1)Gecko/20061208 Firefox/2.0.0 Opera 9.50",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_1)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; .NET4.0C; .NET4.0E; .NET CLR 2.0.50727; .NET CLR 3.0.30729; .NET CLR 3.5.30729; InfoPath.3; rv:11.0) like Gecko",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)",
    "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
    "Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
    "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11",
    "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; TencentTraveler 4.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; The World)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SE 2.X MetaSr 1.0; SE 2.X MetaSr 1.0; .NET CLR 2.0.50727; SE 2.X MetaSr 1.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Avant Browser)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)"
]

def get_headers(cookie=None):
    """获取请求头"""
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': "https://movie.douban.com/",
        'Accept-Language': "zh-CN,zh;q=0.9,en;q=0.8",
    }
    if cookie:
        headers['Cookie'] = cookie
    return headers



def fetch_comments(movie_id, comment_type, num_pages=20, cookie=None):
    """获取指定电影和类型的评论"""
    comments = []
    headers = get_headers(cookie)
    
    for page in range(num_pages):
        url = f"https://movie.douban.com/subject/{movie_id}/comments?percent_type={comment_type}&start={page*20}&limit=20&status=P&sort=new_score"
        print(f"正在获取: {url}")
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for span in soup.select(".comment-content p"):
                comments.append(span.get_text(strip=True))
            
            # 随机延迟防止被封
            time.sleep(random.uniform(0.5, 1))
            
        except Exception as e:
            print(f"获取评论出错: {e}")
            continue
    
    return comments

def save_to_csv(comments, comment_type_counts, filename):
    """将评论保存为CSV文件"""
    # 创建评论类型标签
    types = []
    for t, count in zip(["好评", "差评", "一般"], comment_type_counts):
        types.extend([t] * count)
    
    # 确保评论数量和类型数量一致
    total_comments = sum(comment_type_counts)
    if len(comments) != total_comments:
        comments = comments[:total_comments]  # 截断多余评论
    
    df = pd.DataFrame({
        "Comment": comments,
        "Type": types[:len(comments)]  # 确保长度一致
    })
    
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"已保存到: {filename}")

@timeit
def spider():
    # 读取cookie
    try:
        with open("cookie.txt", "r") as f:
            cookie = f.read().strip()
    except FileNotFoundError:
        cookie = None
        print("未找到cookie.txt文件，将以无cookie方式运行")

    # 电影1: 复仇者联盟4 (ID: 26100958)
    print("\n开始爬取复仇者联盟4评论...")
    avengers_high = fetch_comments("26100958", "h", 20, cookie)  # 好评
    avengers_low = fetch_comments("26100958", "l", 20, cookie)   # 差评
    avengers_median = fetch_comments("26100958", "m", 20, cookie) # 一般
    
    save_to_csv(
        avengers_high + avengers_low + avengers_median,
        [len(avengers_high), len(avengers_low), len(avengers_median)],
        "Avengers-Endgame_comments.csv"
    )

    # 电影2: 雷霆特工队 (ID: 35927475)
    print("\n开始爬取雷霆特工队评论...")
    thunder_high = fetch_comments("35927475", "h", 20, cookie)  # 好评
    thunder_low = fetch_comments("35927475", "l", 20, cookie)   # 差评
    thunder_median = fetch_comments("35927475", "m", 20, cookie) # 一般
    
    save_to_csv(
        thunder_high + thunder_low + thunder_median,
        [len(thunder_high), len(thunder_low), len(thunder_median)],
        "Thunderbolts_comments.csv"
    )

if __name__ == "__main__":
    spider()
