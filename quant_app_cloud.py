import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openai import OpenAI
import os
import requests
from bs4 import BeautifulSoup
from email.utils import parsedate_to_datetime
import json

# --- 页面配置 ---
st.set_page_config(page_title="量化猎手 Pro (Cloud)", page_icon="🚀", layout="wide")

# --- CSS 美化 ---
st.markdown("""
<style>
    .metric-card { background-color: #f9f9f9; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; text-align: center; }
    .news-tag { font-size: 11px; color: #fff; background-color: #ff4757; padding: 2px 6px; border-radius: 4px; margin-right: 5px; }
    .comment-tag { font-size: 11px; color: #fff; background-color: #ffa502; padding: 2px 6px; border-radius: 4px; margin-right: 5px; }
    .hot-tag { font-size: 11px; color: #fff; background-color: #ff6b81; padding: 2px 6px; border-radius: 4px; margin-right: 5px; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 量化猎手 (Cloud Edition)")
st.caption("基于 Streamlit Cloud | 美国节点直连 | 智能舆情分析")

# --- 1. 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 控制台")
    
    # 尝试从 Streamlit Secrets 读取 API Key，如果没有则显示输入框
    default_key = st.secrets.get("GROQ_API_KEY", "")
    
    with st.expander("🔌 API 设置", expanded=not bool(default_key)):
        # 云端不需要代理，默认留空
        proxy_port = st.text_input("代理端口 (云端留空)", value="", help="本地运行填7897，云端部署请留空")
        api_key = st.text_input("AI API Key", value=default_key, type="password")
        api_base = st.text_input("AI Base URL", value="https://api.groq.com/openai/v1")
        model_name = st.text_input("模型名称", value="llama-3.3-70b-versatile")

    st.markdown("---")
    ticker = st.text_input("股票代码", value="NVDA", help="推荐美股: NVDA, TSLA | 港股: 0700.HK")
    
    # 构造代理 (仅当用户手动输入端口时生效)
    PROXIES = None
    if proxy_port:
        proxy_url = f"http://127.0.0.1:{proxy_port}"
        PROXIES = {"http": proxy_url, "https": proxy_url}
    
    # 指标参数
    with st.expander("🛠️ 指标参数"):
        ma_short = st.number_input("MA 短周期", value=5)
        ma_long = st.number_input("MA 长周期", value=20)
        boll_window = st.number_input("BOLL 周期", value=20)

    if st.button("🚀 立即分析", type="primary"):
        st.rerun()

# --- 2. 核心逻辑 (保持 V3.1 的精华) ---

# ... 指标计算函数 (保持不变) ...
def calculate_tech_indicators(df):
    if df.empty: return df
    df['MA_Short'] = df['Close'].rolling(window=int(ma_short)).mean()
    df['MA_Long'] = df['Close'].rolling(window=int(ma_long)).mean()
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = ema12 - ema26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = (df['DIF'] - df['DEA']) * 2
    
    df['BOLL_Mid'] = df['Close'].rolling(window=int(boll_window)).mean()
    df['BOLL_Std'] = df['Close'].rolling(window=int(boll_window)).std()
    df['BOLL_Upper'] = df['BOLL_Mid'] + 2 * df['BOLL_Std']
    df['BOLL_Lower'] = df['BOLL_Mid'] - 2 * df['BOLL_Std']
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = df['RSV'].ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma_tp = tp.rolling(window=14).mean()
    md = tp.rolling(window=14).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    df['CCI'] = (tp - ma_tp) / (0.015 * md)
    
    obv_change = np.where(df['Close'] > df['Close'].shift(1), df['Volume'], 
                          np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0))
    obv_change[0] = 0 
    df['OBV'] = np.cumsum(obv_change)
    return df

# ... 东方财富评论抓取 (V3版直连) ...
def get_eastmoney_comments_v3(ticker_symbol):
    east_code = ""
    try:
        if ticker_symbol.endswith(".SS") or ticker_symbol.endswith(".SZ"):
            east_code = ticker_symbol.split(".")[0]
        elif ticker_symbol.endswith(".HK"):
            raw_code = ticker_symbol.split(".")[0]
            east_code = "hk" + raw_code.zfill(5) 
        else:
            east_code = "us" + ticker_symbol
        url = f"http://guba.eastmoney.com/list,{east_code}.html"
        headers = { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36" }
        # 云端不需要代理，直接访问
        resp = requests.get(url, headers=headers, timeout=6)
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'lxml')
        comments = []
        items = soup.select(".article-h .l3 a")
        if not items: items = soup.select(".listitem .title a")
        for item in items[:10]:
            title = item.get('title') or item.text.strip()
            href = item.get('href')
            if not title or "公告" in title: continue
            if not href.startswith("http"): href = "http://guba.eastmoney.com" + href
            comments.append({'title': title, 'link': href})
        return comments
    except: return []

# ... 热榜逻辑 (API 兜底版) ...
def get_eastmoney_all_hot_fallback():
    hot_list = []
    headers = { "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1" }
    
    # 优先 API (最稳)
    try:
        api_url = "https://emappdata.eastmoney.com/stock/rank/get_hot_stock_list"
        payload = { "appId": "appId01", "globalId": "786826352926379447", "marketType": "", "pageNo": 1, "pageSize": 10 }
        resp = requests.post(api_url, json=payload, headers=headers, timeout=5)
        data = resp.json()
        if 'data' in data:
            for item in data['data']:
                hot_list.append({ 'title': f"🔥 {item.get('name')} (人气榜)", 'link': f"http://guba.eastmoney.com/list,{item.get('code')}.html" })
    except: pass
    return hot_list

# ... 整合数据获取 ...
def get_stock_data_full(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    hist_df = stock.history(period="5y") 
    try: info = stock.info
    except: info = {}
    
    # News (Google)
    news_list = []
    seven_days_ago = datetime.now() - timedelta(days=7)
    
    def get_google_news(query):
        # 注意：云端不需要 when:7d 也可以，但加上更准。
        # 重点：proxies=PROXIES 只有在 PROXIES 有值时才生效
        rss_url = f"https://news.google.com/rss/search?q={query}+when:7d&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
        try:
            resp = requests.get(rss_url, headers={"User-Agent": "Mozilla/5.0"}, proxies=PROXIES, timeout=10)
            soup = BeautifulSoup(resp.content, features="xml")
            items = soup.findAll('item')
            clean = []
            for i in items:
                try:
                    pub_dt = parsedate_to_datetime(i.pubDate.text).replace(tzinfo=None)
                    if pub_dt > seven_days_ago:
                        clean.append({'title': i.title.text, 'link': i.link.text, 'pubDate': i.pubDate.text, 'dt': pub_dt, 'source_type':'google'})
                except: continue
            return clean
        except: return []

    search_query = info.get('longName', ticker_symbol)
    news_list = get_google_news(search_query)
    
    # News (Yahoo Fallback)
    if len(news_list) < 5:
        try:
            y_news = stock.news
            for n in y_news:
                ts = n.get('providerPublishTime')
                if ts and datetime.fromtimestamp(ts) > seven_days_ago:
                    news_list.append({'title': n.get('title'), 'link': n.get('link'), 'pubDate': datetime.fromtimestamp(ts).strftime('%Y-%m-%d'), 'dt': datetime.fromtimestamp(ts), 'source_type': 'yahoo'})
        except: pass
    
    comments = get_eastmoney_comments_v3(ticker_symbol)
    hot_list = get_eastmoney_all_hot_fallback()
    
    return hist_df, info, news_list[:10], comments, hot_list

# ... 辅助函数 ...
def safe_float(val): return f"{val:.2f}" if val and isinstance(val, (int, float)) else "-"
def format_percent(num): return f"{num * 100:.2f}%" if num and isinstance(num, (int, float)) else "-"
def calculate_percentile(current_val, history_series): return (history_series < current_val).mean() * 100 if not history_series.empty else 0

def render_valuation_bar(current, history):
    pct = calculate_percentile(current, history)
    st.markdown(f"""
    <div style="font-size:12px; color:#666;">
        价格分位: <b>{pct:.1f}%</b>
        <div style="width: 100%; background: #eee; height: 6px; border-radius: 3px; margin-top:2px;">
            <div style="width: {pct}%; background: {'#2ecc71' if pct<30 else '#e74c3c'}; height: 6px; border-radius: 3px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ... 绘图 (保持不变，省略部分重复代码以节省篇幅，直接复用之前的 plot_advanced_charts) ...
def plot_advanced_charts(df, ticker, secondary_indicator):
    plot_df = df.tail(250)
    # 简单实现绘图，确保云端运行正常
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2], subplot_titles=[f'{ticker} Price', 'Volume', secondary_indicator])
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='K'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA_Short'], name='MA5'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA_Long'], name='MA20'), row=1, col=1)
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], name='Vol'), row=2, col=1)
    
    if secondary_indicator == "MACD":
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['MACD_Hist'], name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['DIF'], name='DIF'), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['DEA'], name='DEA'), row=3, col=1)
    elif secondary_indicator == "OBV":
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['OBV'], name='OBV'), row=3, col=1)
    elif secondary_indicator == "RSI":
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], name='RSI'), row=3, col=1)
        fig.add_hline(y=70, row=3, col=1); fig.add_hline(y=30, row=3, col=1)
    elif secondary_indicator == "KDJ":
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['K'], name='K'), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['D'], name='D'), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['J'], name='J'), row=3, col=1)
    elif secondary_indicator == "CCI":
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['CCI'], name='CCI'), row=3, col=1)

    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_white")
    return fig

# --- 主程序 ---
with st.spinner("☁️ 正在连接美国服务器获取全球数据..."):
    try:
        raw_df, info, news, comments, hot_list = get_stock_data_full(ticker)
    except Exception as e:
        st.error(f"Error: {e}"); st.stop()

if not raw_df.empty:
    df = calculate_tech_indicators(raw_df)
    last = df.iloc[-1]
    
    with st.container():
        c1, c2, c3, c4, c5 = st.columns([1.5, 1, 1, 1, 1.5])
        c1.metric(f"{info.get('longName', ticker)}", f"{last['Close']:.2f}")
        c2.metric("PE", f"{safe_float(info.get('trailingPE'))}")
        c3.metric("PB", safe_float(info.get('priceToBook')))
        c4.metric("Div", format_percent(info.get('dividendYield')))
        with c5: render_valuation_bar(last['Close'], df['Close'])

    st.divider()
    
    # Chart
    col_sel, _ = st.columns([1, 4])
    with col_sel:
        opt = st.selectbox("Indicator", ["MACD", "KDJ", "RSI", "CCI", "OBV"], label_visibility="collapsed")
    st.plotly_chart(plot_advanced_charts(df, ticker, opt), use_container_width=True)
    
    # Tabs
    t1, t2, t3, t4 = st.tabs(["🤖 AI Report", "📰 News", "💬 Comments", "🔥 Hot"])
    
    with t1:
        if st.button("Generate Report", type="primary"):
            if not api_key: st.error("No API Key")
            else:
                prompt = f"Analyze {ticker}. Close:{last['Close']:.2f}, RSI:{last['RSI']:.2f}. News:{str([n['title'] for n in news[:3]])}. Comments:{str([c['title'] for c in comments[:5]])}. Give investment advice."
                client = OpenAI(api_key=api_key, base_url=api_base)
                resp = client.chat.completions.create(model=model_name, messages=[{"role":"user","content":prompt}])
                st.info(resp.choices[0].message.content)

    with t2:
        for n in news: st.markdown(f"[{n['title']}]({n['link']})"); st.divider()
    with t3:
        for c in comments: st.markdown(f"[{c['title']}]({c['link']})"); st.divider()
    with t4:
        for h in hot_list: st.markdown(f"[{h['title']}]({h['link']})"); st.divider()

else: st.warning("Waiting for input...")