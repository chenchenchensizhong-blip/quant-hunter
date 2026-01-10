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
st.set_page_config(page_title="量化猎手 Pro (云端版)", page_icon="🚀", layout="wide")

# --- CSS 美化 ---
st.markdown("""
<style>
    .metric-card { background-color: #f9f9f9; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; text-align: center; }
    .news-tag { font-size: 11px; color: #fff; background-color: #ff4757; padding: 2px 6px; border-radius: 4px; margin-right: 5px; }
    .comment-tag { font-size: 11px; color: #fff; background-color: #ffa502; padding: 2px 6px; border-radius: 4px; margin-right: 5px; }
    .hot-tag { font-size: 11px; color: #fff; background-color: #ff6b81; padding: 2px 6px; border-radius: 4px; margin-right: 5px; }
    /* 调整 Tab 字体 */
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 量化猎手 (云端版)")
st.caption("基于 Streamlit Cloud | 美国节点直连 | 智能舆情分析")

# --- 1. 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 控制台")
    
    # 尝试从 Streamlit Secrets 读取 API Key
    default_key = st.secrets.get("GROQ_API_KEY", "")
    
    with st.expander("🔌 API 设置", expanded=not bool(default_key)):
        api_key = st.text_input("AI API Key", value=default_key, type="password", help="请输入 Groq 或其他兼容 OpenAI 的 Key")
        api_base = st.text_input("AI Base URL", value="https://api.groq.com/openai/v1")
        model_name = st.text_input("模型名称", value="llama-3.3-70b-versatile")

    st.markdown("---")
    ticker = st.text_input("股票代码", value="NVDA", help="推荐美股: NVDA, TSLA | 港股: 0700.HK | A股: 600519.SS")
    
    # 云端不需要代理设置，直接隐藏或移除
    
    # 指标参数
    with st.expander("🛠️ 指标参数"):
        ma_short = st.number_input("MA 短周期", value=5)
        ma_long = st.number_input("MA 长周期", value=20)
        boll_window = st.number_input("BOLL 周期", value=20)

    if st.button("🚀 立即分析", type="primary"):
        st.rerun()

# --- 2. 核心逻辑 ---

# ... 指标计算函数 ...
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

# ... 东方财富评论抓取 (云端版：尝试直连 API，失败则网页) ...
def get_eastmoney_comments_cloud(ticker_symbol):
    east_code = ""
    try:
        if ticker_symbol.endswith(".SS") or ticker_symbol.endswith(".SZ"):
            east_code = ticker_symbol.split(".")[0]
        elif ticker_symbol.endswith(".HK"):
            raw_code = ticker_symbol.split(".")[0]
            east_code = "hk" + raw_code.zfill(5) 
        else:
            east_code = "us" + ticker_symbol
            
        # 优先尝试 HTML 抓取 (通常内容更全)
        url = f"http://guba.eastmoney.com/list,{east_code}.html"
        headers = { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36" }
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

# ... 全站热榜 (云端修复版：API 优先) ...
def get_eastmoney_all_hot_cloud():
    hot_list = []
    # 模拟手机 User-Agent
    headers = { "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1" }
    
    # === 策略更改：优先使用 API ===
    # 原因：Streamlit 服务器在美国，直接抓取东方财富 HTML 网页极易被识别为爬虫并返回空白/验证码。
    # API 返回的是纯 JSON 数据，对 IP 限制通常较宽。
    try:
        # 东方财富个股人气榜 API
        api_url = "https://emappdata.eastmoney.com/stock/rank/get_hot_stock_list"
        payload = {
            "appId": "appId01", 
            "globalId": "786826352926379447", 
            "marketType": "", 
            "pageNo": 1, 
            "pageSize": 12
        }
        # POST 请求
        resp = requests.post(api_url, json=payload, headers=headers, timeout=5)
        data = resp.json()
        
        if 'data' in data:
            for item in data['data']:
                name = item.get('name')
                code = item.get('code')
                # 构造链接
                link = f"http://guba.eastmoney.com/list,{code}.html"
                hot_list.append({
                    'title': f"🔥 {name} (全网人气飙升)", 
                    'link': link
                })
    except Exception as e:
        print(f"API Failed: {e}")

    # 如果 API 失败，才尝试备用的网页抓取 (虽然在云端概率较低)
    if not hot_list:
        try:
            url = "http://mguba.eastmoney.com/"
            resp = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(resp.text, 'lxml')
            items = soup.find_all('a')
            for item in items:
                title = item.text.strip()
                link = item.get('href')
                if len(title) < 4 or not link: continue
                if "注册" in title or "下载" in title: continue
                if not link.startswith("http"): link = "http://mguba.eastmoney.com" + link
                if any(h['title'] == title for h in hot_list): continue
                hot_list.append({'title': title, 'link': link})
                if len(hot_list) >= 10: break
        except: pass
        
    return hot_list[:10]

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
        # 强制中文搜索
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
        try:
            # 云端不需要代理
            resp = requests.get(rss_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
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
    
    comments = get_eastmoney_comments_cloud(ticker_symbol)
    hot_list = get_eastmoney_all_hot_cloud()
    
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

# ... 绘图 (带中文标题) ...
def plot_advanced_charts(df, ticker, secondary_indicator):
    plot_df = df.tail(250)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2], 
                        subplot_titles=[f'{ticker} 股价趋势', '成交量', secondary_indicator])
    
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='K线'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA_Short'], name=f'MA{int(ma_short)}'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA_Long'], name=f'MA{int(ma_long)}'), row=1, col=1)
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], name='成交量'), row=2, col=1)
    
    if secondary_indicator == "MACD":
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['MACD_Hist'], name='MACD柱'), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['DIF'], name='DIF'), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['DEA'], name='DEA'), row=3, col=1)
    elif secondary_indicator == "OBV":
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['OBV'], name='OBV能量潮'), row=3, col=1)
    elif secondary_indicator == "RSI":
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    elif secondary_indicator == "KDJ":
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['K'], name='K'), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['D'], name='D'), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['J'], name='J'), row=3, col=1)
    elif secondary_indicator == "CCI":
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['CCI'], name='CCI'), row=3, col=1)

    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_white", hovermode="x unified")
    return fig

# --- 主程序 ---
with st.spinner("☁️ 正在连接全球金融节点获取数据..."):
    try:
        raw_df, info, news, comments, hot_list = get_stock_data_full(ticker)
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        st.stop()

if not raw_df.empty:
    df = calculate_tech_indicators(raw_df)
    last = df.iloc[-1]
    
    with st.container():
        c1, c2, c3, c4, c5 = st.columns([1.5, 1, 1, 1, 1.5])
        c1.metric(f"{info.get('longName', ticker)}", f"{last['Close']:.2f}")
        c2.metric("市盈率 PE", f"{safe_float(info.get('trailingPE'))}")
        c3.metric("市净率 PB", safe_float(info.get('priceToBook')))
        c4.metric("股息率 Div", format_percent(info.get('dividendYield')))
        with c5: render_valuation_bar(last['Close'], df['Close'])

    st.divider()
    
    # Chart
    col_sel, _ = st.columns([1, 4])
    with col_sel:
        opt = st.selectbox("选择副图指标", ["MACD", "KDJ", "RSI", "CCI", "OBV"], label_visibility="collapsed")
    st.plotly_chart(plot_advanced_charts(df, ticker, opt), use_container_width=True)
    
    # Tabs
    t1, t2, t3, t4 = st.tabs(["🤖 AI 研报", "📰 新闻资讯", "💬 股吧热评", "🔥 全网热榜"])
    
    with t1:
        if st.button("生成深度分析报告", type="primary"):
            if not api_key: st.error("请先在左侧配置 API Key")
            else:
                prompt = f"""
                请作为一位资深的华尔街与A股双栖基金经理，分析股票 {ticker}。
                
                【技术面数据】
                最新价: {last['Close']:.2f}
                RSI指标: {last['RSI']:.2f} (强弱参考)
                
                【基本面数据】
                PE市盈率: {safe_float(info.get('trailingPE'))}
                PB市净率: {safe_float(info.get('priceToBook'))}
                
                【舆情面】
                最新新闻: {str([n['title'] for n in news[:3]])}
                散户热评: {str([c['title'] for c in comments[:5]])}
                
                请用**中文**生成一份简报：
                1. **多空博弈分析**：机构观点与散户情绪是否对立？
                2. **技术形态诊断**：是否存在背离或买卖信号？
                3. **操作建议**：激进型与稳健型投资者的策略。
                """
                client = OpenAI(api_key=api_key, base_url=api_base)
                with st.spinner("AI 正在撰写中文研报..."):
                    resp = client.chat.completions.create(model=model_name, messages=[{"role":"user","content":prompt}])
                    st.info(resp.choices[0].message.content)

    with t2:
        for n in news: 
            st.markdown(f"[{n['title']}]({n['link']})")
            st.caption(f"来源: {n.get('source_type', 'Web')} | 时间: {n.get('pubDate', '')}")
            st.divider()
    with t3:
        if comments:
            for c in comments: st.markdown(f"[{c['title']}]({c['link']})"); st.divider()
        else: st.info("暂无评论数据")
    with t4:
        if hot_list:
            st.caption("来源：东方财富全网人气榜 (API直连)")
            for i, h in enumerate(hot_list):
                st.markdown(f"""
                <div style="margin-bottom: 8px;">
                    <span class="hot-tag">TOP {i+1}</span>
                    <a href="{h.get('link')}" target="_blank" style="text-decoration:none; color:#333; font-weight:bold;">{h.get('title')}</a>
                </div>
                """, unsafe_allow_html=True)
                st.divider()
        else: st.info("热榜数据获取超时，可能受云端网络限制。")

else: st.warning("请在左侧输入股票代码并点击运行。")