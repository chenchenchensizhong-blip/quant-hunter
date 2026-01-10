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
st.set_page_config(page_title="量化猎手 V5.0 (基本面深钻版)", page_icon="💎", layout="wide")

st.markdown("""
<style>
    .metric-card { background-color: #f9f9f9; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; text-align: center; }
    .news-tag { font-size: 11px; color: #fff; background-color: #ff4757; padding: 2px 6px; border-radius: 4px; margin-right: 5px; }
    .comment-tag { font-size: 11px; color: #fff; background-color: #ffa502; padding: 2px 6px; border-radius: 4px; margin-right: 5px; }
    .hot-tag { font-size: 11px; color: #fff; background-color: #ff6b81; padding: 2px 6px; border-radius: 4px; margin-right: 5px; }
    /* 紧凑型 Metric */
    div[data-testid="stMetricValue"] { font-size: 18px; }
    div[data-testid="stMetricLabel"] { font-size: 12px; color: #666; }
</style>
""", unsafe_allow_html=True)

st.title("💎 量化猎手 V5.0 (基本面深钻版)")
st.caption("Streamlit Cloud | 20+ 财务维度 | 机构目标价 | AI 杜邦分析")

# --- 1. 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 控制台")
    default_key = st.secrets.get("GROQ_API_KEY", "")
    with st.expander("🔌 API 设置", expanded=not bool(default_key)):
        api_key = st.text_input("AI API Key", value=default_key, type="password")
        api_base = st.text_input("AI Base URL", value="https://api.groq.com/openai/v1")
        model_name = st.text_input("模型名称", value="llama-3.3-70b-versatile")

    st.markdown("---")
    ticker = st.text_input("股票代码", value="NVDA", help="美股: NVDA | 港股: 0700.HK | A股: 600519.SS")
    
    with st.expander("🛠️ 指标参数"):
        ma_short = st.number_input("MA 短周期", value=5)
        ma_long = st.number_input("MA 长周期", value=20)
        boll_window = st.number_input("BOLL 周期", value=20)

    if st.button("🚀 深度扫描", type="primary"):
        st.rerun()

# --- 2. 核心逻辑 ---

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

# --- 数据获取模块 ---
def get_eastmoney_comments_cloud(ticker_symbol):
    """ 获取股吧评论 (HTML直连) """
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

def get_eastmoney_all_hot_cloud():
    """ 获取全站热榜 (API优先) """
    hot_list = []
    headers = { "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1" }
    try:
        api_url = "https://emappdata.eastmoney.com/stock/rank/get_hot_stock_list"
        payload = { "appId": "appId01", "globalId": "786826352926379447", "marketType": "", "pageNo": 1, "pageSize": 12 }
        resp = requests.post(api_url, json=payload, headers=headers, timeout=5)
        data = resp.json()
        if 'data' in data:
            for item in data['data']:
                hot_list.append({ 'title': f"🔥 {item.get('name')} (全网人气飙升)", 'link': f"http://guba.eastmoney.com/list,{item.get('code')}.html" })
    except: pass
    return hot_list[:10]

def get_stock_data_full(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    hist_df = stock.history(period="5y") 
    try: info = stock.info
    except: info = {}
    
    # Google News
    news_list = []
    seven_days_ago = datetime.now() - timedelta(days=7)
    try:
        rss_url = f"https://news.google.com/rss/search?q={info.get('longName', ticker_symbol)}+when:7d&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
        resp = requests.get(rss_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(resp.content, features="xml")
        items = soup.findAll('item')
        for i in items:
            try:
                pub_dt = parsedate_to_datetime(i.pubDate.text).replace(tzinfo=None)
                if pub_dt > seven_days_ago:
                    news_list.append({'title': i.title.text, 'link': i.link.text, 'pubDate': i.pubDate.text, 'dt': pub_dt, 'source_type':'google'})
            except: continue
    except: pass
    
    comments = get_eastmoney_comments_cloud(ticker_symbol)
    hot_list = get_eastmoney_all_hot_cloud()
    return hist_df, info, news_list[:10], comments, hot_list

# --- 辅助函数 ---
def safe_float(val): return f"{val:.2f}" if val and isinstance(val, (int, float)) else "-"
def format_percent(num): return f"{num * 100:.2f}%" if num and isinstance(num, (int, float)) else "-"
def format_large(num):
    if not num: return "-"
    if num > 1e12: return f"{num/1e12:.2f}T"
    if num > 1e9: return f"{num/1e9:.2f}B"
    if num > 1e6: return f"{num/1e6:.2f}M"
    return str(num)

def calculate_percentile(current_val, history_series): return (history_series < current_val).mean() * 100 if not history_series.empty else 0

def render_valuation_bar(current, history):
    pct = calculate_percentile(current, history)
    st.markdown(f"""
    <div style="font-size:12px; color:#666;">
        十年价格分位: <b>{pct:.1f}%</b>
        <div style="width: 100%; background: #eee; height: 6px; border-radius: 3px; margin-top:2px;">
            <div style="width: {pct}%; background: {'#2ecc71' if pct<30 else '#e74c3c'}; height: 6px; border-radius: 3px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ... 绘图 (保持精简) ...
def plot_advanced_charts(df, ticker, secondary_indicator):
    plot_df = df.tail(250)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2], 
                        subplot_titles=[f'{ticker} 价格趋势', '成交量', secondary_indicator])
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='K线'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA_Short'], name='MA5'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA_Long'], name='MA20'), row=1, col=1)
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], name='Vol'), row=2, col=1)
    
    if secondary_indicator == "MACD":
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['MACD_Hist'], name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['DIF'], name='DIF'), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['DEA'], name='DEA'), row=3, col=1)
    elif secondary_indicator == "OBV": fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['OBV'], name='OBV'), row=3, col=1)
    elif secondary_indicator == "RSI": fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], name='RSI'), row=3, col=1); fig.add_hline(y=70, row=3, col=1); fig.add_hline(y=30, row=3, col=1)
    elif secondary_indicator == "KDJ": fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['K'], name='K'), row=3, col=1); fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['D'], name='D'), row=3, col=1); fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['J'], name='J'), row=3, col=1)
    elif secondary_indicator == "CCI": fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['CCI'], name='CCI'), row=3, col=1)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_white")
    return fig

# --- 主程序 ---
with st.spinner("💎 正在挖掘深度财务数据..."):
    try:
        raw_df, info, news, comments, hot_list = get_stock_data_full(ticker)
    except Exception as e:
        st.error(f"Error: {e}"); st.stop()

if not raw_df.empty:
    df = calculate_tech_indicators(raw_df)
    last = df.iloc[-1]
    
    # === 顶部概览 ===
    with st.container():
        c1, c2, c3, c4, c5 = st.columns([1.5, 1, 1, 1, 1.5])
        c1.metric(f"{info.get('longName', ticker)}", f"{last['Close']:.2f}")
        c2.metric("PE (TTM)", f"{safe_float(info.get('trailingPE'))}")
        c3.metric("机构目标价", safe_float(info.get('targetMeanPrice')))
        c4.metric("推荐评级", info.get('recommendationKey', '-').upper())
        with c5: render_valuation_bar(last['Close'], df['Close'])

    st.divider()
    
    # Chart
    col_sel, _ = st.columns([1, 4])
    with col_sel: opt = st.selectbox("副图指标", ["MACD", "KDJ", "RSI", "CCI", "OBV"], label_visibility="collapsed")
    st.plotly_chart(plot_advanced_charts(df, ticker, opt), use_container_width=True)
    
    # === 深度基本面数据 (20+ 维度) ===
    with st.expander("📊 深度财务透视 (Valuation / Growth / Cash / Debt)", expanded=True):
        t_fund1, t_fund2, t_fund3, t_fund4 = st.tabs(["💰 估值与回报", "🚀 成长与盈利", "🛡️ 负债与现金流", "📅 股息与机构"])
        
        with t_fund1: # 估值
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("企业价值/EBITDA", safe_float(info.get('enterpriseToEbitda')), help="比PE更纯粹的估值指标，排除债务影响")
            c2.metric("市销率 (P/S)", safe_float(info.get('priceToSalesTrailing12Months')), help="适合亏损但有营收的成长股")
            c3.metric("PEG Ratio", safe_float(info.get('pegRatio')), help="<1 通常视为低估")
            c4.metric("ROE (净资产收益率)", format_percent(info.get('returnOnEquity')), help="巴菲特核心指标")
            
        with t_fund2: # 成长
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("季度营收增长 (YoY)", format_percent(info.get('revenueGrowth')))
            c2.metric("季度盈利增长 (YoY)", format_percent(info.get('earningsGrowth')))
            c3.metric("毛利率", format_percent(info.get('grossMargins')))
            c4.metric("净利率", format_percent(info.get('profitMargins')))
            
        with t_fund3: # 健康
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("自由现金流 (FCF)", format_large(info.get('freeCashflow')), help="公司的真实印钞能力")
            c2.metric("总现金", format_large(info.get('totalCash')))
            c3.metric("总负债", format_large(info.get('totalDebt')))
            c4.metric("流动比率", safe_float(info.get('currentRatio')), help=">1.5 为健康，短期偿债能力")
            
        with t_fund4: # 股东
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("股息率", format_percent(info.get('dividendYield')))
            c2.metric("派息比率", format_percent(info.get('payoutRatio')), help="利润中多少分给了股东")
            c3.metric("做空比例", format_percent(info.get('shortPercentOfFloat')), help="太高说明被机构看空")
            c4.metric("机构持仓", format_percent(info.get('heldPercentInstitutions')))

    # === AI 分析区 ===
    t1, t2, t3, t4 = st.tabs(["🤖 AI 深度投研", "📰 新闻资讯", "💬 股吧热评", "🔥 全网热榜"])
    
    with t1:
        if st.button("生成 AI 深度分析报告", type="primary"):
            if not api_key: st.error("请先在左侧配置 API Key")
            else:
                # 构造超级详细的 Prompt
                fund_str = f"""
                估值: PE={safe_float(info.get('trailingPE'))}, PEG={safe_float(info.get('pegRatio'))}, PS={safe_float(info.get('priceToSalesTrailing12Months'))}
                盈利: ROE={format_percent(info.get('returnOnEquity'))}, 毛利率={format_percent(info.get('grossMargins'))}
                成长: 营收增长={format_percent(info.get('revenueGrowth'))}, 盈利增长={format_percent(info.get('earningsGrowth'))}
                现金流: FCF={format_large(info.get('freeCashflow'))}, 现金={format_large(info.get('totalCash'))}, 负债={format_large(info.get('totalDebt'))}
                机构: 目标价={safe_float(info.get('targetMeanPrice'))}, 评级={info.get('recommendationKey')}
                """
                
                tech_str = f"Close:{last['Close']:.2f}, RSI:{last['RSI']:.2f}, MA20:{last['MA_Long']:.2f}, OBV趋势:{'上升' if last['OBV']>df.iloc[-5]['OBV'] else '下降'}"
                
                news_summary = str([n['title'] for n in news[:3]])
                
                prompt = f"""
                你是一位华尔街顶级基金经理。请基于以下全维度数据，对 {info.get('longName', ticker)} 进行深度投资价值分析。
                
                【硬核财务数据】
                {fund_str}
                
                【技术面信号】
                {tech_str}
                
                【近期舆情】
                {news_summary}
                
                请用**中文**生成一份严谨的研报，包含：
                1. **杜邦分析视角**：公司的ROE是由高利润率驱动，还是高杠杆驱动？(结合毛利率和负债分析)
                2. **安全边际**：当前估值(PE/PEG)配合现金流情况，股价是否便宜？
                3. **风险预警**：是否存在增收不增利、现金流枯竭或债务过高等红灯信号？
                4. **最终操作建议**：(买入/持有/卖出)及理由。
                """
                
                client = OpenAI(api_key=api_key, base_url=api_base)
                with st.spinner("AI 正在进行杜邦分析与风险测算..."):
                    resp = client.chat.completions.create(model=model_name, messages=[{"role":"user","content":prompt}])
                    st.markdown(f"""
                    <div style='background-color:#f8f9fa; padding:20px; border-radius:10px; border-left: 5px solid #4b7bec;'>
                        {resp.choices[0].message.content}
                    </div>
                    """, unsafe_allow_html=True)

    with t2:
        for n in news: 
            st.markdown(f"[{n['title']}]({n['link']})")
            st.caption(f"{n.get('source_type', 'Web')} | {n.get('pubDate', '')}")
            st.divider()
    with t3:
        for c in comments: st.markdown(f"[{c['title']}]({c['link']})"); st.divider()
    with t4:
        for h in hot_list: st.markdown(f"[{h['title']}]({h['link']})"); st.divider()

else: st.info("👈 请在左侧输入代码，例如 NVDA 或 0700.HK，然后点击'深度扫描'")