import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import json
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 1. 定義家族密碼
FAMILY_PASSWORD = "26283188" # 您可以自己改成喜歡的數字或英文

# 2. 建立密碼檢查機制
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("### 🔒 家族專屬戰情室")
    pwd_input = st.text_input("請輸入通關密碼：", type="password")
    if st.button("解鎖進入"):
        if pwd_input == FAMILY_PASSWORD:
            st.session_state.authenticated = True
            st.rerun() # 密碼正確，重新載入畫面
        else:
            st.error("❌ 密碼錯誤，請重新輸入！")
    st.stop() # 密碼如果沒過，程式就在這裡停止，不執行下方的看盤畫布


# ==========================================
# --- 網頁基礎設定 ---
# ==========================================
st.set_page_config(page_title="專業多重技術分析與選股系統", layout="wide")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

INDUSTRY_PREFIXES = {
    "11": "水泥工業", "12": "食品工業", "13": "塑膠工業", "14": "紡織纖維",
    "15": "電機機械", "16": "電器電纜", "17": "化學工業", "18": "玻璃陶瓷",
    "19": "造紙工業", "20": "鋼鐵工業", "21": "橡膠工業", "22": "汽車工業",
    "23": "電子工業 (含半導體)", "24": "電子零組件", "25": "建材營造", "26": "航運業",
    "27": "觀光事業", "28": "金融保險業", "29": "貿易百貨", "30": "電腦及週邊設備",
    "31": "光電業", "32": "通信網路業", "33": "資訊服務業", "34": "其他電子業",
    "35": "電子通路業", "36": "資訊服務業", "37": "體育及休閒", "41": "生技醫療業",
    "43": "生技醫療業", "44": "紡織纖維", "45": "電機機械", "47": "化學工業",
    "49": "通信網路業", "50": "電腦及週邊設備", "52": "貿易百貨", "53": "半導體業",
    "54": "電子零組件", "55": "觀光事業", "56": "生技醫療業", "58": "金融保險業", 
    "60": "金融保險業", "61": "電子通路業", "62": "通信網路業", "64": "生技醫療業", 
    "65": "生技醫療業", "66": "半導體業", "68": "生技醫療業", "80": "電腦及週邊設備", 
    "81": "光電業", "82": "電子零組件", "83": "生技醫療業", "84": "生技醫療業",
    "89": "其他", "99": "其他 (雜項)"
}

def get_industry(ticker_str):
    code = str(ticker_str).split('.')[0].strip()
    if not code.isdigit() or len(code) != 4:
        return "ETF / 權證 / 外國企業"
    prefix = code[:2]
    return INDUSTRY_PREFIXES.get(prefix, "其他 / 特殊類股")

@st.cache_data(ttl=3600)
def get_tw_stock_list():
    tickers = []
    try:
        twse_url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        twse_res = requests.get(twse_url, headers=headers, timeout=15, verify=False)
        if twse_res.status_code == 200:
            for item in twse_res.json():
                if len(item['Code']) == 4 and item['Code'].isdigit():
                    tickers.append(f"{item['Code']}.TW - {item['Name']}")
        tpex_url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
        tpex_res = requests.get(tpex_url, headers=headers, timeout=15, verify=False)
        if tpex_res.status_code == 200:
            for item in tpex_res.json():
                if len(item['SecuritiesCompanyCode']) == 4 and item['SecuritiesCompanyCode'].isdigit():
                    tickers.append(f"{item['SecuritiesCompanyCode']}.TWO - {item['CompanyName']}")
        return tickers
    except Exception:
        return []

# ==========================================
# --- 核心技術指標運算 ---
# ==========================================
def calculate_indicators(df):
    if len(df) == 0: return df
    df['Real_Open'] = df['Open'].copy()
    df['Real_High'] = df['High'].copy()
    df['Real_Low'] = df['Low'].copy()
    df['Real_Close'] = df['Close'].copy()

    df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df['HA_Open'] = df['Open'].copy()
    for i in range(1, len(df)):
        df.iloc[i, df.columns.get_loc('HA_Open')] = (df.iloc[i-1, df.columns.get_loc('HA_Open')] + df.iloc[i-1, df.columns.get_loc('HA_Close')]) / 2
    df['HA_High'] = df[['High', 'HA_Open', 'HA_Close']].max(axis=1)
    df['HA_Low'] = df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)

    df['SMA_20'] = df['Real_Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Real_Close'].rolling(window=60).mean()
    df['EMA_12'] = df['Real_Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Real_Close'].ewm(span=26, adjust=False).mean()

    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']

    delta = df['Real_Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    low_min = df['Real_Low'].rolling(window=9).min()
    high_max = df['Real_High'].rolling(window=9).max()
    df['RSV'] = (df['Real_Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = df['RSV'].ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()

    df['Score'] = (
        (df['Real_Close'] > df['SMA_60']).astype(int) +
        (df['Real_Close'] > df['SMA_20']).astype(int) +
        (df['Real_Close'] > df['EMA_12']).astype(int) +
        (df['K'] > df['D']).astype(int) +
        (df['RSI_14'] > 50).astype(int) +
        (df['MACD_Histogram'] > 0).astype(int) +
        (df['HA_Close'] > df['HA_Open']).astype(int)
    )
    df['First_Resonance'] = (df['Score'] == 7) & (df['Score'].shift(1) < 7)

    period = 10
    multiplier = 3
    high, low, close = df['Real_High'].values, df['Real_Low'].values, df['Real_Close'].values
    tr = np.zeros(len(df))
    for i in range(1, len(df)): tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    
    atr = np.zeros(len(df))
    atr[0] = tr[0]
    for i in range(1, len(df)): atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
    hl2 = (high + low) / 2
    basic_ub = hl2 + multiplier * atr
    basic_lb = hl2 - multiplier * atr
    
    final_ub, final_lb, trend, st_line = np.zeros(len(df)), np.zeros(len(df)), np.ones(len(df)), np.zeros(len(df))
    for i in range(1, len(df)):
        if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]: final_ub[i] = basic_ub[i]
        else: final_ub[i] = final_ub[i-1]
        if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]: final_lb[i] = basic_lb[i]
        else: final_lb[i] = final_lb[i-1]
        if close[i] > final_ub[i-1]: trend[i] = 1
        elif close[i] < final_lb[i-1]: trend[i] = -1
        else: trend[i] = trend[i-1]
        if trend[i] == 1: st_line[i] = final_lb[i]
        else: st_line[i] = final_ub[i]
            
    df['SuperTrend'] = st_line
    df['SuperTrend_Dir'] = trend

    return df.dropna()

# ==========================================
# --- ✨ SMC 訂單塊 (Order Block) 運算引擎 ---
# ==========================================
def get_active_obs(df, lookback, level_name):
    """
    尋找歷史上未被跌破/突破的機構訂單塊
    Bullish OB: 產生波段低點前的最後一根陰線
    Bearish OB: 產生波段高點前的最後一根陽線
    """
    if len(df) < lookback * 2: return []
    bullish_obs = []
    bearish_obs = []
    highs = df['Real_High'].values
    lows = df['Real_Low'].values
    closes = df['Real_Close'].values
    opens = df['Real_Open'].values

    for i in range(lookback, len(df) - 1):
        # 尋找波段低點 -> Bullish OB
        window_lows = lows[max(0, i-lookback) : min(len(df), i+lookback+1)]
        if lows[i] == min(window_lows):
            for j in range(i, max(-1, i-lookback-1), -1):
                if closes[j] < opens[j]: # 找到低點前的最後一根陰線
                    bullish_obs.append({'type': 'bull', 'start': j, 'top': highs[j], 'bottom': lows[j], 'level': level_name})
                    break

        # 尋找波段高點 -> Bearish OB
        window_highs = highs[max(0, i-lookback) : min(len(df), i+lookback+1)]
        if highs[i] == max(window_highs):
            for j in range(i, max(-1, i-lookback-1), -1):
                if closes[j] > opens[j]: # 找到高點前的最後一根陽線
                    bearish_obs.append({'type': 'bear', 'start': j, 'top': highs[j], 'bottom': lows[j], 'level': level_name})
                    break

    # 過濾掉已經被價格實體收盤穿越（失效）的訂單塊
    valid_obs = []
    for ob in bullish_obs:
        breached = False
        for k in range(ob['start'] + 1, len(df)):
            if closes[k] < ob['bottom']:
                breached = True
                break
        if not breached: valid_obs.append(ob)

    for ob in bearish_obs:
        breached = False
        for k in range(ob['start'] + 1, len(df)):
            if closes[k] > ob['top']:
                breached = True
                break
        if not breached: valid_obs.append(ob)

    return valid_obs

def scan_stocks(ticker_list):
    # 雷達掃描邏輯保留
    results = []
    total = len(ticker_list)
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, ticker in enumerate(ticker_list):
        ticker = str(ticker).strip().upper()
        if not ticker or ticker == 'NAN': continue
        status_text.text(f"正在掃描: {ticker} ({i+1}/{total})")
        try:
            if ticker.isdigit() and len(ticker) == 4:
                stock_data = yf.Ticker(f"{ticker}.TW").history(period="3y")
                if stock_data.empty:
                    ticker = f"{ticker}.TWO"
                    stock_data = yf.Ticker(ticker).history(period="3y")
                else: ticker = f"{ticker}.TW"
            else: stock_data = yf.Ticker(ticker).history(period="3y")
                
            if not stock_data.empty:
                df = calculate_indicators(stock_data)
                consecutive_7_days, latest_score = 0, 0
                for idx in range(len(df) - 1, -1, -1):
                    s = int(df.iloc[idx]['Score'])
                    if idx == len(df) - 1: latest_score = s 
                    if s == 7: consecutive_7_days += 1
                    else: break
                results.append({
                    "股號": ticker, "產業類別": get_industry(ticker), "最新資料日": df.index[-1].strftime('%Y-%m-%d'),
                    "收盤價": round(df.iloc[-1]['Real_Close'], 2), "多頭分數 (滿分7)": latest_score,
                    "連續滿分天數": consecutive_7_days,
                    "狀態": "🟢 全綠燈 (極強)" if latest_score == 7 else f"🟡 {latest_score}/7 燈" if latest_score >= 4 else "🔴 偏弱"
                })
        except Exception: pass
        progress_bar.progress((i + 1) / total)
        time.sleep(0.5) 
    status_text.text("掃描完成！")
    return pd.DataFrame(results)

# ==========================================
# --- 前端 Canvas 智慧無縫引擎 ---
# ==========================================
def render_javascript_canvas(ticker_symbol, k_type, show_st, show_resonance, show_pd, show_ob_current, show_ob_higher):
    ticker = yf.Ticker(ticker_symbol)
    period_map = {"日K": "1y", "周K": "3y", "月K": "7y"}
    raw_data = ticker.history(period=period_map.get(k_type, "3y"))
    
    if k_type == "周K":
        raw_data = raw_data.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    elif k_type == "月K":
        raw_data = raw_data.resample('M').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        
    if raw_data.empty:
        st.error("❌ 該股票暫時無法取得數據。")
        return

    df = calculate_indicators(raw_data)

    # ✨ ========================================================
    # ✨ 核心戰術主控台：安裝 SuperTrend 守備位與 7 燈指標儀表板
    # ✨ ========================================================
    latest_row = df.iloc[-1]
    st.markdown(f"### 🛠️ {ticker_symbol} 核心戰術主控台 ({k_type}視窗)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("當前收盤價", f"{latest_row['Real_Close']:.2f} 元")
    m2.metric("SuperTrend 守備位", f"{latest_row['SuperTrend']:.2f} 元")
    m3.metric("技術面總體評分", f"{int(latest_row['Score'])}/7 燈")
    m4.metric("歷史資料截取日", df.index[-1].strftime('%Y-%m-%d'))
    st.markdown("<br>", unsafe_allow_html=True)
    # ===========================================================
    
    # 計算機構訂單塊 (OB)
    # 當前週期：回看 6 根 K 線尋找波段
    # 更高週期：回看 20 根 K 線尋找大波段
    obs_current = get_active_obs(df, 6, 'current')
    obs_higher = get_active_obs(df, 20, 'higher')
    all_obs = obs_current + obs_higher
    
    df = df.replace({np.nan: None})
    
    df_json = []
    for idx, row in df.iterrows():
        df_json.append({
            "date": idx.strftime('%Y/%m/%d'),
            "open": float(row['HA_Open']), "high": float(row['HA_High']),
            "low": float(row['HA_Low']), "close": float(row['HA_Close']),
            "ro": float(row['Real_Open']), "rh": float(row['Real_High']),
            "rl": float(row['Real_Low']), "rc": float(row['Real_Close']),
            "sma20": float(row['SMA_20']) if row['SMA_20'] else None,
            "sma60": float(row['SMA_60']) if row['SMA_60'] else None,
            "st_val": float(row['SuperTrend']), # 傳遞給浮動標籤用
            "st_up": float(row['SuperTrend']) if row['SuperTrend_Dir'] == 1 else None,
            "st_down": float(row['SuperTrend']) if row['SuperTrend_Dir'] == -1 else None,
            "resonance": bool(row['First_Resonance']),
            "m_sma60": 1 if row['Real_Close'] > row['SMA_60'] else 0,
            "m_sma20": 1 if row['Real_Close'] > row['SMA_20'] else 0,
            "m_ema12": 1 if row['Real_Close'] > row['EMA_12'] else 0,
            "m_kd": 1 if row['K'] > row['D'] else 0,
            "m_rsi": 1 if row['RSI_14'] > 50 else 0,
            "m_macd": 1 if row['MACD_Histogram'] > 0 else 0,
            "m_ha": 1 if row['HA_Close'] > row['HA_Open'] else 0
        })
    
    json_data_str = json.dumps(df_json)
    json_ob_str = json.dumps(all_obs)
    
    # 傳遞前端標記參數
    flags = {
        "st": "true" if show_st else "false",
        "res": "true" if show_resonance else "false",
        "pd": "true" if show_pd else "false",
        "ob_curr": "true" if show_ob_current else "false",
        "ob_high": "true" if show_ob_higher else "false"
    }

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                background-color: #131722; color: #d1d4dc;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                margin: 0; padding: 10px; overflow: hidden; user-select: none;
            }}
            #control-bar {{
                display: flex; gap: 10px; margin-bottom: 10px; background: #1c2030;
                padding: 10px; border-radius: 6px; align-items: center;
            }}
            .btn {{
                background: #2a2e39; color: #d1d4dc; border: 1px solid #434651;
                padding: 6px 14px; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 13px;
            }}
            .btn.danger {{ background: #d32f2f; color: white; border-color: #d32f2f; margin-left: auto; }}
            #canvas-container {{
                position: relative; width: 100%; height: 750px;
                background: #131722; border: 1px solid #2a2e39; border-radius: 4px;
                cursor: crosshair;
            }}
            canvas {{ position: absolute; left: 0; top: 0; width: 100%; height: 100%; }}
            #hud {{
                position: absolute; top: 10px; left: 10px; background: rgba(28, 32, 48, 0.85);
                padding: 8px 12px; border-radius: 4px; font-size: 12px; line-height: 1.6;
                pointer-events: none; border: 1px solid #2a2e39; z-index: 10;
            }}
        </style>
    </head>
    <body>

        <div id="control-bar">
            <span style="font-weight:bold; color: #00e676;">⚡ 滑鼠左鍵自動並存模式開啟：</span>
            <span style="font-size:13px; color:#84858a;">👉「滑鼠按住拖曳」＝ 平移線圖 ｜「滑鼠輕點兩下」＝ 標記區間與測量波段</span>
            <button class="btn danger" onclick="clearMeasurements()">🗑️ 清除所有測量線</button>
        </div>

        <div id="canvas-container">
            <div id="hud">讀取數據中...</div>
            <canvas id="stockCanvas"></canvas>
        </div>

        <script>
            const data = {json_data_str};
            const obData = {json_ob_str};
            
            const showSuperTrend = {flags['st']};
            const showResonance = {flags['res']};
            const showPD = {flags['pd']};
            const showOBCurrent = {flags['ob_curr']};
            const showOBHigher = {flags['ob_high']};
            
            const container = document.getElementById('canvas-container');
            const canvas = document.getElementById('stockCanvas');
            const ctx = canvas.getContext('2d');
            const hud = document.getElementById('hud');

            function resizeCanvas() {{
                canvas.width = container.clientWidth;
                canvas.height = container.clientHeight;
            }}
            resizeCanvas();

            const chartHeight = canvas.height * 0.70;
            const matrixTop = canvas.height * 0.75;
            const matrixHeight = canvas.height * 0.23;
            const rightMargin = 60; 

            let candleWidth = 8;
            let gap = 2;
            let offsetX = 0;
            
            let isMouseDown = false;
            let isDragging = false; 
            let startMouseX = 0, startMouseY = 0, startOffsetX = 0;
            let currentMouseX = 0, currentMouseY = 0, isMouseHovering = false;
            
            let measureStartIdx = null;
            let measureEndIdx = null;

            const totalCandlesWidth = data.length * (candleWidth + gap);
            offsetX = canvas.width - rightMargin - totalCandlesWidth - 50;

            function clearMeasurements() {{
                measureStartIdx = null;
                measureEndIdx = null;
                draw();
            }}

            function draw() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                if (data.length === 0) return;

                // --- 1. 計算畫面可視範圍的高低點 ---
                let minPrice = Infinity, maxPrice = -Infinity;
                for (let i = 0; i < data.length; i++) {{
                    let x = offsetX + i * (candleWidth + gap);
                    if (x + candleWidth > 0 && x < canvas.width - rightMargin) {{
                        if (data[i].high > maxPrice) maxPrice = data[i].high;
                        if (data[i].low < minPrice) minPrice = data[i].low;
                        if (data[i].sma20 > maxPrice) maxPrice = data[i].sma20;
                        if (data[i].sma20 && data[i].sma20 < minPrice) minPrice = data[i].sma20;
                        if (data[i].sma60 > maxPrice) maxPrice = data[i].sma60;
                        if (data[i].sma60 && data[i].sma60 < minPrice) minPrice = data[i].sma60;
                    }}
                }}
                let priceRange = maxPrice - minPrice;
                maxPrice += priceRange * 0.05; minPrice -= priceRange * 0.05; priceRange = maxPrice - minPrice;

                function getX(index) {{ return offsetX + index * (candleWidth + gap) + candleWidth / 2; }}
                function getY(price) {{ return chartHeight - ((price - minPrice) / priceRange) * chartHeight; }}
                function getPriceFromY(y) {{ return minPrice + ((chartHeight - y) / chartHeight) * priceRange; }}

                // --- ✨ SMC 溢價/折價 區域繪製 (PD Zones) ---
                if (showPD) {{
                    let midPrice = (maxPrice + minPrice) / 2;
                    let yMax = getY(maxPrice);
                    let yMid = getY(midPrice);
                    let yMin = getY(minPrice);

                    // Premium Zone (上方紅色區)
                    ctx.fillStyle = 'rgba(239, 83, 80, 0.06)';
                    ctx.fillRect(0, yMax, canvas.width - rightMargin, yMid - yMax);

                    // Discount Zone (下方綠色區)
                    ctx.fillStyle = 'rgba(38, 166, 154, 0.06)';
                    ctx.fillRect(0, yMid, canvas.width - rightMargin, yMin - yMid);

                    // Equilibrium (50% 分界線)
                    ctx.strokeStyle = '#84858a';
                    ctx.setLineDash([4, 4]);
                    ctx.beginPath(); ctx.moveTo(0, yMid); ctx.lineTo(canvas.width - rightMargin, yMid); ctx.stroke();
                    ctx.setLineDash([]);

                    ctx.fillStyle = 'rgba(239, 83, 80, 0.6)'; ctx.font = 'bold 12px Arial';
                    ctx.fillText('Premium (溢價區)', 15, yMax + 20);
                    ctx.fillStyle = '#84858a';
                    ctx.fillText('Equilibrium (均衡)', 15, yMid - 5);
                    ctx.fillStyle = 'rgba(38, 166, 154, 0.6)';
                    ctx.fillText('Discount (折價區)', 15, yMin - 5);
                }}

                // --- ✨ SMC 訂單塊繪製 (Order Blocks) ---
                if (showOBCurrent || showOBHigher) {{
                    for (let ob of obData) {{
                        if (ob.level === 'current' && !showOBCurrent) continue;
                        if (ob.level === 'higher' && !showOBHigher) continue;

                        let startX = getX(ob.start) - candleWidth/2;
                        if (startX > canvas.width) continue; // 超出畫面右側

                        let endX = canvas.width - rightMargin; // 延伸到右側邊界
                        let yTop = getY(ob.top);
                        let yBot = getY(ob.bottom);
                        let h = yBot - yTop;

                        // 牛市訂單塊(綠色支撐), 熊市訂單塊(紅色壓力)
                        ctx.fillStyle = ob.type === 'bull' ? 'rgba(38, 166, 154, 0.15)' : 'rgba(239, 83, 80, 0.15)';
                        ctx.fillRect(startX, yTop, endX - startX, h);

                        ctx.strokeStyle = ob.type === 'bull' ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)';
                        ctx.lineWidth = ob.level === 'higher' ? 2 : 1;
                        ctx.strokeRect(startX, yTop, endX - startX, h);
                        
                        ctx.fillStyle = ctx.strokeStyle; ctx.font = '10px Arial'; ctx.textAlign = 'left';
                        ctx.fillText(ob.level === 'higher' ? '【OB】HTF' : '【OB】', startX + 5, yTop + 12);
                    }}
                }}

                // 背景格線
                ctx.strokeStyle = '#2a2e39'; ctx.lineWidth = 1; ctx.fillStyle = '#84858a'; ctx.font = '10px Arial';
                for (let i = 1; i < 5; i++) {{
                    let y = (chartHeight / 5) * i;
                    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width - rightMargin, y); ctx.stroke();
                    ctx.fillText(getPriceFromY(y).toFixed(2), canvas.width - rightMargin + 5, y + 4);
                }}

                // 均線
                ctx.beginPath(); ctx.strokeStyle = '#2962ff'; ctx.lineWidth = 1.5;
                let firstSma20 = true;
                for (let i = 0; i < data.length; i++) {{
                    if(data[i].sma20) {{
                        let x = getX(i), y = getY(data[i].sma20);
                        if (firstSma20) {{ ctx.moveTo(x, y); firstSma20 = false; }} else ctx.lineTo(x, y);
                    }}
                }}
                ctx.stroke();

                ctx.beginPath(); ctx.strokeStyle = '#9c27b0'; ctx.lineWidth = 1.5;
                let firstSma60 = true;
                for (let i = 0; i < data.length; i++) {{
                    if(data[i].sma60) {{
                        let x = getX(i), y = getY(data[i].sma60);
                        if (firstSma60) {{ ctx.moveTo(x, y); firstSma60 = false; }} else ctx.lineTo(x, y);
                    }}
                }}
                ctx.stroke();

                if (showSuperTrend) {{
                    ctx.beginPath(); ctx.strokeStyle = '#00E676'; ctx.lineWidth = 2; ctx.setLineDash([4, 4]);
                    let firstStUp = true;
                    for(let i=0; i<data.length; i++) {{
                        if(data[i].st_up) {{
                            let x = getX(i), y = getY(data[i].st_up);
                            if(firstStUp) {{ ctx.moveTo(x,y); firstStUp=false; }} else ctx.lineTo(x,y);
                        }} else {{ firstStUp = true; }}
                    }}
                    ctx.stroke();
                    
                    ctx.beginPath(); ctx.strokeStyle = '#FF5252';
                    let firstStDown = true;
                    for(let i=0; i<data.length; i++) {{
                        if(data[i].st_down) {{
                            let x = getX(i), y = getY(data[i].st_down);
                            if(firstStDown) {{ ctx.moveTo(x,y); firstStDown=false; }} else ctx.lineTo(x,y);
                        }} else {{ firstStDown = true; }}
                    }}
                    ctx.stroke();
                    ctx.setLineDash([]);
                }}

                // K線實體
                ctx.textAlign = 'center';
                for (let i = 0; i < data.length; i++) {{
                    let x = offsetX + i * (candleWidth + gap);
                    if (x + candleWidth < 0 || x > canvas.width - rightMargin) continue;

                    let isUp = data[i].close >= data[i].open;
                    let color = isUp ? '#ef5350' : '#26a69a';
                    ctx.fillStyle = color; ctx.strokeStyle = color; ctx.lineWidth = 1.5;

                    ctx.beginPath();
                    ctx.moveTo(x + candleWidth / 2, getY(data[i].high));
                    ctx.lineTo(x + candleWidth / 2, getY(data[i].low));
                    ctx.stroke();

                    let yOpen = getY(data[i].open), yClose = getY(data[i].close);
                    let bodyHeight = Math.max(Math.abs(yClose - yOpen), 1);
                    ctx.fillRect(x, Math.min(yOpen, yClose), candleWidth, bodyHeight);

                    if (showResonance && data[i].resonance) {{
                        ctx.fillStyle = '#FFD700'; ctx.font = 'bold 12px Arial';
                        ctx.fillText('🔥起漲', getX(i), getY(data[i].rl) + 20);
                    }}
                }}

                // 測量區間
                if (measureStartIdx !== null) {{
                    let endIdx = (measureEndIdx !== null) ? measureEndIdx : getIndexFromX(currentMouseX);
                    if (endIdx >= 0 && endIdx < data.length) {{
                        let xStart = getX(measureStartIdx), xEnd = getX(endIdx);
                        let pStart = data[measureStartIdx].rc;
                        let pEnd = data[endIdx].rc;
                        let yStart = getY(pStart), yEnd = getY(pEnd);

                        ctx.strokeStyle = '#FFD700'; ctx.lineWidth = 1.5;
                        ctx.beginPath(); ctx.moveTo(0, yStart); ctx.lineTo(canvas.width - rightMargin, yStart); ctx.stroke();
                        ctx.fillStyle = '#FFD700'; ctx.fillRect(canvas.width - rightMargin, yStart - 12, rightMargin, 24);
                        ctx.fillStyle = '#000000'; ctx.font = 'bold 11px Arial'; ctx.textAlign = 'center';
                        ctx.fillText(pStart.toFixed(2), canvas.width - (rightMargin / 2), yStart + 4);
                        ctx.fillStyle = '#FFD700'; ctx.textAlign = 'left';
                        ctx.fillText(`📌 起點: ${{data[measureStartIdx].date}}`, 15, yStart - 6);

                        ctx.strokeStyle = '#ffeb3b';
                        ctx.beginPath(); ctx.moveTo(0, yEnd); ctx.lineTo(canvas.width - rightMargin, yEnd); ctx.stroke();
                        ctx.fillStyle = '#ffeb3b'; ctx.fillRect(canvas.width - rightMargin, yEnd - 12, rightMargin, 24);
                        ctx.fillStyle = '#000000'; ctx.font = 'bold 11px Arial'; ctx.textAlign = 'center';
                        ctx.fillText(pEnd.toFixed(2), canvas.width - (rightMargin / 2), yEnd + 4);

                        ctx.fillStyle = 'rgba(255, 235, 59, 0.08)';
                        ctx.fillRect(Math.min(xStart, xEnd), Math.min(yStart, yEnd), Math.abs(xEnd - xStart), Math.abs(yEnd - yStart));
                        
                        let pct = ((pEnd - pStart) / pStart) * 100;
                        let diff = pEnd - pStart;
                        let barsCount = Math.abs(endIdx - measureStartIdx) + 1;

                        ctx.fillStyle = '#ffeb3b'; ctx.font = 'bold 14px Arial'; ctx.textAlign = 'left';
                        let summaryText = `📐 波段回報：${{pct >= 0 ? '＋' : ''}}${{pct.toFixed(2)}}% (${{diff >= 0 ? '＋' : ''}}${{diff.toFixed(2)}} 元, ${{barsCount}} 根K線)`;
                        ctx.fillText(summaryText, Math.max(xStart, xEnd) + 12, yEnd + (yStart > yEnd ? 18 : -10));
                    }}
                }}

                // 七星燈歷史矩陣
                const matrixLabels = ['HA', 'MACD', 'RSI', 'KD', 'EMA 12', 'SMA 20', 'SMA 60'];
                const rowH = matrixHeight / 7;

                for (let i = 0; i < data.length; i++) {{
                    let x = offsetX + i * (candleWidth + gap);
                    if (x + candleWidth < 0 || x > canvas.width - rightMargin) continue;
                    
                    const indicators = [data[i].m_ha, data[i].m_macd, data[i].m_rsi, data[i].m_kd, data[i].m_ema12, data[i].m_sma20, data[i].m_sma60];
                    for (let j = 0; j < 7; j++) {{
                        ctx.fillStyle = indicators[j] === 1 ? '#388e3c' : '#d32f2f';
                        ctx.fillRect(x, matrixTop + j * rowH, candleWidth, rowH - 1);
                    }}
                }}

                ctx.fillStyle = 'rgba(19, 23, 34, 0.85)';
                ctx.fillRect(0, matrixTop, 60, matrixHeight);
                ctx.fillStyle = '#d1d4dc'; ctx.font = '10px Arial'; ctx.textAlign = 'left';
                for (let j = 0; j < 7; j++) {{
                    ctx.fillText(matrixLabels[j], 5, matrixTop + j * rowH + rowH/1.5);
                }}

                // 十字追蹤準心線
                if (isMouseHovering) {{
                    ctx.strokeStyle = '#84858a'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
                    if (currentMouseX < canvas.width - rightMargin) {{
                        ctx.beginPath(); ctx.moveTo(currentMouseX, 0); ctx.lineTo(currentMouseX, canvas.height); ctx.stroke();
                    }}
                    if (currentMouseY > 0 && currentMouseY <= chartHeight) {{
                        ctx.beginPath(); ctx.moveTo(0, currentMouseY); ctx.lineTo(canvas.width - rightMargin, currentMouseY); ctx.stroke();
                        let currentPrice = getPriceFromY(currentMouseY).toFixed(2);
                        ctx.fillStyle = '#1e88e5'; 
                        ctx.fillRect(canvas.width - rightMargin, currentMouseY - 12, rightMargin, 24);
                        ctx.fillStyle = 'white'; ctx.font = 'bold 12px Arial'; ctx.textAlign = 'center';
                        ctx.fillText(currentPrice, canvas.width - (rightMargin / 2), currentMouseY + 4);
                    }}
                    ctx.setLineDash([]); 
                }}
            }}

            function getIndexFromX(x) {{ return Math.floor((x - offsetX) / (candleWidth + gap)); }}

            canvas.addEventListener('mouseenter', () => {{ isMouseHovering = true; }});
            canvas.addEventListener('mouseleave', () => {{ isMouseHovering = false; draw(); }});

            canvas.addEventListener('mousedown', (e) => {{
                const rect = canvas.getBoundingClientRect();
                startMouseX = e.clientX - rect.left;
                startMouseY = e.clientY - rect.top;
                
                isMouseDown = true;
                isDragging = false; 
                startOffsetX = offsetX;
            }});

            canvas.addEventListener('mousemove', (e) => {{
                const rect = canvas.getBoundingClientRect();
                currentMouseX = e.clientX - rect.left;
                currentMouseY = e.clientY - rect.top; 
                let idx = getIndexFromX(currentMouseX);

                if (idx >= 0 && idx < data.length) {{
                    hud.innerHTML = `
                        <b>🗓️ 日期: ${{data[idx].date}}</b><br>
                        開盤: ${{data[idx].ro.toFixed(2)}} | 最高: ${{data[idx].rh.toFixed(2)}}<br>
                        最低: ${{data[idx].rl.toFixed(2)}} | <b>收盤: ${{data[idx].rc.toFixed(2)}}</b><br>
                        <span style="color:#2962ff">SMA20: ${{data[idx].sma20 ? data[idx].sma20.toFixed(2) : '-'}}</span> |
                        <span style="color:#9c27b0">SMA60: ${{data[idx].sma60 ? data[idx].sma60.toFixed(2) : '-'}}</span><br>
                        <span style="color:#ff9800; font-weight:bold;">🛡️ ST守備位: ${{data[idx].st_val ? data[idx].st_val.toFixed(2) : '-'}}</span>
                    `;
                }}

                if (isMouseDown) {{
                    let moveX = Math.abs(currentMouseX - startMouseX);
                    if (moveX > 5) {{
                        isDragging = true;
                    }}
                    if (isDragging) {{
                        offsetX = startOffsetX + (currentMouseX - startMouseX);
                    }}
                }}
                draw(); 
            }});

            window.addEventListener('mouseup', () => {{
                if (!isMouseDown) return;
                isMouseDown = false;

                if (isDragging) {{
                    isDragging = false;
                }} else {{
                    let idx = getIndexFromX(startMouseX);
                    if (idx >= 0 && idx < data.length) {{
                        if (measureStartIdx === null) {{
                            measureStartIdx = idx;
                            measureEndIdx = null;
                        }} else if (measureStartIdx !== null && measureEndIdx === null) {{
                            measureEndIdx = idx;
                        }} else {{
                            measureStartIdx = idx;
                            measureEndIdx = null;
                        }}
                    }}
                }}
                draw();
            }});

            canvas.addEventListener('wheel', (e) => {{
                e.preventDefault();
                const rect = canvas.getBoundingClientRect();
                let mouseX = e.clientX - rect.left;
                let idxBefore = getIndexFromX(mouseX);
                if (e.deltaY < 0) {{ if (candleWidth < 50) candleWidth += 1; }} 
                else {{ if (candleWidth > 3) candleWidth -= 1; }}
                let idxAfter = getIndexFromX(mouseX);
                offsetX += (idxAfter - idxBefore) * (candleWidth + gap);
                draw();
            }}, {{ passive: false }});

            draw();
        </script>
    </body>
    </html>
    """
    st.components.v1.html(html_code, height=850, scrolling=False)

# ==========================================
# --- 網頁前端 UI 佈局 ---
# ==========================================
st.title("🚦 專業多重技術分析與選股系統")

with st.spinner("系統初始化中..."):
    tw_stock_list = get_tw_stock_list()

tab1, tab2 = st.tabs(["📊 JavaScript 零延遲技術線圖", "🚀 強勢多頭掃描器"])

with tab1:
    col_input, col_chart = st.columns([1, 5])
    with col_input:
        st.markdown("### 指標設定")
        if tw_stock_list:
            default_index = 0
            for i, s in enumerate(tw_stock_list):
                if "2330.TW" in s:
                    default_index = i
                    break
            selected_stock = st.selectbox("🔍 選擇台股", options=tw_stock_list, index=default_index)
            user_ticker = selected_stock.split(" - ")[0]
        else:
            user_ticker = st.text_input("🔍 手動輸入股號", value="2330.TW")

        st.markdown("---")
        k_line_type = st.radio(" K 線時間週期變更", ["日K", "周K", "月K"], horizontal=True)

        st.markdown("---")
        st.markdown("### 🎯 戰術與 SMC 聰明錢標記")
        show_supertrend = st.checkbox("啟用 SuperTrend (超級趨勢線)", value=False)
        show_resonance = st.checkbox("標示 首次共振轉多 (🔥 起漲點)", value=True)
        # ✨ 新增的 SMC 選項
        show_pd = st.checkbox("【PD】溢價/折價 區域", value=False)
        show_ob_current = st.checkbox("【OB】訂單塊 (當前週期)", value=False)
        show_ob_higher = st.checkbox("【OB】訂單塊 (更高週期)", value=False)

    with col_chart:
        if user_ticker:
            with st.spinner("載入底層數據中..."):
                render_javascript_canvas(user_ticker.upper(), k_line_type, show_supertrend, show_resonance, show_pd, show_ob_current, show_ob_higher)

with tab2:
    st.markdown("### 建立您的「核心觀察池」")
    input_method = st.radio("選擇名單匯入方式:", ["⌨️ 手動輸入", "📁 上傳 Excel / CSV 檔案", "🏢 選擇特定產業 (全市場過濾)"], horizontal=True)
    final_ticker_list = []
    
    if input_method == "⌨️ 手動輸入":
        default_list = "2330.TW, 2317.TW, 2454.TW, 2603.TW"
        user_list_str = st.text_area("📋 輸入觀察名單 (半形逗號分隔)", value=default_list)
        if user_list_str: final_ticker_list = user_list_str.split(",")
    elif input_method == "📁 上傳 Excel / CSV 檔案":
        uploaded_file = st.file_uploader("選擇檔案", type=['xlsx', 'csv'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'): df_upload = pd.read_csv(uploaded_file)
                else: df_upload = pd.read_excel(uploaded_file)
                first_col = df_upload.columns[0]
                raw_list = df_upload[first_col].dropna().astype(str).tolist()
                final_ticker_list = [t for t in raw_list if t.strip()]
                st.success(f"✅ 成功從檔案中讀取 {len(final_ticker_list)} 檔股票名單！")
            except Exception as e: st.error(f"❌ 檔案讀取失敗: {e}")
    elif input_method == "🏢 選擇特定產業 (全市場過濾)":
        if tw_stock_list:
            industry_groups = {}
            for item in tw_stock_list:
                ticker_part = item.split(" - ")[0]
                pure_ticker = ticker_part.split(".")[0]
                ind = get_industry(pure_ticker)
                if ind not in industry_groups: industry_groups[ind] = []
                industry_groups[ind].append(ticker_part)
                
            valid_industries = [ind for ind in industry_groups.keys() if ind != "其他 / 特殊類股" and ind != "ETF / 權證 / 外國企業"]
            valid_industries.sort()
            valid_industries.append("其他 / 特殊類股")
            valid_industries.append("ETF / 權證 / 外國企業")
            selected_ind = st.selectbox("📊 選擇要掃描的產業族群", options=valid_industries)
            if selected_ind:
                final_ticker_list = industry_groups[selected_ind]
                st.info(f"💡 【{selected_ind}】族群共有 {len(final_ticker_list)} 檔股票。")

    if st.button("⚡ 開始一鍵掃描", type="primary"):
        if final_ticker_list:
            result_df = scan_stocks(final_ticker_list)
            if not result_df.empty:
                perfect_scores = result_df[result_df["多頭分數 (滿分7)"] == 7]
                st.markdown("---")
                if not perfect_scores.empty:
                    perfect_scores = perfect_scores.sort_values(by="連續滿分天數", ascending=True)
                    st.success(f"🎉 發現 {len(perfect_scores)} 檔「七燈全亮」的極強勢股！")
                    st.dataframe(perfect_scores, use_container_width=True)
                with st.expander("查看完整掃描報告", expanded=True):
                    sorted_full_report = result_df.sort_values(by=["多頭分數 (滿分7)", "連續滿分天數"], ascending=[False, True])
                    st.dataframe(sorted_full_report, use_container_width=True)