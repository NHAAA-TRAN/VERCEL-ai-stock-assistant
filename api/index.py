from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import requests
import json
import os
import re
import time
from datetime import datetime, timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bộ nhớ đệm In-Memory Cache (TTL: 300 giây = 5 phút)
ANALYSIS_CACHE = {}

class StockRequest(BaseModel):
    symbol: str

def get_next_trading_days(start_date: datetime, count: int = 5):
    days = []
    curr = start_date + timedelta(days=1)
    while len(days) < count:
        if curr.weekday() < 5:
            days.append(curr.strftime("%d/%m"))
        curr += timedelta(days=1)
    return days

@app.post("/api/analyze")
@app.post("/analyze")
@app.post("/")
@app.post("/api/index.py")
def analyze_stock(req: StockRequest):
    sym = req.symbol.upper().strip()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Chưa cấu hình GEMINI_API_KEY trên Vercel Environment Variables")

    # 1. Kiểm tra Cache trong 5 phút (Chống spam quota API)
    now = time.time()
    if sym in ANALYSIS_CACHE:
        cached_data, cached_time = ANALYSIS_CACHE[sym]
        if now - cached_time < 300:  # Dưới 5 phút trả về kết quả lưu sẵn ngay lập tức
            return cached_data

    # 2. Lấy dữ liệu 6 tháng từ Yahoo Finance
    try:
        ticker = f"{sym}.VN"
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo", interval="1d")
        if df is None or df.empty or len(df) < 20:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy dữ liệu cho mã '{sym}'. Vui lòng kiểm tra lại mã cổ phiếu.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi truy xuất dữ liệu: {str(e)}")

    # 3. Tính toán các chỉ báo kỹ thuật
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    curr_price = float(latest["Close"])
    change = curr_price - float(prev["Close"])
    pct_change = (change / float(prev["Close"])) * 100

    recent_10 = df.tail(10)
    history_dates = [pd.to_datetime(d).strftime("%d/%m") for d in recent_10.index]
    history_prices = [round(float(p), 0) for p in recent_10["Close"]]

    last_trade_date = pd.to_datetime(recent_10.index[-1]).to_pydatetime()
    future_dates = get_next_trading_days(last_trade_date, count=5)

    metrics = {
        "symbol": sym,
        "current_price": curr_price,
        "change": change,
        "percent_change": pct_change,
        "volume": int(latest["Volume"]),
        "avg_vol_20": int(df["Volume"].tail(20).mean()),
        "rsi": round(float(latest["RSI"]), 1) if not pd.isna(latest["RSI"]) else 50.0,
        "sma20": round(float(latest["SMA20"]), 0) if not pd.isna(latest["SMA20"]) else curr_price,
        "sma50": round(float(latest["SMA50"]), 0) if not pd.isna(latest["SMA50"]) else curr_price,
        "support_20": float(df["Low"].tail(20).min()),
        "resistance_20": float(df["High"].tail(20).max()),
        "history_dates": history_dates,
        "history_prices": history_prices,
        "future_dates": future_dates
    }

    # 4. Gửi sang Gemini 3.6 Flash
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={api_key}"
    prompt = f"""
Bạn là Chuyên gia Tư vấn Đầu tư Chứng khoán. Phân tích mã {sym}:
- Giá: {metrics['current_price']:,.0f} VNĐ ({metrics['percent_change']:+.2f}%)
- Khối lượng: {metrics['volume']:,} CP (TB 20P: {metrics['avg_vol_20']:,} CP)
- RSI(14): {metrics['rsi']} | SMA20: {metrics['sma20']:,.0f} | SMA50: {metrics['sma50']:,.0f}
- Hỗ trợ: {metrics['support_20']:,.0f} | Kháng cự: {metrics['resistance_20']:,.0f}
- 10 phiên qua: {history_prices}

Trả về DUY NHẤT 1 JSON Object:
{{
  "action": "MUA MỚI" | "MUA GIA TĂNG" | "NẮM GIỮ" | "BÁN HẠ TỶ TRỌNG" | "BÁN CẮT LỖ" | "THEO DÕI",
  "buy_zone": "Mức giá mua tối ưu",
  "target_price": "Mục tiêu giá",
  "stop_loss": "Mức giá cắt lỗ",
  "risk_reward_ratio": "Tỷ lệ R:R",
  "trend_weekly": "TĂNG" | "GIẢM" | "TÍCH LŨY",
  "trend_monthly": "TĂNG" | "GIẢM" | "TÍCH LŨY",
  "catalysts": [
    "Nhận xét dòng tiền & khối lượng",
    "Nhận xét RSI và MA",
    "Chiến lược giao dịch đề xuất"
  ],
  "capital_allocation": "Tỷ trọng giải ngân (% NAV)",
  "predicted_5d_prices": [giá_T1, giá_T2, giá_T3, giá_T4, giá_T5],
  "prediction_comment": "Nhận xét ngắn về đường đi của giá 5 phiên tới"
}}
"""
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 800,
            "response_mime_type": "application/json"
        }
    }

    try:
        resp = requests.post(gemini_url, json=payload, timeout=40)
        res_json = resp.json()
        
        # Bắt lỗi Quota Exceeded (HTTP 429)
        if resp.status_code == 429 or ("error" in res_json and "quota" in res_json["error"].get("message", "").lower()):
            raise HTTPException(
                status_code=429, 
                detail="⚠️ Hạn mức gọi AI miễn phí trong phút này đã đạt giới hạn. Vui lòng đợi 30 giây rồi bấm lại."
            )

        if "error" in res_json:
            raise HTTPException(status_code=500, detail=res_json["error"].get("message", "Lỗi Gemini API"))

        raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
        clean_text = re.sub(r"^```json\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()
        advice = json.loads(clean_text)

        result_payload = {"metrics": metrics, "advice": advice}
        
        # Lưu vào cache 5 phút
        ANALYSIS_CACHE[sym] = (result_payload, time.time())
        return result_payload

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")
