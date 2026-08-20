from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import requests
import json
import os
import re

# Export biến 'app' chuẩn ASGI để Vercel nhận diện
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StockRequest(BaseModel):
    symbol: str

@app.post("/api/analyze")
def analyze_stock(req: StockRequest):
    sym = req.symbol.upper().strip()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Chưa cấu hình GEMINI_API_KEY trên Vercel Environment Variables")

    # 1. Lấy dữ liệu 6 tháng từ Yahoo Finance Gateway
    try:
        ticker = f"{sym}.VN"
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo", interval="1d")
        if df is None or df.empty or len(df) < 20:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy dữ liệu cho mã '{sym}'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi truy xuất dữ liệu: {str(e)}")

    # 2. Tính toán các chỉ báo kỹ thuật định lượng
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
    }

    # 3. Gửi sang Gemini 3.6 Flash
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={api_key}"
    prompt = f"""
Bạn là Chuyên gia Tư vấn Đầu tư Chứng khoán Cao cấp. Phân tích dữ liệu mã {sym}:
- Giá hiện tại: {metrics['current_price']:,.0f} VNĐ ({metrics['percent_change']:+.2f}%)
- Khối lượng: {metrics['volume']:,} CP (TB 20P: {metrics['avg_vol_20']:,} CP)
- RSI(14): {metrics['rsi']} | SMA20: {metrics['sma20']:,.0f} | SMA50: {metrics['sma50']:,.0f}
- Hỗ trợ (20P): {metrics['support_20']:,.0f} VNĐ | Kháng cự (20P): {metrics['resistance_20']:,.0f} VNĐ

Trả về DUY NHẤT 1 JSON Object:
{{
  "action": "MUA MỚI" | "MUA GIA TĂNG" | "NẮM GIỮ" | "BÁN HẠ TỶ TRỌNG" | "BÁN CẮT LỖ" | "THEO DÕI",
  "buy_zone": "Mức giá mua tối ưu (VNĐ)",
  "target_price": "Mục tiêu giá ngắn hạn",
  "stop_loss": "Mức giá cắt lỗ",
  "risk_reward_ratio": "Tỷ lệ R:R",
  "trend_weekly": "TĂNG" | "GIẢM" | "TÍCH LŨY",
  "trend_monthly": "TĂNG" | "GIẢM" | "TÍCH LŨY",
  "catalysts": ["Nhận định dòng tiền", "Trạng thái kỹ thuật", "Kế hoạch đi lệnh"],
  "capital_allocation": "Tỷ trọng giải ngân (% NAV)"
}}
"""
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "response_mime_type": "application/json"}
    }

    try:
        resp = requests.post(gemini_url, json=payload, timeout=45).json()
        if "error" in resp:
            raise Exception(resp["error"].get("message", "Lỗi Gemini API"))
        raw_text = resp["candidates"][0]["content"]["parts"][0]["text"].strip()
        clean_text = re.sub(r"^```json\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()
        advice = json.loads(clean_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý AI: {str(e)}")

    return {"metrics": metrics, "advice": advice}