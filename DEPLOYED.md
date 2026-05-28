# InferAI Deployment Guide

## क्या deploy करना है?

इस प्रोजेक्ट में दो मुख्य हिस्से हैं:

1. `api/app.py` — FastAPI backend जो `/analyze` endpoint चलाता है।
2. `frontend/app.py` — Streamlit frontend UI जो backend API से डेटा मांगता है।

> अगर आप केवल backend deploy करना चाहते हैं, तो `InferAI/` root से deploy करें क्योंकि `vercel.json` भी root में configured है।

---

## कौन सा फोल्डर उपयोग करें?

- `InferAI/` — मुख्य repository root
- `InferAI/api/` — backend service फ़ोल्डर
- `InferAI/frontend/` — local Streamlit UI फ़ोल्डर
- `InferAI/web_frontend/` — optional static web UI फ़ोल्डर (यदि आप HTML/JS based frontend deploy करना चाहें)

## कisko कहाँ deploy करें?

- `api/app.py` को deploy करें: Vercel या कोई Python-compatible cloud host
- `frontend/app.py` को deploy करें: local machine, Streamlit Cloud, या किसी VM/container पर
- `web_frontend/` को deploy करें: static site host जैसे Vercel, Netlify, या GitHub Pages
- ML मॉडल फाइलें (`models/infer_model.pkl`, `models/label_encoder.pkl`, `models/shap_background.npy`) backend के साथ ही रहनी चाहिए क्योंकि backend उन्हें runtime पर लोड करता है

---

## Local Deployment (सबसे आसान)

### 1) Root फ़ोल्डर में जाएँ

```powershell
cd c:\Users\munta\Desktop\ak-pr\InferAI
```

### 2) वर्चुअल एन्वाइरनमेंट बनाएं और सक्रिय करें

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3) dependencies install करें

```powershell
pip install -r requirements.txt
```

### 4) Backend API चलाएँ

```powershell
python -m uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

### 5) Frontend UI चलाएँ

```powershell
streamlit run frontend/app.py --server.port 8501 --server.headless true
```

### यूआरएल्स

- Backend API: `http://127.0.0.1:8000`
- Swagger Docs: `http://127.0.0.1:8000/docs`
- Frontend UI: `http://127.0.0.1:8501`

---

## Deploy on Vercel (API के लिए)

इस repository में पहले से `vercel.json` मौजूद है। इसका मतलब है कि backend API Vercel पर deploy किया जा सकता है सीधे `api/app.py` से।

### कैसे deploy करें

1. GitHub पर repo push करें।
2. Vercel में नया प्रोजेक्ट बनाएँ और इस repo को connect करें।
3. Vercel auto-detect करेगा कि Python backend है।
4. deploy पूरा होने पर आपको एक public Vercel URL मिलेगा।

### Vercel config

`vercel.json` में यह बताया गया है:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "(.*)",
      "dest": "api/app.py"
    }
  ]
}
```

> यह configuration पूरे repo को API सेवा की तरह deploy करती है। इसका अर्थ है कि `api/app.py` root deployment entrypoint है।

---

## Static Frontend Deploy करने का तरीका

### A) Streamlit Cloud

यदि आप GUI को cloud में deploy करना चाहते हैं, तो `frontend/app.py` को Streamlit Cloud पर डालें।

- `frontend/app.py` रोल करता है Streamlit app
- backend API को चलाना होगा या Vercel से deploy किया हुआ यूआरएल देना होगा

यदि backend Vercel पर है तो Streamlit को `INFERAI_API_URL` में उस URL की ज़रूरत होगी।

> हां, Streamlit Cloud पर frontend deploy करने से यह काम करेगा, बशर्ते backend API अलग से चल रही हो।

### B) Vercel Static Frontend

यदि आप स्टैटिक HTML/JS frontend deploy करना चाहें, तो `web_frontend/` folder use कर सकते हैं।
- `web_frontend/index.html`
- `web_frontend/app.js`
- `web_frontend/style.css`
- `web_frontend/vercel.json`

यह एक अलग प्रोजेक्ट के रूप में deploy होगा, और API URL को अपने backend Vercel endpoint पर point करना होगा।

---

## URL सेटिंग

यदि frontend को backend API का custom host चाहिए, तो environment variable सेट करें:

```powershell
$env:INFERAI_API_URL="http://your-api-host:8000"
streamlit run frontend/app.py --server.port 8501 --server.headless true
```

या Linux/macOS पर:

```bash
INFERAI_API_URL=http://your-api-host:8000 streamlit run frontend/app.py --server.port 8501 --server.headless true
```

---

## क्या deploy करें?

### सबसे आसान deployment path

1. `api/app.py` को Vercel पर deploy करें।
2. `frontend/app.py` को local machine पर चलाएँ या Streamlit Cloud पर deploy करें।
3. Frontend को backend API URL से connect करें।

### अगर आप पूरी app को एक साथ deploy करना चाहते हैं

- Backend के लिए Vercel
- Frontend के लिए Streamlit Cloud या `web_frontend/` से अलग static deployment

---

## Quick Checklist

- [ ] `cd InferAI`
- [ ] `pip install -r requirements.txt`
- [ ] `python -m uvicorn api.app:app --reload --host 127.0.0.1 --port 8000`
- [ ] `streamlit run frontend/app.py --server.port 8501 --server.headless true`
- [ ] Vercel पर deploy करने के लिए GitHub repo connect करें
- [ ] अगर static frontend deploy करना है, तो `web_frontend/` use करें
- [ ] `INFERAI_API_URL` सही backend endpoint पर सेट करें
