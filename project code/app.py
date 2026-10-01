
# ============================================================
# PHISHGUARD AI — DEPLOYMENT BACKEND
# ============================================================

import os
import sys
import re
import math
import joblib
import numpy as np
import pandas as pd
import gradio as gr

from scipy.sparse import hstack
from urllib.parse import urlparse
import ipaddress
from urllib.parse import parse_qs


# ============================================================
# LOAD PRODUCTION MODEL ARTIFACTS
# ============================================================

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

balanced_rf_model = joblib.load(
    os.path.join(MODEL_DIR, "balanced_rf_model.pkl")
)

balanced_logistic_model = joblib.load(
    os.path.join(MODEL_DIR, "balanced_logistic_model.pkl")
)

tfidf = joblib.load(
    os.path.join(MODEL_DIR, "balanced_tfidf_vectorizer.pkl")
)

tld_encoder = joblib.load(
    os.path.join(MODEL_DIR, "balanced_tld_encoder.pkl")
)

numeric_features = joblib.load(
    os.path.join(MODEL_DIR, "numeric_features.pkl")
)

# Random Forest is the production model used by the UI
best_model = balanced_rf_model

print("✓ Production Random Forest loaded")
print("✓ Logistic Regression loaded")
print("✓ TF-IDF vectorizer loaded")
print("✓ TLD encoder loaded")
print("✓ Numeric feature list loaded")

print(
    "✓ Expected feature count:",
    best_model.n_features_in_
)


# ============================================================
# URL NORMALIZATION
# ============================================================

def normalize_url(url):

    if url is None:
        return ""

    url = str(url).strip()

    if not url:
        return ""

    if not url.lower().startswith(("http://", "https://")):
        url = "http://" + url

    return url


def calculate_entropy(text):

    if not text:
        return 0.0

    probabilities = [
        text.count(char) / len(text)
        for char in set(text)
    ]

    return -sum(
        p * math.log2(p)
        for p in probabilities
        if p > 0
    )


# ------------------------------------------------------------
# 3. Extract URL features
# ------------------------------------------------------------


def extract_url_features(url):

    url = str(url).strip()

    if not url:
        raise ValueError("URL cannot be empty.")

    original_url = url

    # Add scheme if missing
    if not re.match(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
        url
    ):
        url = "http://" + url

    parsed = urlparse(url)

    hostname = parsed.netloc.split("@")[-1].split(":")[0]

    # URL length
    url_length = len(original_url)

    # Dots
    num_dots = original_url.count(".")

    # HTTPS
    has_https = int(
        parsed.scheme.lower() == "https"
    )

    # IP
    has_ip = int(
        bool(
            re.match(
                r"^\d{1,3}(\.\d{1,3}){3}$",
                hostname
            )
        )
    )

    # Subdirectories
    num_subdirs = len([
        x
        for x in parsed.path.split("/")
        if x
    ])

    # Parameters
    num_params = len(
        parse_qs(parsed.query)
    )

    # Suspicious words
    suspicious_terms = [
        "login",
        "verify",
        "account",
        "update",
        "secure",
        "password",
        "confirm",
        "bank",
        "paypal",
        "signin",
        "security",
        "credential",
        "authenticate",
        "wallet",
        "payment"
    ]

    lower_url = original_url.lower()

    suspicious_words = sum(
        1
        for word in suspicious_terms
        if word in lower_url
    )

    # Special characters
    special_char_count = sum(
        1
        for c in original_url
        if not c.isalnum()
    )

    # Digits
    digits_count = sum(
        1
        for c in original_url
        if c.isdigit()
    )

    # Entropy
    if original_url:

        counts = {
            char: original_url.count(char)
            for char in set(original_url)
        }

        entropy = -sum(
            (count / len(original_url))
            * math.log2(count / len(original_url))
            for count in counts.values()
        )

    else:

        entropy = 0.0

    # TLD
    if "." in hostname:

        tld = hostname.split(".")[-1].lower()

    else:

        tld = "unknown"

    return {
        "url_length": url_length,
        "num_dots": num_dots,
        "has_https": has_https,
        "has_ip": has_ip,
        "num_subdirs": num_subdirs,
        "num_params": num_params,
        "suspicious_words": suspicious_words,
        "special_char_count": special_char_count,
        "digits_count": digits_count,
        "entropy": entropy,
        "tld": tld
    }


# ============================================================
# PRODUCTION PREDICTION BRIDGE
# ============================================================

def predict_url_app(url):

    url = normalize_url(url)

    if not url:
        raise ValueError("Please enter a URL.")

    features = extract_url_features(url)
    features = dict(features)

    if "tld" not in features or features["tld"] is None:
        features["tld"] = "unknown"

    # Numerical features
    numeric_values = []

    for feature in numeric_features:

        if feature not in features:
            raise KeyError(
                f"Missing feature '{feature}' from URL extractor."
            )

        numeric_values.append(
            float(features[feature])
        )

    X_num = np.asarray(
        numeric_values,
        dtype=float
    ).reshape(1, -1)

    # TLD
    tld_value = str(
        features["tld"]
    ).strip().lower()

    if not tld_value:
        tld_value = "unknown"

    X_tld = tld_encoder.transform(
        pd.DataFrame({
            "tld": [tld_value]
        })
    )

    # Character TF-IDF
    X_tfidf = tfidf.transform([url])

    # Combined 10,239-feature representation
    X = hstack([
        X_num,
        X_tld,
        X_tfidf
    ]).tocsr()

    # Safety check
    expected_features = best_model.n_features_in_
    actual_features = X.shape[1]

    if expected_features != actual_features:

        raise ValueError(
            "Feature dimension mismatch.\n"
            f"Model expects: {expected_features}\n"
            f"Prediction contains: {actual_features}"
        )

    prediction = int(
        best_model.predict(X)[0]
    )

    probabilities = best_model.predict_proba(X)[0]

    classes = list(best_model.classes_)

    probability_map = {
        int(cls): float(prob)
        for cls, prob in zip(classes, probabilities)
    }

    legitimate_probability = probability_map.get(0, 0.0)
    phishing_probability = probability_map.get(1, 0.0)

    confidence = max(
        legitimate_probability,
        phishing_probability
    )

    return {
        "url": url,
        "prediction": prediction,
        "legitimate_probability": legitimate_probability,
        "phishing_probability": phishing_probability,
        "confidence": confidence,
        "features": features
    }



# ============================================================
# PHISHGUARD AI — DEPLOYMENT UI (v2)
# ============================================================

import html as _h
from datetime import datetime

RISKY_TLDS = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "click", "zip", "work", "support", "country", "kim"}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root, .gradio-container, .dark {
  --ink:#e8f1ff; --muted:rgba(178,199,235,.70); --bg:#080d1a; --cyan:#22d3ee; --safe:#34f5a0; --warn:#ffc14d; --bad:#ff4d6d;
  --glass:rgba(16,26,50,.58); --edge:rgba(120,180,255,.18);
  --neu-out:9px 9px 20px rgba(0,0,0,.55), -7px -7px 16px rgba(44,62,106,.20);
  --neu-in:inset 6px 6px 12px rgba(0,0,0,.55), inset -5px -5px 10px rgba(44,62,106,.22);
  --body-background-fill:transparent; --background-fill-primary:transparent; --background-fill-secondary:transparent;
  --block-background-fill:transparent; --block-border-width:0px; --block-border-color:transparent;
  --body-text-color:var(--ink); --block-label-text-color:var(--muted); --block-title-text-color:var(--muted);
  --input-background-fill:#0a1224; --input-border-color:transparent; --input-border-width:0px;
  --input-placeholder-color:rgba(178,199,235,.40); --border-color-primary:var(--edge);
  --table-even-background-fill:transparent; --table-odd-background-fill:rgba(255,255,255,.03);
  --button-secondary-background-fill:#0d1730; --button-secondary-text-color:var(--ink); --button-secondary-border-color:transparent;
}
body, .gradio-container { font-family:'Inter',system-ui,sans-serif !important; color:var(--ink) !important;
  background:
    radial-gradient(600px 420px at 8% 0%, rgba(34,211,238,.20), transparent 70%),
    radial-gradient(640px 480px at 95% 12%, rgba(124,92,255,.22), transparent 70%),
    linear-gradient(rgba(120,180,255,.045) 1px, transparent 1px) 0 0/46px 46px,
    linear-gradient(90deg, rgba(120,180,255,.045) 1px, transparent 1px) 0 0/46px 46px,
    var(--bg) fixed !important; }
.gradio-container { max-width:1200px !important; margin:auto !important; }
.gradio-container label span { color:var(--muted) !important; }
button[role="tab"] { color:var(--muted) !important; font-weight:600 !important; border:none !important; }
button[role="tab"][aria-selected="true"] { color:var(--cyan) !important; box-shadow:inset 0 -2px 0 var(--cyan) !important; }
.pg-head { padding:30px 4px 6px; } .pg-head h1 { font-size:34px; font-weight:700; margin:0; letter-spacing:-.6px; color:var(--ink); }
.pg-head h1 i { font-style:normal; color:var(--cyan); text-shadow:0 0 22px rgba(34,211,238,.6); }
.pg-head p { color:var(--muted); margin:8px 0 14px; font-size:15px; max-width:64ch; }
.pg-chips { display:flex; flex-wrap:wrap; gap:10px; } .pg-chips span { font-size:12px; padding:7px 13px; border-radius:99px; background:#0b1428; box-shadow:var(--neu-out); color:var(--muted); }
.pg-stats { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:8px 0 4px; }
.pg-tile { background:#0b1428; border-radius:18px; padding:16px 18px; box-shadow:var(--neu-out); }
.pg-tile .n { font-size:30px; font-weight:700; font-family:'JetBrains Mono',monospace; } .pg-tile .t { font-size:12px; color:var(--muted); margin-top:2px; }
.pg-tile.c .n{color:var(--cyan)} .pg-tile.b .n{color:var(--bad)} .pg-tile.s .n{color:var(--safe)}
.pg-panel { background:var(--glass); border:1px solid var(--edge); border-radius:22px; padding:22px; color:var(--ink);
  backdrop-filter:blur(20px) saturate(150%); -webkit-backdrop-filter:blur(20px) saturate(150%);
  box-shadow:0 18px 50px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.08); }
.pg-panel h3 { margin:0 0 4px; font-size:16px; font-weight:600; color:var(--ink); }
.pg-sub { color:var(--muted); font-size:13px; margin-bottom:14px; }
.pg-scanid { font-family:'JetBrains Mono',monospace; font-size:12px; color:var(--muted); margin-bottom:12px; }
.pg-url textarea, .pg-url input { font-family:'JetBrains Mono',monospace !important; font-size:15px !important; color:var(--ink) !important;
  background:#0a1224 !important; border:none !important; border-radius:16px !important; box-shadow:var(--neu-in) !important; padding:14px !important; }
.pg-url textarea:focus, .pg-url input:focus { box-shadow:var(--neu-in), 0 0 0 2px rgba(34,211,238,.55) !important; }
.pg-go { background:linear-gradient(135deg,#22d3ee,#6d5cff) !important; color:#04101c !important; border:none !important; font-weight:700 !important;
  min-height:48px; border-radius:16px !important; box-shadow:var(--neu-out), 0 0 26px rgba(34,211,238,.35) !important; }
.pg-verdict { border-radius:18px; padding:18px 20px; display:flex; gap:16px; align-items:center; border:1px solid; }
.pg-verdict .ico { font-size:36px; line-height:1; } .pg-verdict h2 { margin:0; font-size:25px; font-weight:700; color:var(--ink); }
.pg-verdict p { margin:4px 0 0; font-size:14px; color:var(--muted); }
.v-safe { background:rgba(52,245,160,.10); border-color:rgba(52,245,160,.5); box-shadow:0 0 34px rgba(52,245,160,.16); }
.v-warn { background:rgba(255,193,77,.10); border-color:rgba(255,193,77,.5); box-shadow:0 0 34px rgba(255,193,77,.16); }
.v-bad  { background:rgba(255,77,109,.12); border-color:rgba(255,77,109,.6); box-shadow:0 0 38px rgba(255,77,109,.24); }
.pg-anat { font-family:'JetBrains Mono',monospace; font-size:15px; background:#070d1c; color:rgba(178,199,235,.6); border-radius:14px; padding:14px 16px;
  overflow-x:auto; white-space:nowrap; margin-top:16px; box-shadow:var(--neu-in); }
.pg-anat .dom { color:#fff; font-weight:500; border-bottom:2px solid var(--safe); } .pg-anat .sub { color:var(--warn); }
.pg-anat .tld-bad { border-bottom-color:var(--bad); } .pg-anat .scheme-bad { color:var(--bad); }
.pg-note { color:var(--muted); font-size:13px; margin-top:8px; } .pg-note b { color:var(--ink); }
.pg-meter { margin-top:20px; } .pg-meter .lbl { display:flex; justify-content:space-between; font-size:13px; color:var(--muted); margin-bottom:8px; }
.pg-meter .bar { position:relative; height:12px; border-radius:99px; background:linear-gradient(90deg,#34f5a0 0%,#34f5a0 22%,#ffc14d 50%,#ff4d6d 78%,#ff4d6d 100%); box-shadow:var(--neu-in); }
.pg-meter .pin { position:absolute; top:-6px; width:6px; height:24px; background:#fff; border-radius:4px; transform:translateX(-3px); box-shadow:0 0 14px rgba(255,255,255,.9); }
.pg-meter .scale { display:flex; justify-content:space-between; font-size:11px; color:var(--muted); margin-top:6px; }
.pg-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:14px; margin-top:14px; }
.pg-stat { background:#0b1428; border-radius:14px; padding:11px 13px; box-shadow:var(--neu-out); }
.pg-stat .k { font-size:12px; color:var(--muted); } .pg-stat .v { font-size:18px; font-weight:600; margin-top:2px; font-family:'JetBrains Mono',monospace; }
.pg-sig { list-style:none; margin:10px 0 0; padding:0; display:grid; gap:8px; }
.pg-sig li { display:flex; gap:10px; align-items:flex-start; font-size:14px; padding:10px 13px; border-radius:13px; border:1px solid; }
.pg-sig .bad { background:rgba(255,77,109,.10); border-color:rgba(255,77,109,.45); } .pg-sig .warn { background:rgba(255,193,77,.09); border-color:rgba(255,193,77,.42); }
.pg-sig .good { background:rgba(52,245,160,.09); border-color:rgba(52,245,160,.40); }
.pg-empty { text-align:center; color:var(--muted); padding:60px 20px; } .pg-empty .big { font-size:42px; }
.pg-hist, .pg-tq { width:100%; border-collapse:collapse; font-size:13px; } .pg-hist td, .pg-tq td, .pg-tq th { padding:8px 6px; border-top:1px solid rgba(120,180,255,.12); text-align:left; }
.pg-tq th { color:var(--muted); font-weight:500; border-top:none; }
.pg-hist .u, .pg-tq .u { font-family:'JetBrains Mono',monospace; max-width:230px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.pill { font-size:12px; font-weight:600; padding:3px 10px; border-radius:99px; border:1px solid; white-space:nowrap; }
.pill.safe { background:rgba(52,245,160,.12); color:var(--safe); border-color:rgba(52,245,160,.45); }
.pill.warn { background:rgba(255,193,77,.12); color:var(--warn); border-color:rgba(255,193,77,.45); }
.pill.bad  { background:rgba(255,77,109,.14); color:var(--bad); border-color:rgba(255,77,109,.5); }
.pg-foot { color:var(--muted); font-size:12px; text-align:center; padding:24px 8px; }
@media (max-width:640px){ .pg-head h1{font-size:26px} .pg-verdict h2{font-size:20px} .pg-stats{grid-template-columns:1fr} .pg-panel{padding:16px} }
"""


EMPTY = """<div class="pg-panel pg-empty"><div class="big">🔎</div>
<h3>Paste a link to check it</h3>
<div class="pg-sub">You'll see a verdict, the part of the address that really matters, and the reasons behind the result.</div></div>"""


def _esc(x):
    return _h.escape(str(x), quote=True)


def _anatomy(url, tld, has_ip):
    p = urlparse(url)
    host = p.netloc.split("@")[-1].split(":")[0]
    rest = url[len(p.scheme) + 3:] if "://" in url else url
    tail = rest[len(p.netloc):] if rest.startswith(p.netloc) else ""
    labels = host.split(".")
    if has_ip or len(labels) < 2:
        sub, dom = "", host
    else:
        sub, dom = ".".join(labels[:-2]), ".".join(labels[-2:])
    scheme_cls = "scheme-bad" if p.scheme.lower() == "http" else ""
    dom_cls = "dom tld-bad" if (tld in RISKY_TLDS or has_ip) else "dom"
    parts = f'<span class="{scheme_cls}">{_esc(p.scheme)}://</span>'
    if "@" in p.netloc:
        parts += f'<span class="sub">{_esc(p.netloc.split("@")[0])}@</span>'
    if sub:
        parts += f'<span class="sub">{_esc(sub)}.</span>'
    parts += f'<span class="{dom_cls}">{_esc(dom)}</span>'
    if ":" in p.netloc.split("@")[-1]:
        parts += _esc(":" + p.netloc.split(":")[-1])
    parts += _esc(tail)
    note = (f'This link opens <b>{_esc(dom)}</b>. Anything in amber before it is a subdomain '
            f'the site owner can name freely.' if sub else
            f'This link opens <b>{_esc(dom)}</b>.')
    return f'<div class="pg-anat">{parts}</div><div class="pg-note">{note}</div>'


def _signals(f, url):
    host = urlparse(url).netloc.split("@")[-1].split(":")[0]
    s = []
    for b in brand_flags(url):
        s.append(("bad", f"Mentions <b>{_esc(b)}</b> but is not a {_esc(b)} domain. Possible brand impersonation."))
    if f.get("has_ip"): s.append(("bad", "The address is a raw IP number instead of a website name."))
    if "@" in urlparse(url).netloc: s.append(("bad", "Contains an @ sign, which can hide the real destination."))
    if "xn--" in host: s.append(("bad", "Uses punycode, which can imitate look-alike characters."))
    if f.get("tld") in RISKY_TLDS and not f.get("has_ip"):
        s.append(("warn", f"The .{_esc(f['tld'])} ending is frequently used by throwaway sites."))
    if not f.get("has_https"): s.append(("warn", "Not encrypted (HTTP). Legitimate login pages almost always use HTTPS."))
    n = int(f.get("suspicious_words", 0))
    if n: s.append(("warn", f"{n} sensitive keyword(s) such as login, verify or bank appear in the link."))
    if host.count(".") >= 3: s.append(("warn", "Many subdomain levels, a common way to disguise the real domain."))
    if host.count("-") >= 3: s.append(("warn", "Several hyphens in the domain name."))
    if f.get("url_length", 0) > 100: s.append(("warn", f"Very long link ({int(f['url_length'])} characters)."))
    if f.get("num_params", 0) >= 3: s.append(("warn", "Many query parameters."))
    if f.get("digits_count", 0) >= 8: s.append(("warn", "Unusually many digits."))
    if f.get("entropy", 0) > 4.6: s.append(("warn", "Random-looking characters in the link."))
    if not s: s.append(("good", "No structural red flags in this link."))
    elif f.get("has_https") and not any(k == "bad" for k, _ in s):
        s.append(("good", "Uses HTTPS. This only means the connection is encrypted, not that the site is trustworthy."))
    return s


def _tier(p):
    if p >= 0.75: return "bad", "🚨", "Likely phishing", "Do not enter passwords or personal details on this site."
    if p >= 0.5:  return "bad", "⚠️", "Probably phishing", "The model leans towards phishing. Avoid it unless you can verify the source."
    if p >= 0.25: return "warn", "🤔", "Uncertain, check before trusting", "Mixed signals. Open the official site directly rather than this link."
    return "safe", "✅", "Looks legitimate", "No strong phishing pattern found. Stay careful with unexpected messages."


def render_result(r):
    f, p = r["features"], r["phishing_probability"]
    cls, ico, title, sub = _tier(p)
    stats = [("Length", int(f["url_length"])), ("Dots", int(f["num_dots"])), ("Path levels", int(f["num_subdirs"])),
             ("Parameters", int(f["num_params"])), ("Digits", int(f["digits_count"])),
             ("Entropy", f"{f['entropy']:.2f}"), ("Ending", "IP" if f["has_ip"] else "." + _esc(f["tld"]))]
    grid = "".join(f'<div class="pg-stat"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in stats)
    sig = "".join(f'<li class="{k}"><span>{ {"bad":"⛔","warn":"⚠️","good":"✔️"}[k] }</span><span>{t}</span></li>' for k, t in _signals(f, r["url"]))
    return f"""
<div class="pg-panel">
  <div class="pg-scanid">Scan {hashlib.md5(r["url"].encode()).hexdigest()[:8]} · {datetime.now().strftime("%d %b %Y, %H:%M")}</div>
  <div class="pg-verdict v-{cls}"><div class="ico">{ico}</div><div><h2>{title}</h2><p>{sub}</p></div></div>
  <div class="pg-meter">
    <div class="lbl"><span>Phishing likelihood</span><b style="color:var(--ink)">{p:.0%}</b></div>
    <div class="bar"><div class="pin" style="left:{min(max(p,0.01),0.99)*100:.1f}%"></div></div>
    <div class="scale"><span>Legitimate</span><span>Uncertain</span><span>Phishing</span></div>
  </div>
  {_anatomy(r["url"], f["tld"], f["has_ip"])}
  <h3 style="margin-top:22px">What we noticed</h3>
  <ul class="pg-sig">{sig}</ul>
  <h3 style="margin-top:22px">Link measurements</h3>
  <div class="pg-grid">{grid}</div>
</div>"""


def render_history(hist):
    if not hist:
        return '<div class="pg-sub">Your checks will be listed here.</div>'
    rows = "".join(f'<tr><td class="u" title="{_esc(u)}">{_esc(u)}</td><td style="text-align:right"><span class="pill {c}">{_esc(t)}</span></td></tr>' for u, c, t in hist[:8])
    return f'<table class="pg-hist">{rows}</table>'


import hashlib

BRANDS = {"paypal": ["paypal.com"], "amazon": ["amazon.com", "amazon.in"], "google": ["google.com"],
          "microsoft": ["microsoft.com", "live.com", "office.com"], "apple": ["apple.com", "icloud.com"],
          "facebook": ["facebook.com"], "instagram": ["instagram.com"], "netflix": ["netflix.com"],
          "linkedin": ["linkedin.com"], "whatsapp": ["whatsapp.com"], "dhl": ["dhl.com"], "fedex": ["fedex.com"],
          "binance": ["binance.com"], "coinbase": ["coinbase.com"], "paytm": ["paytm.com"],
          "hdfc": ["hdfcbank.com"], "icici": ["icicibank.com"], "sbi": ["onlinesbi.sbi", "sbi.co.in"]}


def brand_flags(url):
    host = urlparse(url).netloc.split("@")[-1].split(":")[0].lower()
    out = []
    for b, legit in BRANDS.items():
        if b in url.lower() and not any(host == d or host.endswith("." + d) for d in legit):
            out.append(b)
    return out


def render_stats(hist):
    n = len(hist); bad = sum(1 for h in hist if h[1] == "bad"); ok = sum(1 for h in hist if h[1] == "safe")
    t = lambda c, v, l: f'<div class="pg-tile {c}"><div class="n">{v}</div><div class="t">{l}</div></div>'
    return '<div class="pg-stats">' + t("c", n, "Links checked this session") + t("b", bad, "Flagged as phishing") + t("s", ok, "Looked legitimate") + "</div>"


def frontend_predict(url, hist):
    hist = hist or []
    if not url or not str(url).strip():
        return EMPTY.replace("Paste a link to check it", "Enter a link first"), render_history(hist), render_stats(hist), hist
    try:
        r = predict_url_app(url)
        cls, _, title, _ = _tier(r["phishing_probability"])
        hist = [(r["url"], cls, title)] + [h for h in hist if h[0] != r["url"]][:499]
        return render_result(r), render_history(hist), render_stats(hist), hist
    except Exception as e:
        err = (f'<div class="pg-panel"><div class="pg-verdict v-bad"><div class="ico">❌</div><div>'
               f'<h2>We could not check that link</h2><p>{_esc(type(e).__name__)}: {_esc(e)}</p></div></div>'
               f'<div class="pg-note">Check the address for typos, or confirm the model files sit next to app.py.</div></div>')
        return err, render_history(hist), render_stats(hist), hist


def typosquat_variants(domain, limit=24):
    domain = domain.lower().strip().replace("https://", "").replace("http://", "").split("/")[0]
    if domain.startswith("www."): domain = domain[4:]
    if "." not in domain: return domain, []
    name, tld = domain.rsplit(".", 1)
    v = []
    v += [name[:i] + name[i + 1:] + "." + tld for i in range(len(name))]                                # missing letter
    v += [name[:i] + name[i + 1] + name[i] + name[i + 2:] + "." + tld for i in range(len(name) - 1)]    # swapped letters
    for a, b in [("o", "0"), ("l", "1"), ("i", "1"), ("e", "3"), ("a", "4"), ("m", "rn"), ("w", "vv")]:  # look-alike characters
        if a in name: v.append(name.replace(a, b, 1) + "." + tld)
    v += [name[:i] + "-" + name[i:] + "." + tld for i in range(1, len(name))][:3]                        # hyphen inserted
    v += [name + "." + t for t in ("net", "co", "org", "xyz", "top", "info", "tk") if t != tld]          # other endings
    v += [f"secure-{name}.{tld}", f"{name}-login.{tld}", f"{name}-verify.{tld}", f"my{name}.{tld}"]     # common add-ons
    seen, out = {domain}, []
    for x in v:
        if x not in seen and x.count(".") == 1 and len(x) > 4: seen.add(x); out.append(x)
    return domain, out[:limit]


def typosquat_scan(domain):
    if not domain or not domain.strip():
        return EMPTY.replace("Paste a link to check it", "Enter a domain to check")
    try:
        base, variants = typosquat_variants(domain)
        if not variants:
            return '<div class="pg-panel"><div class="pg-sub">Enter a domain such as example.com.</div></div>'
        rows = []
        for d in variants:
            p = predict_url_app("http://" + d)["phishing_probability"]
            rows.append((p, d))
        rows.sort(reverse=True)
        body = "".join(f'<tr><td class="u">{_esc(d)}</td><td>{p:.0%}</td><td><span class="pill {_tier(p)[0]}">{_esc(_tier(p)[2])}</span></td></tr>' for p, d in rows)
        return (f'<div class="pg-panel"><h3>Look-alike variants of {_esc(base)}</h3>'
                f'<div class="pg-sub">{len(rows)} possible typosquats, ranked by the model\'s phishing likelihood. '
                f'These are guesses at what an attacker might register. We do not check whether any of them exist.</div>'
                f'<table class="pg-tq"><tr><th>Domain</th><th>Risk</th><th>Verdict</th></tr>{body}</table></div>')
    except Exception as e:
        return f'<div class="pg-panel"><div class="pg-verdict v-bad"><div class="ico">❌</div><div><h2>Could not generate variants</h2><p>{_esc(e)}</p></div></div></div>'


EXAMPLES = ["https://www.wikipedia.org/wiki/Phishing",
            "http://secure-paypal-login.account-verify.tk/signin?id=8831",
            "http://192.168.10.5/bank/confirm-password",
            "https://github.com/anthropics"]

with gr.Blocks(title="PhishGuard AI", css=CSS, theme=gr.themes.Base()) as app:
    state = gr.State([])
    gr.HTML("""<div class="pg-head"><h1>🛡️ Phish<i>Guard</i> AI</h1>
      <p>Scan a link before you click it. Get a verdict, the real domain behind it, and the reasons, plus a finder for look-alike domains.</p>
      <div class="pg-chips"><span>Machine-learning URL analysis</span><span>Brand impersonation check</span><span>Typosquat finder</span></div></div>""")
    with gr.Tabs():
        with gr.Tab("URL scanner"):
            stats_out = gr.HTML(render_stats([]))
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=320):
                    gr.HTML('<div class="pg-panel" style="padding-bottom:8px"><h3>Scan a link</h3>'
                            '<div class="pg-sub">Nothing is opened or visited. Only the text of the link is analysed.</div></div>')
                    url_input = gr.Textbox(label="Web address", placeholder="https://example.com/login", lines=1, elem_classes="pg-url")
                    with gr.Row():
                        scan = gr.Button("Scan link", variant="primary", elem_classes="pg-go", scale=3)
                        gr.ClearButton([url_input], value="Clear", scale=1)
                    gr.Examples(EXAMPLES, inputs=url_input, label="Try an example")
                    gr.HTML('<div class="pg-panel" style="margin-top:10px"><h3>Recent scans</h3></div>')
                    history_out = gr.HTML(render_history([]))
                with gr.Column(scale=7, min_width=360):
                    result_out = gr.HTML(EMPTY)
            for trigger in (scan.click, url_input.submit):
                trigger(frontend_predict, [url_input, state], [result_out, history_out, stats_out, state], show_progress="minimal")
        with gr.Tab("Typosquat finder"):
            gr.HTML('<div class="pg-panel"><h3>Find look-alike domains</h3><div class="pg-sub">Enter a domain you own or want to protect. '
                    'We generate the misspellings and swaps attackers commonly register, and rate each one.</div></div>')
            dom_in = gr.Textbox(label="Domain", placeholder="example.com", lines=1, elem_classes="pg-url")
            dom_btn = gr.Button("Generate and rate variants", variant="primary", elem_classes="pg-go")
            dom_out = gr.HTML("")
            dom_btn.click(typosquat_scan, dom_in, dom_out, show_progress="minimal")
            dom_in.submit(typosquat_scan, dom_in, dom_out, show_progress="minimal")
    gr.HTML('<div class="pg-foot">PhishGuard AI estimates risk from the text of a link using a Random Forest model. '
            'It can be wrong, so verify important sites through official channels.</div>')

if __name__ == "__main__":
    # Colab needs share=True to give you a public link; locally/servers it is not needed.
    IN_COLAB = "google.colab" in sys.modules
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=IN_COLAB,
    )
