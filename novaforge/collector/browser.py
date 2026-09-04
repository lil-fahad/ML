from __future__ import annotations
import hashlib, re, urllib.parse
from dataclasses import dataclass, asdict

PRIVATE_PREFIXES=("localhost","127.","0.0.0.0","::1","169.254.","10.","192.168.")

def public_web_url(url: str) -> bool:
    try:
        p=urllib.parse.urlsplit(url); host=(p.hostname or "").lower()
        if p.scheme not in {"http","https"}: return False
        if any(host==x or host.startswith(x) for x in PRIVATE_PREFIXES): return False
        if re.match(r"^172\.(1[6-9]|2\d|3[01])\.",host): return False
        return True
    except Exception: return False

@dataclass
class EvidenceRecord:
    url: str
    title: str
    text: str
    quality: float
    relevance: float
    sha256: str

async def collect_topic(topic: str, max_pages: int = 10):
    try:
        from playwright.async_api import async_playwright
        import trafilatura
    except Exception as e:
        raise RuntimeError("Install browser extras: pip install -e .[browser] and playwright install chromium") from e
    query_url="https://www.bing.com/search?q="+urllib.parse.quote_plus(topic)
    records=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True)
        ctx=await browser.new_context()
        page=await ctx.new_page()
        await page.goto(query_url,wait_until="domcontentloaded")
        links=await page.locator("a[href]").evaluate_all("els => els.map(a => a.href)")
        seen=set()
        for raw in links:
            if len(records)>=max_pages: break
            if not public_web_url(raw): continue
            host=urllib.parse.urlsplit(raw).hostname or ""
            if host.endswith("bing.com") or raw in seen: continue
            seen.add(raw)
            try:
                await page.goto(raw,wait_until="domcontentloaded",timeout=20000)
                html=await page.content(); title=await page.title()
                text=(trafilatura.extract(html,url=page.url,favor_precision=True) or "").strip()
                if len(text)<100: continue
                norm=" ".join(text.split()); sha=hashlib.sha256(norm.encode()).hexdigest()
                q=min(1.0,0.3+len(text)/20000)
                toks=set(topic.lower().split()); page_toks=set(text.lower().split()); rel=len(toks & page_toks)/max(1,len(toks))
                records.append(EvidenceRecord(page.url,title,text,q,rel,sha))
            except Exception: continue
        await browser.close()
    return [asdict(r) for r in records]
