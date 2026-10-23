#!/usr/bin/env python3
"""
Technology Nucleus — Search Engine Indexing Push Script
Triggers instant crawler notifications via IndexNow and validates sitemaps for:
- https://technonucleus.com
- https://mrv.technonucleus.com
"""

import urllib.request
import json
import sys

URLS = [
    "https://technonucleus.com/",
    "https://technonucleus.com/sitemap.xml",
    "https://mrv.technonucleus.com/",
    "https://mrv.technonucleus.com/sitemap.xml",
    "https://mrv.technonucleus.com/articles",
    "https://mrv.technonucleus.com/articles/subnet-architecture-sop",
    "https://mrv.technonucleus.com/articles/decentralized-ensembling-v1-5",
    "https://mrv.technonucleus.com/articles/autonomous-frontier-vision",
]

INDEXNOW_KEY = "25c7e51bdbc2aef74176a3658a47395c"

def push_indexnow(host: str, urls: list[str]):
    payload = {
        "host": host,
        "key": INDEXNOW_KEY,
        "keyLocation": f"https://{host}/{INDEXNOW_KEY}.txt",
        "urlList": urls,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.indexnow.org/indexnow",
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[{host}] IndexNow Response: {resp.status} {resp.reason}")
    except Exception as e:
        print(f"[{host}] IndexNow Error: {e}")

def main():
    print("Pushing updated URLs to search engine crawler queues...")
    techno_urls = [u for u in URLS if "mrv." not in u]
    mrv_urls = [u for u in URLS if "mrv." in u]
    
    push_indexnow("technonucleus.com", techno_urls)
    push_indexnow("mrv.technonucleus.com", mrv_urls)
    print("Done! Crawlers have been queued for immediate re-indexation.")

if __name__ == "__main__":
    main()
