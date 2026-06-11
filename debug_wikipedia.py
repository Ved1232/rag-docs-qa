# debug_wikipedia.py — Run this to diagnose the Wikipedia issue
# Usage: python debug_wikipedia.py

import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

print("=" * 55)
print("Wikipedia API Debug")
print("=" * 55)

# Test 1 — Action API
print("\nTest 1: Action API (action=parse)")
try:
    params = {
        "action": "parse",
        "page": "Cloud computing",
        "prop": "text",
        "format": "json",
        "redirects": 1,
    }
    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params=params,
        headers=HEADERS,
        timeout=15
    )
    print(f"  Status code : {r.status_code}")
    print(f"  Content-Type: {r.headers.get('content-type', 'none')}")
    print(f"  Body length : {len(r.text)} chars")
    print(f"  Body preview: {r.text[:200]}")
except Exception as e:
    print(f"  Exception: {e}")

# Test 2 — REST v1 summary
print("\nTest 2: REST v1 summary API")
try:
    r = requests.get(
        "https://en.wikipedia.org/api/rest_v1/page/summary/Cloud_computing",
        headers=HEADERS,
        timeout=15
    )
    print(f"  Status code : {r.status_code}")
    print(f"  Content-Type: {r.headers.get('content-type', 'none')}")
    print(f"  Body length : {len(r.text)} chars")
    print(f"  Body preview: {r.text[:200]}")
except Exception as e:
    print(f"  Exception: {e}")

# Test 3 — Basic connectivity
print("\nTest 3: Basic Wikipedia connectivity")
try:
    r = requests.get("https://en.wikipedia.org", headers=HEADERS, timeout=15)
    print(f"  Status code : {r.status_code}")
    print(f"  Reachable   : YES")
except Exception as e:
    print(f"  Reachable   : NO — {e}")

print("\n" + "=" * 55)
