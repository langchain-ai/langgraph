"""QA Verification Script - Objective Testing"""
import httpx
import json
import time

base = "http://127.0.0.1:2024"

def post_json(path, payload):
    r = httpx.post(f"{base}{path}", json=payload, timeout=30.0)
    print(f"\n{'='*60}")
    print(f"Request: {path}")
    print(f"Status: {r.status_code}")
    print("Response:")
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text[:500])  # Limit output for streams
    print('='*60)
    return r.status_code

both_wait = {
    "assistant_id": "agent",
    "input": {"messages": [{"role": "user", "content": "Hello"}]},
    "config": {"configurable": {"thread_id": "tid_both_wait"}},
    "context": {"user_id": "u1", "request_id": "r1"}
}

both_stream = {
    "assistant_id": "agent",
    "input": {"messages": [{"role": "user", "content": "Hello"}]},
    "config": {"configurable": {"thread_id": "tid_both_stream"}},
    "context": {"user_id": "u1", "request_id": "r1"}
}

cfg_only = {
    "assistant_id": "agent",
    "input": {"messages": [{"role": "user", "content": "Hello"}]},
    "config": {"configurable": {"thread_id": "tid_cfg"}}
}

ctx_only = {
    "assistant_id": "agent",
    "input": {"messages": [{"role": "user", "content": "Hello"}]},
    "context": {"thread_id": "tid_ctx"}
}

print("QA VERIFICATION - OBJECTIVE TEST RESULTS")
print("="*60)

s1 = post_json("/runs/wait", both_wait)
time.sleep(0.5)
s2 = post_json("/runs/stream", both_stream)
time.sleep(0.5)
s3 = post_json("/runs/wait", cfg_only)
time.sleep(0.5)
s4 = post_json("/runs/wait", ctx_only)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"wait both context+config: {s1}")
print(f"stream both context+config: {s2}")
print(f"wait config only: {s3}")
print(f"wait context only: {s4}")
print("="*60)

print("\nVERDICT")
if s1 == 200 and s2 == 200 and s3 == 200 and s4 == 200:
    print("fixed")
else:
    print("not fixed")
print("="*60)

