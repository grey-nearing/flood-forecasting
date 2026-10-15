with open('frontend/weather_sync.py', 'r') as f:
    content = f.read()

content = content.replace("    started = _utc_now_str()\n        proc", "    started = _utc_now_str()\n    proc")

with open('frontend/weather_sync.py', 'w') as f:
    f.write(content)

