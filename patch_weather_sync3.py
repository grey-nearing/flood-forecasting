with open('frontend/weather_sync.py', 'r') as f:
    content = f.read()

# Fix import time
content = content.replace("import time\n", "")
content = "import time\n" + content

# Fix Line too long
content = content.replace(
    'logger.error("Weather sync subprocess timed out after %s seconds", SUBPROCESS_TIMEOUT_S)',
    'logger.error(\n                "Weather sync subprocess timed out after %s seconds",\n                SUBPROCESS_TIMEOUT_S,\n            )'
)

with open('frontend/weather_sync.py', 'w') as f:
    f.write(content)

