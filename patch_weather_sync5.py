with open('frontend/weather_sync.py', 'r') as f:
    content = f.read()

content = content.replace("import time\n", "")
content = content.replace("import sys", "import sys\nimport time")

with open('frontend/weather_sync.py', 'w') as f:
    f.write(content)

