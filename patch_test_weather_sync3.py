import re

with open('frontend/tests/test_weather_sync.py', 'r') as f:
    content = f.read()

content = content.replace("('[ecmwf_hres] starting\n', 'Traceback (most recent call last):\nModuleNotFoundError: No module named foo\n')", "('[ecmwf_hres] starting\\n', 'Traceback (most recent call last):\\nModuleNotFoundError: No module named foo\\n')")
content = content.replace("('', 'some warning\n')", "('', 'some warning\\n')")
content = content.replace("('Another weather sync holds lock\n', '')", "('Another weather sync holds lock\\n', '')")

with open('frontend/tests/test_weather_sync.py', 'w') as f:
    f.write(content)

