with open('frontend/weather_sync.py', 'r') as f:
    content = f.read()

content = content.replace("f'\n=== {started}", "f'\\n=== {started}").replace("(exit {proc.returncode})\n'", "(exit {proc.returncode})\\n'")

with open('frontend/weather_sync.py', 'w') as f:
    f.write(content)

