import re

with open('frontend/weather_sync.py', 'r') as f:
    content = f.read()

# Fix return type
content = content.replace(
    ') -> dict[str, Any]:\n    """Run one sync in a separate process',
    ') -> dict[str, Any] | bool:\n    """Run one sync in a separate process'
)

new_run = """    started = _utc_now_str()
    import time
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=_REPO_ROOT,
    )
    start_time = time.monotonic()
    while proc.poll() is None:
        if time.monotonic() - start_time > SUBPROCESS_TIMEOUT_S:
            proc.kill()
            proc.wait()
            logger.error("Weather sync subprocess timed out after %s seconds", SUBPROCESS_TIMEOUT_S)
            return False
        time.sleep(0.5)
    stdout, stderr = proc.communicate()
    _append_log(
        resolved_root,
        f'\\n=== {started} {" ".join(cmd)} (exit {proc.returncode})\\n'
        f'{stdout}{stderr}',
    )
    status = read_sync_status(resolved_root)
    if (status.get('last_check_utc') or '') < started:
        if proc.returncode == 0:
            status.update(
                {
                    'last_check_utc': _utc_now_str(),
                    'check_interval_minutes': CHECK_INTERVAL_MINUTES,
                    'last_result': 'busy',
                    'message': 'Another weather sync is in progress.',
                }
            )
        else:
            stderr_lines = (stderr or '').strip().splitlines()
            detail = (
                stderr_lines[-1]
                if stderr_lines
                else f'Sync subprocess exited with code {proc.returncode}'
            )"""

content = re.sub(
    r'    started = _utc_now_str\(\)\n    proc = subprocess\.run\(.*?if proc\.returncode == 0:\n.*?\n        else:\n            stderr_lines = \(proc\.stderr or .*\).strip\(\).splitlines\(\)\n            detail = \(\n                stderr_lines\[-1\]\n                if stderr_lines\n                else f.Sync subprocess exited with code \{proc\.returncode\}.\n            \)',
    new_run,
    content,
    flags=re.DOTALL
)

with open('frontend/weather_sync.py', 'w') as f:
    f.write(content)

