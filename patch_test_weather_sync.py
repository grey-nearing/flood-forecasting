import re

with open('frontend/tests/test_weather_sync.py', 'r') as f:
    content = f.read()

# Replace all @mock.patch('subprocess.run') with @mock.patch('subprocess.Popen') for the relevant tests.
# And mock proc.poll(), proc.communicate(), proc.returncode, proc.kill, proc.wait
# Wait, let's just do a string replacement on the relevant methods.

def replacement(match):
    return match.group(0).replace("mock.patch('subprocess.run')", "mock.patch('subprocess.Popen')")

# Actually I can just write a script that updates those four tests completely
new_tests = """
    @mock.patch('subprocess.Popen')
    def test_run_sync_subprocess_executes_isolated_python(
        self, mock_popen: mock.MagicMock
    ) -> None:
        \"\"\"Verify CLI flags and correct virtualenv are passed down.\"\"\"
        with tempfile.TemporaryDirectory() as tmp_root:
            mock_proc = mock.MagicMock()
            mock_proc.poll.return_value = 0
            mock_proc.communicate.return_value = ('', '')
            mock_proc.returncode = 0
            mock_popen.return_value = mock_proc
            ws.run_sync_subprocess(
                python='fake_python',
                models=['ecmwf_hres', 'noaa_gfs'],
                force=True,
                root=tmp_root,
            )
            assert mock_popen.called
            cmd = mock_popen.call_args[0][0]
            assert cmd[:3] == ['fake_python', '-m', 'frontend.weather_sync']
            assert '--data-root' in cmd
            assert str(Path(tmp_root).resolve()) in cmd
            assert '--models' in cmd
            assert 'ecmwf_hres,noaa_gfs' in cmd
            assert '--force' in cmd
            expected_repo_root = str(Path(ws.__file__).resolve().parent.parent)
            assert mock_popen.call_args.kwargs['cwd'] == expected_repo_root

    @mock.patch('subprocess.Popen')
    def test_run_sync_subprocess_records_error_on_crash(
        self, mock_popen: mock.MagicMock
    ) -> None:
        \"\"\"Verify non-zero subprocess exit without fresh status writes error.\"\"\"
        with tempfile.TemporaryDirectory() as tmp_root:
            mock_proc = mock.MagicMock()
            mock_proc.poll.return_value = 1
            mock_proc.communicate.return_value = ('[ecmwf_hres] starting\\n', 'Traceback (most recent call last):\\nModuleNotFoundError: No module named foo\\n')
            mock_proc.returncode = 1
            mock_popen.return_value = mock_proc
            status = ws.run_sync_subprocess(python='fake_python', root=tmp_root)
            assert status['last_result'] == 'error'
            assert status['message'] == (
                'ModuleNotFoundError: No module named foo'
            )
            persisted = ws.read_sync_status(tmp_root)
            assert persisted['last_result'] == 'error'
            assert persisted['message'] == status['message']

            # Empty stderr falls back to exit code (never stdout log lines).
            mock_proc.communicate.return_value = ('', '')
            mock_proc.returncode = 2
            mock_proc.poll.return_value = 2
            Path(tmp_root, 'sync_status.json').unlink()
            status2 = ws.run_sync_subprocess(
                python='fake_python', root=tmp_root
            )
            assert status2['last_result'] == 'error'
            assert status2['message'] == 'Sync subprocess exited with code 2'

    @mock.patch('subprocess.Popen')
    def test_run_sync_subprocess_preserves_fresh_child_status(
        self, mock_popen: mock.MagicMock
    ) -> None:
        \"\"\"Verify child-written partial status is preserved on exit code 1.\"\"\"
        with tempfile.TemporaryDirectory() as tmp_root:
            status_file = Path(tmp_root) / 'sync_status.json'

            def write_child_partial(
                *_args: object, **_kwargs: object
            ) -> mock.MagicMock:
                status_file.write_text(
                    json.dumps(
                        {
                            'last_check_utc': '9999-12-31T23:59:59Z',
                            'last_result': 'partial',
                            'message': (
                                'Downloaded new run: ecmwf_aifs; failed:'
                                ' noaa_gfs: timeout'
                            ),
                        }
                    ),
                    encoding='utf-8',
                )
                res = mock.MagicMock()
                res.poll.return_value = 1
                res.returncode = 1
                res.communicate.return_value = ('', 'some warning\\n')
                return res

            mock_popen.side_effect = write_child_partial
            status = ws.run_sync_subprocess(python='fake_python', root=tmp_root)
            assert status['last_result'] == 'partial'
            assert 'Downloaded new run: ecmwf_aifs' in status['message']

    @mock.patch('subprocess.Popen')
    def test_run_sync_subprocess_reports_busy_when_status_stale_on_exit_zero(
        self, mock_popen: mock.MagicMock
    ) -> None:
        \"\"\"Verify exit 0 with stale sync_status.json reports busy, not stale.\"\"\"
        with tempfile.TemporaryDirectory() as tmp_root:
            status_file = Path(tmp_root) / 'sync_status.json'
            status_file.write_text(
                json.dumps(
                    {
                        'last_check_utc': '2020-01-01T00:00:00Z',
                        'last_result': 'updated',
                        'message': 'Downloaded new run: old',
                    }
                ),
                encoding='utf-8',
            )
            mock_proc = mock.MagicMock()
            mock_proc.poll.return_value = 0
            mock_proc.communicate.return_value = ('Another weather sync holds lock\\n', '')
            mock_proc.returncode = 0
            mock_popen.return_value = mock_proc
            status = ws.run_sync_subprocess(python='fake_python', root=tmp_root)
            assert status['last_result'] == 'busy'
            assert 'in progress' in status['message']

    @mock.patch('time.monotonic', side_effect=[0.0, 1.0, 100000.0])
    @mock.patch('subprocess.Popen')
    def test_run_sync_subprocess_timeout_returns_false(
        self, mock_popen: mock.MagicMock, mock_time: mock.MagicMock
    ) -> None:
        \"\"\"Verify process timeout kills proc and returns False cleanly.\"\"\"
        with tempfile.TemporaryDirectory() as tmp_root:
            mock_proc = mock.MagicMock()
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc
            status = ws.run_sync_subprocess(python='fake_python', root=tmp_root)
            assert status is False
            assert mock_proc.kill.called
            assert mock_proc.wait.called
"""

content = re.sub(
    r"    @mock\.patch\('subprocess\.run'\)\n    def test_run_sync_subprocess_executes_isolated_python.*?(?=    def test_get_sync_status_auto_update_false)",
    new_tests,
    content,
    flags=re.DOTALL
)

with open('frontend/tests/test_weather_sync.py', 'w') as f:
    f.write(content)

