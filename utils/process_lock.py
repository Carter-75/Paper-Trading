import os

# --- Encoding-safe printing (avoid UnicodeEncodeError on some Windows services) ---
def _safe_print(msg: str):
    try:
        print(msg)
    except Exception:
        try:
            import sys
            sys.stdout.buffer.write((str(msg) + "\n").encode("utf-8", errors="replace"))
        except Exception:
            pass


def _pid_looks_like_bot(pid: int) -> bool:
    """Best-effort check: is PID a python.exe running runner.py?"""
    try:
        import subprocess
        # WMIC is deprecated but widely present; works without extra deps
        out = subprocess.check_output(
            f'wmic process where "ProcessId={pid}" get CommandLine /value',
            shell=True,
            stderr=subprocess.DEVNULL,
        )
        cmd = out.decode(errors="ignore").lower()
        return ("python" in cmd) and ("runner.py" in cmd)
    except Exception:
        return False


class ProcessLock:
    def __init__(self, lock_file: str = "bot.lock"):
        self.lock_file = lock_file

    def acquire(self, force_kill: bool = True) -> bool:
        """Acquire a PID lockfile.

        If a lock exists and the PID is alive:
        - if it looks like *this bot* (python runner.py):
            - force_kill=True: terminate it and take over
            - force_kill=False: return False
        - otherwise treat it as stale and overwrite.
        """
        if os.path.exists(self.lock_file):
            try:
                with open(self.lock_file, "r") as f:
                    content = f.read().strip()
                pid = int(content) if content else -1

                # If lock file already contains our PID, treat as ours/stale and continue
                if pid == os.getpid():
                    _safe_print('[LOCK] Lock file contains current PID; continuing.')
                    pid = -1

                if pid > 0:
                    import ctypes
                    kernel32 = ctypes.windll.kernel32
                    process = kernel32.OpenProcess(0x0400 | 0x0001, False, pid)
                    if process:
                        try:
                            if not _pid_looks_like_bot(pid):
                                _safe_print(f"[LOCK] PID {pid} in lock file is not runner.py; treating as stale.")
                            else:
                                if force_kill:
                                    _safe_print(f"[LOCK] Existing bot detected (PID {pid}). Force-killing...")
                                    import subprocess
                                    subprocess.run(
                                        f"taskkill /F /PID {pid}",
                                        shell=True,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL,
                                    )
                                    _safe_print("[LOCK] Previous process terminated. Taking over.")
                                else:
                                    _safe_print(f"[LOCK] Bot already running (PID {pid}). Exiting.")
                                    return False
                        finally:
                            kernel32.CloseHandle(process)
                    else:
                        _safe_print("[LOCK] Found stale lock file. Overwriting.")
            except Exception as e:
                _safe_print(f"[LOCK] Lock file corrupt or check failed ({e}). Overwriting.")

        # Create/overwrite lock
        try:
            with open(self.lock_file, "w") as f:
                f.write(str(os.getpid()))
            return True
        except IOError:
            return False

    def release(self) -> None:
        try:
            if os.path.exists(self.lock_file):
                try:
                    with open(self.lock_file, "r") as f:
                        pid = int(f.read().strip())
                    if pid == os.getpid():
                        os.remove(self.lock_file)
                except Exception:
                    pass
        except OSError:
            pass
