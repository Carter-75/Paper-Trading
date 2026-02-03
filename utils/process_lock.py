
import os
import sys
import tempfile

class ProcessLock:
    def __init__(self, lock_file="bot.lock"):
        self.lock_file = lock_file
        self._lock_fd = None

    def acquire(self):
        """Try to acquire lock. Exit if fails."""
        if os.path.exists(self.lock_file):
            # Check if process is actually running
            try:
                with open(self.lock_file, 'r') as f:
                    pid = int(f.read().strip())
                
                # Check if PID exists (Windows)
                import ctypes
                kernel32 = ctypes.windll.kernel32
                process = kernel32.OpenProcess(0x1000, False, pid) # SYNCHRONIZE access
                if process:
                    kernel32.CloseHandle(process)
                    print(f"Bot already running (PID {pid}). Exiting.")
                    return False
                else:
                    # Stale lock
                    print("Found stale lock file. Overwriting.")
            except (ValueError, OSError):
                pass # Corrupt or unreadable, overwrite

        # Create lock
        try:
            with open(self.lock_file, 'w') as f:
                f.write(str(os.getpid()))
            return True
        except IOError:
            return False

    def release(self):
        try:
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
        except OSError:
            pass
