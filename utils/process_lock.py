
import os
import sys
import tempfile

class ProcessLock:
    def __init__(self, lock_file="bot.lock"):
        self.lock_file = lock_file
        self._lock_fd = None

    def acquire(self, force_kill=True):
        """
        Try to acquire lock. 
        If force_kill=True, it will KILL the existing process and take over.
        """
        if os.path.exists(self.lock_file):
            # Check if process is actually running
            try:
                with open(self.lock_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        pid = int(content)
                    else:
                        pid = -1
                
                if pid > 0:
                    # Check if PID exists (Windows)
                    import ctypes
                    kernel32 = ctypes.windll.kernel32
                    # 0x0400 = PROCESS_QUERY_INFORMATION, 0x0001 = PROCESS_TERMINATE
                    process = kernel32.OpenProcess(0x0400 | 0x0001, False, pid)
                    
                    if process:
                        if force_kill:
                            print(f"⚠️  ZOMBIE DETECTED (PID {pid}). EXECUTING ORDER 66... 🔫")
                            import subprocess
                            subprocess.run(f"taskkill /F /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            kernel32.CloseHandle(process)
                            print(f"✅ Target destroyed. Taking over.")
                        else:
                            kernel32.CloseHandle(process)
                            print(f"Bot already running (PID {pid}). Exiting.")
                            return False
                    else:
                        # Stale lock
                        print("Found stale lock file. Overwriting.")
            except (ValueError, OSError, ImportError) as e:
                print(f"Lock file corrupt or checking failed ({e}). Overwriting.")

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
                # Only delete if it's OUR lock
                try:
                    with open(self.lock_file, 'r') as f:
                        pid = int(f.read().strip())
                    if pid == os.getpid():
                        os.remove(self.lock_file)
                except:
                    pass
        except OSError:
            pass
