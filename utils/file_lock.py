
import os
import time

class FileLock:
    """
    A simple cross-platform file locking mechanism using a sidecar .lock file.
    Ensures safe concurrent access to the target file by multiple processes.
    """
    def __init__(self, target_file_path, timeout=10):
        self.lock_file = target_file_path + ".lock"
        self.timeout = timeout
        self.fd = None

    def __enter__(self):
        start_time = time.time()
        while True:
            try:
                # O_CREAT | O_EXCL ensures atomic creation. Fails if file exists.
                self.fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                return self
            except OSError:
                # Lock file exists
                if time.time() - start_time > self.timeout:
                    # Timeout reached
                    raise TimeoutError(f"Could not acquire lock for {self.lock_file} (Timeout {self.timeout}s)")
                time.sleep(0.1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fd:
            os.close(self.fd)
            try:
                os.remove(self.lock_file)
            except OSError:
                pass
