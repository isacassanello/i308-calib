import threading

from i308_calib.calib.tool_base import detect_checkerboard


class CheckerboardDetector:
    """
    Runs checkerboard detection in a background thread so the UI
    loop never blocks on the (slow) detection step.

    Latest-frame semantics: if a new frame is submitted while a
    detection is still running, the worker will pick up the newest
    frame as soon as it finishes the current one.  Stale frames are
    dropped, never queued.
    """

    def __init__(self, args):
        self.args = args

        self._thread = None
        self._running = False

        self._condition = threading.Condition()
        self._pending_frame = None
        self._busy = False
        self._result = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self):
        with self._condition:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._thread.start()

    def stop(self):
        with self._condition:
            if not self._running:
                return
            self._running = False
            self._condition.notify_all()

        if self._thread is not None:
            self._thread.join()
            self._thread = None

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def submit(self, frame):
        """Non-blocking.  Stores the latest frame to be detected."""
        with self._condition:
            self._pending_frame = frame
            self._condition.notify_all()

    def get_result(self):
        """Non-blocking.  Returns the latest completed detection (or None)."""
        with self._condition:
            return self._result

    def is_busy(self):
        with self._condition:
            return self._busy

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _worker_loop(self):
        while True:
            with self._condition:
                # wait for a pending frame or shutdown
                while self._running and self._pending_frame is None:
                    self._condition.wait()

                if not self._running:
                    return

                frame = self._pending_frame
                self._pending_frame = None
                self._busy = True

            # run detection outside the lock
            try:
                detection = detect_checkerboard(self.args, frame)
            except Exception as ex:
                print(f"error detecting checkerboard: {ex}")
                detection = None

            with self._condition:
                self._busy = False
                if detection is not None:
                    self._result = detection


class StereoCheckerboardDetector:
    """
    Wraps two CheckerboardDetector instances (left/right) so the
    stereo script can start / stop / submit / read results with a
    single object.
    """

    def __init__(self, args):
        self._left = CheckerboardDetector(args)
        self._right = CheckerboardDetector(args)

    def start(self):
        self._left.start()
        self._right.start()

    def stop(self):
        self._left.stop()
        self._right.stop()

    def submit(self, left_frame, right_frame):
        """Non-blocking.  Stores the latest left/right frames to be detected."""
        self._left.submit(left_frame)
        self._right.submit(right_frame)

    def get_results(self):
        """Non-blocking.  Returns (left_detection, right_detection)."""
        return self._left.get_result(), self._right.get_result()
