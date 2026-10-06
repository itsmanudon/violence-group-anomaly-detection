"""Per-browser revision guards for delayed preview and inference responses."""

from threading import Lock


class SessionRevisions:
    def __init__(self):
        self.latest = {}
        self.lock = Lock()

    def observe(self, session: str | None, revision: int) -> bool:
        if revision == 0:
            return True
        if not session or type(revision) is not int or revision < 0:
            raise ValueError("Invalid browser session/input revision")
        with self.lock:
            previous = self.latest.get(session, 0)
            if revision < previous:
                return False
            self.latest[session] = revision
            return True

    def current(self, session: str | None, revision: int) -> bool:
        if revision == 0:
            return True
        with self.lock:
            return self.latest.get(session) == revision
