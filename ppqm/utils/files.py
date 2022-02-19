import tempfile
import weakref as _weakref
from pathlib import Path


class WorkDir(tempfile.TemporaryDirectory):
    """ Like TemporaryDirectory, with the possiblity of keeping log files for debug"""

    def __init__(self, suffix=None, prefix=None, dir=None, keep=False):
        # super().__init__(suffix=suffix, dir=dir, prefix=prefix)
        self.keep_directory = keep
        self.name = tempfile.mkdtemp(suffix, prefix, dir)
        if not keep:
            self._finalizer = _weakref.finalize(
                self,
                super()._cleanup,
                self.name,
                warn_message="Implicitly cleaning up {!r}".format(self),
            )

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self.keep_directory:
            super().__exit__(exc_type, exc_val, exc_tb)

    def get_path(self):
        return Path(self.name)
