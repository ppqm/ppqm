import copy
import json
import pickle
import tempfile
import weakref as _weakref
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np


class WorkDir(tempfile.TemporaryDirectory):
    """Like TemporaryDirectory, with the possiblity of keeping log files for debug"""

    def __init__(
        self,
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
        dir: Optional[Union[str, Path]] = None,
        keep: bool = False,
    ) -> None:
        self.keep_directory = keep
        self.name = tempfile.mkdtemp(suffix, prefix, dir)
        if not keep:
            self._finalizer = _weakref.finalize(
                self,
                super()._cleanup,  # type: ignore
                self.name,
                warn_message="Implicitly cleaning up {!r}".format(self),
            )

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self.keep_directory:
            super().__exit__(exc_type, exc_val, exc_tb)

    def get_path(self) -> Path:
        return Path(self.name)


def save_obj(name: str, obj: Any) -> None:
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name: str) -> Any:
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


def save_array(arr: np.ndarray) -> str:
    s = StringIO()
    np.savetxt(s, arr)
    return s.getvalue()


def load_array(txt: str) -> np.ndarray:
    s = StringIO(txt)
    arr = np.loadtxt(s)
    return arr


def str_json(dictionary: Dict, indent: int = 4, translate_types: bool = False) -> str:
    if translate_types:
        dictionary = json_friendly(dictionary)
    return json.dumps(dictionary, indent=indent)


def save_json(name: Path, obj: dict, indent: int = 4, translate_types: bool = False) -> None:
    """Save dictionary as a JSON file.

    translate_types: Change instances types of dictionary values to make it
    json friendly

    """

    if translate_types:
        obj = json_friendly(obj)

    if not name.suffix == ".json":
        name = name.with_suffix(".json")

    with open(name, "w") as f:
        json.dump(obj, f, indent=indent)


def json_friendly(dictionary: dict) -> dict:
    """Change the types of a dictionary to make it JSON friendly"""

    dictionary = copy.deepcopy(dictionary)

    for key, value in dictionary.items():

        if isinstance(value, dict):
            value = json_friendly(value)
            dictionary[key] = value

        elif isinstance(value, np.ndarray):
            value = value.tolist()
            dictionary[key] = value

    return dictionary


def load_json(name: Path) -> dict:

    if not name.suffix == ".json":
        name = name.with_suffix(".json")

    with open(name, "r") as f:
        content: dict = json.loads(f.read())

    return content
