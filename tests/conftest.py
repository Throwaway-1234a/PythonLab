from pathlib import Path

import pytest


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    docs = {
        "moby_dick.txt": "Call me Ishmael. Some years ago never mind how long precisely. The great white whale sailed the ocean.",
        "alice.txt": "Alice was beginning to get very tired of sitting by her sister on the bank. Down the rabbit hole she went.",
        "hamlet.txt": "To be or not to be that is the question. The ghost appeared upon the castle platform at night.",
    }
    for filename, text in docs.items():
        (tmp_path / filename).write_text(text, encoding="utf-8")
    return tmp_path
