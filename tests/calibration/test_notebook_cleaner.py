import json

from pyccapt.calibration.tutorials import notebook_cleaner


def test_clean_notebook_content_strips_outputs_and_widget_metadata():
    notebook = {
        "metadata": {"widgets": {"state": "large"}},
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 7,
                "outputs": [{"output_type": "stream", "text": "hi"}],
                "metadata": {
                    "collapsed": True,
                    "ExecuteTime": {"start_time": "x"},
                    "jupyter": {"outputs_hidden": True, "source_hidden": True},
                },
                "source": ["print('hi')"],
            }
        ],
    }

    cleaned, changed = notebook_cleaner.clean_notebook_content(notebook)

    assert changed is True
    assert cleaned["metadata"] == {}
    assert cleaned["cells"][0]["execution_count"] is None
    assert cleaned["cells"][0]["outputs"] == []
    assert cleaned["cells"][0]["metadata"] == {}


def test_clean_notebook_file_updates_json_in_place(tmp_path):
    notebook_path = tmp_path / "demo.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "metadata": {},
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": 3,
                        "outputs": [{"output_type": "execute_result", "data": {"text/plain": ["1"]}}],
                        "metadata": {},
                        "source": ["1"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    changed = notebook_cleaner.clean_notebook_file(notebook_path)
    cleaned = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert changed is True
    assert cleaned["cells"][0]["execution_count"] is None
    assert cleaned["cells"][0]["outputs"] == []
