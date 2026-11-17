from pathlib import Path
import sys
import traceback
import nbformat
from nbclient import NotebookClient

notebook_path = Path(sys.argv[1]).resolve()
status_path = Path(sys.argv[2]).resolve()
working_directory = Path(sys.argv[3]).resolve()
status_path.write_text('RUNNING\n')
try:
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(notebook, timeout=7200, kernel_name='python3',
                            resources={'metadata': {'path': str(working_directory)}}, allow_errors=False)
    client.execute()
    nbformat.write(notebook, notebook_path)
    status_path.write_text('SUCCESS\n')
    print(f'EXECUTION_SUCCESS {notebook_path}', flush=True)
except Exception as exc:
    status_path.write_text('FAILED: ' + repr(exc) + '\n' + traceback.format_exc())
    print(f'EXECUTION_FAILED {notebook_path}: {exc!r}', flush=True)
    raise
