"""Request a Python 3.14 stack dump from a running process."""
import argparse
import sys
import textwrap
from pathlib import Path


REMOTE_SCRIPT = """
import os
import sys
import threading
import traceback
from pathlib import Path

output = Path(f"/tmp/batchbpe-stacks-{os.getpid()}.txt")
with output.open("a", encoding="utf-8") as stream:
    stream.write("\\n=== remote inspection ===\\n")
    for thread_id, frame in sys._current_frames().items():
        thread = next((item for item in threading.enumerate() if item.ident == thread_id), None)
        name = thread.name if thread is not None else "unknown"
        stream.write(f"\\n--- thread {thread_id} ({name}) ---\\n")
        traceback.print_stack(frame, file=stream)
"""


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("pid", type=int, help="PID of the CPython 3.14 process to inspect")
args = parser.parse_args()

payload = Path("/tmp/batchbpe-remote-inspection-payload.py")
payload.write_text(textwrap.dedent(REMOTE_SCRIPT), encoding="utf-8")
sys.remote_exec(args.pid, payload)
print(f"Inspection requested. Read /tmp/batchbpe-stacks-{args.pid}.txt shortly.")
