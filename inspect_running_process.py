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
import __main__
from pathlib import Path

output = Path(f"/tmp/batchbpe-stacks-{os.getpid()}.txt")
with output.open("a", encoding="utf-8") as stream:
    stream.write("\\n=== remote inspection ===\\n")
    dataset = getattr(__main__, "finewebedu_10b_stream", None)
    splits = getattr(getattr(dataset, "info", None), "splits", None) or {}
    split = splits.get("train")
    total = getattr(split, "num_examples", None)
    stream.write(f"declared train documents: {total if total is not None else 'unknown'}\\n")
    for thread_id, frame in sys._current_frames().items():
        thread = next((item for item in threading.enumerate() if item.ident == thread_id), None)
        name = thread.name if thread is not None else "unknown"
        stream.write(f"\\n--- thread {thread_id} ({name}) ---\\n")
        probe = frame
        while probe is not None:
            locals_ = probe.f_locals
            if probe.f_code.co_name == "_iter_chunk_arrays":
                text = locals_.get("text")
                stream.write(f"input document characters: {len(text) if text is not None else 'unknown'}\\n")
            elif probe.f_code.co_name == "from_chunk_iter":
                completed = sum(locals_.get("chunk_counts", ()))
                buffered = len(locals_.get("buf", ()))
                stream.write(
                    f"documents written: {completed}; "
                    f"current shard buffer: {buffered}\\n"
                )
            elif probe.f_code.co_name == "_encode_chunk_core":
                stream.write(
                    "remaining token IDs: "
                    f"{locals_.get('len_chunk', 'unknown')} "
                    f"(base merges: {len(locals_['self'].merges)})\\n"
                )
            probe = probe.f_back
        traceback.print_stack(frame, file=stream)
"""


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("pid", type=int, help="PID of the CPython 3.14 process to inspect")
args = parser.parse_args()

payload = Path(f"/tmp/batchbpe-remote-inspection-payload-{args.pid}.py")
payload.write_text(textwrap.dedent(REMOTE_SCRIPT), encoding="utf-8")
sys.remote_exec(args.pid, payload)
print(f"Inspection requested. Read /tmp/batchbpe-stacks-{args.pid}.txt shortly.")
