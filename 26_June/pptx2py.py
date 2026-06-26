#!/usr/bin/env python3
"""
pptx2py.py  —  Store a .pptx (or any file) inside a self-contained .py file,
and get the original back byte-for-byte by running that .py file.

USAGE
    Encode:  python pptx2py.py mydeck.pptx
                 -> writes mydeck.py  (the deck, embedded as base64)

    Restore: python mydeck.py
                 -> writes mydeck.pptx  (identical to the original)

The generated .py is portable: it needs only Python's standard library,
no python-pptx, no third-party packages.
"""
import base64
import hashlib
import os
import sys
import textwrap

# Template for the self-extracting file we generate.
# Doubled braces {{ }} are literal; single {name}/{b64}/{sha}/{data} get filled in.
TEMPLATE = '''#!/usr/bin/env python3
"""
Self-extracting archive of "{name}".
Run this file to recreate the original, byte-for-byte:

    python {pyname}

Or import it and call extract("some/other/name.pptx").
"""
import base64, hashlib, os, sys

ORIGINAL_FILENAME = "{name}"
SHA256 = "{sha}"

# --- original file, base64-encoded ---
DATA = (
{data})
# --- end of data ---


def extract(path=ORIGINAL_FILENAME, verify=True):
    raw = base64.b64decode(DATA)
    if verify:
        got = hashlib.sha256(raw).hexdigest()
        if got != SHA256:
            raise ValueError("checksum mismatch: data is corrupted")
    with open(path, "wb") as f:
        f.write(raw)
    return path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else ORIGINAL_FILENAME
    if os.path.exists(out):
        ans = input(f'"{{out}}" already exists. Overwrite? [y/N] ').strip().lower()
        if ans != "y":
            print("Aborted.")
            sys.exit(0)
    extract(out)
    print(f"Restored {{out}} ({{os.path.getsize(out)}} bytes)")
'''


def encode(src_path, py_path=None):
    name = os.path.basename(src_path)
    if py_path is None:
        py_path = os.path.splitext(src_path)[0] + ".py"

    with open(src_path, "rb") as f:
        raw = f.read()

    sha = hashlib.sha256(raw).hexdigest()
    b64 = base64.b64encode(raw).decode("ascii")

    # Wrap the base64 into short, quoted lines so the .py stays readable
    # and no single line is absurdly long.
    lines = textwrap.wrap(b64, 76)
    data_block = "\n".join('    "{}"'.format(ln) for ln in lines)

    out = TEMPLATE.format(
        name=name,
        pyname=os.path.basename(py_path),
        sha=sha,
        data=data_block,
    )
    with open(py_path, "w") as f:
        f.write(out)
    return py_path, len(raw), len(out)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src = sys.argv[1]
    py_path = sys.argv[2] if len(sys.argv) > 2 else None
    out, src_bytes, py_bytes = encode(src, py_path)
    print(f"Wrote {out}")
    print(f"  source : {src_bytes:,} bytes")
    print(f"  python : {py_bytes:,} bytes  (+{py_bytes / src_bytes - 1:.0%})")


if __name__ == "__main__":
    main()
