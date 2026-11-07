"""Run a command while holding the machine's one GPU slot, so two jobs never share MPS.

usage: python gpulock.py [--wait-msg] -- cmd args...

Blocks until no other gpulock'd command is running, then runs cmd and releases the slot on exit
(the kernel drops the lock if this process dies). The lock file is /tmp/striate-gpu.lock.
"""

import fcntl, os, subprocess, sys, time

LOCK = "/tmp/striate-gpu.lock"


def main():
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    if not argv:
        sys.exit(__doc__)
    with open(LOCK, "a+") as f:
        t0 = time.time()
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.seek(0)
            print(f"[gpulock] waiting for: {f.read().strip() or '?'}", file=sys.stderr, flush=True)
            fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0); f.truncate(); f.write(f"pid {os.getpid()}: {' '.join(argv)[:200]}\n"); f.flush()
        print(f"[gpulock] acquired after {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        sys.exit(subprocess.call(argv))


if __name__ == "__main__":
    main()
