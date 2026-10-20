"""Authenticate copied native bodies against the fork's immutable Git object."""
import subprocess
PIN = "1d6e370e7b43a332074809f170532a216c2212fb"

def read_pinned_source(root, path):
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    authority = subprocess.run(["git", "-C", str(root), "show", PIN + ":" + relative],
                               capture_output=True, check=True).stdout
    current = path.read_bytes()
    if current != authority:
        raise ValueError("native source differs from pinned object: " + relative)
    return current.decode()
