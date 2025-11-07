#!/usr/bin/env python3
"""
Add a default `tag: "Notes"` front matter to existing posts in content/posts/**
Only modifies files that have YAML front matter and do not already define a `tag` key.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
POSTS = ROOT / 'content' / 'posts'
DEFAULT_TAG = 'Notes'

fm_pat = re.compile(r'^---\s*$\n', re.MULTILINE)


def process_file(p: Path):
    text = p.read_text(encoding='utf-8')
    # find YAML front matter block
    if not text.startswith('---'):
        print(f"Skipping (no front matter): {p}")
        return
    # locate closing --- after the first
    parts = text.split('---', 2)
    if len(parts) < 3:
        print(f"Skipping (malformed front matter): {p}")
        return
    # parts[1] contains the front matter contents (leading/trailing newlines trimmed by split behavior)
    fm = parts[1]
    body = parts[2]
    # if 'tag:' already present, skip
    if re.search(r'^tag\s*:\s*', fm, re.MULTILINE):
        print(f"Already has tag: {p}")
        return
    # inject tag at end of front matter (before closing)
    new_fm = fm.rstrip() + f"\ntag: \"{DEFAULT_TAG}\"\n"
    new_text = '---' + new_fm + '---' + body
    p.write_text(new_text, encoding='utf-8')
    print(f"Added tag to: {p}")


if __name__ == '__main__':
    if not POSTS.exists():
        print("No posts folder found at content/posts")
        raise SystemExit(1)
    for p in POSTS.rglob('*.md'):
        process_file(p)
    print('Done.')
