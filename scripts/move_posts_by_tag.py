#!/usr/bin/env python3
"""
Move Hugo posts from content/posts/ into content/{notes,thoughts,projects}/
based on `tag:` front matter.

Usage:
  python3 scripts/move_posts_by_tag.py --dry-run
  python3 scripts/move_posts_by_tag.py --execute

It will skip items where destination already exists and print a report.
"""
import argparse
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / 'content'
POSTS_DIR = CONTENT / 'posts'

ALLOWED = {
    'notes': 'notes',
    'note': 'notes',
    'thoughts': 'thoughts',
    'thought': 'thoughts',
    'projects': 'projects',
    'project': 'projects',
}

FRONT_MATTER_RE = re.compile(r'^---\s*$(.*?)^---\s*$', re.DOTALL | re.MULTILINE)
TAG_RE = re.compile(r'^[ \t]*tag:\s*["\']?(?P<tag>[^"\'\n]+)["\']?', re.MULTILINE)


def read_front_matter(md_path: Path) -> str:
    text = md_path.read_text(encoding='utf-8')
    m = FRONT_MATTER_RE.search(text)
    if not m:
        return None
    fm = m.group(1)
    tm = TAG_RE.search(fm)
    if tm:
        return tm.group('tag').strip()
    return None


def find_post_items():
    if not POSTS_DIR.exists():
        print(f"No {POSTS_DIR} directory found. Nothing to do.")
        return []
    items = []
    for child in POSTS_DIR.iterdir():
        # support both file.md and folder/index.md
        if child.is_dir():
            md = child / 'index.md'
            if md.exists():
                items.append((child, md))
            else:
                # maybe folder has .md inside
                mds = list(child.glob('*.md'))
                if mds:
                    items.append((child, mds[0]))
        elif child.is_file() and child.suffix.lower() == '.md':
            items.append((child, child))
    return items


def move_item(src_path: Path, md_path: Path, dest_root: Path, execute: bool):
    # Determine slug/name
    name = src_path.name
    dest_dir = dest_root / name
    if dest_dir.exists():
        return False, f"destination exists: {dest_dir}"
    if execute:
        # ensure parent exists
        dest_root.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(src_path), str(dest_dir))
        except Exception as e:
            return False, f"move failed: {e}"
    return True, f"moved to {dest_dir}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true', help='Show planned moves')
    p.add_argument('--execute', action='store_true', help='Perform moves')
    args = p.parse_args()

    items = find_post_items()
    if not items:
        return

    planned = []
    skipped = []
    for src, md in items:
        tag = read_front_matter(md)
        if not tag:
            skipped.append((src, 'no-tag'))
            continue
        key = tag.strip().lower()
        target = ALLOWED.get(key)
        if not target:
            skipped.append((src, f'unsupported-tag:{tag}'))
            continue
        dest_root = CONTENT / target
        ok, msg = move_item(src, md, dest_root, execute=args.execute)
        planned.append((src, target, ok, msg))

    # Report
    print('\nPlanned moves:')
    for src, target, ok, msg in planned:
        print(f" - {src} -> content/{target}/  : {'OK' if ok else 'SKIP'} ({msg})")

    if skipped:
        print('\nSkipped items:')
        for src, reason in skipped:
            print(f" - {src}: {reason}")

    if args.dry_run:
        print('\nDry-run complete. No files were moved.')
    elif args.execute:
        print('\nExecution complete.')
    else:
        print('\nRun with --dry-run to preview or --execute to perform the moves.')


if __name__ == '__main__':
    main()
