#!/usr/bin/env python3
import sys
import os
from datetime import date
from pathlib import Path
import re

def main():
    if len(sys.argv) != 2:
        print("Usage: update_changelog.py <version>")
        sys.exit(1)

    version = sys.argv[1]
    changelog_path = Path("CHANGELOG.md")

    if not changelog_path.exists():
        print("❌ CHANGELOG.md not found")
        sys.exit(1)

    content = changelog_path.read_text(encoding="utf-8").splitlines()

    repo = os.getenv("GITHUB_REPOSITORY", "owner/repo")
    new_entry = f"## [{version}](https://github.com/{repo}/releases/tag/{version}) - {date.today().isoformat()}"

    updated = []
    inserted = False
    marker_moved = False

    for i, line in enumerate(content):
        # Find the "## Next Release" heading
        if not marker_moved and re.match(r"^##\s+Next Release", line):
            # Ensure marker comes right after "Next Release"
            updated.append(line + "\n")
            updated.append("<!-- insertion marker -->")
            updated.append(new_entry)
            marker_moved = True

        # Skip the old insertion marker wherever it was
        elif "<!-- insertion marker -->" in line and marker_moved:
            continue
        
        else:
            updated.append(line)

    if not marker_moved:
        print("❌ Could not find '## Next Release' in CHANGELOG.md")
        sys.exit(1)

    changelog_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    print(f"✅ CHANGELOG.md updated for version {version}")


if __name__ == "__main__":
    main()
