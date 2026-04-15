#!/usr/bin/env python3
"""
mempalace migrate — Legacy ChromaDB version migration.

This module previously handled ChromaDB 0.6.x → 1.5.x format migrations.
MemPalace now uses IndentiaGraph as its storage backend; ChromaDB is no
longer a dependency. This file is kept as a stub so that existing CLI
commands fail gracefully instead of with an ImportError.
"""


def migrate(palace_path: str, dry_run: bool = False, confirm: bool = False):
    print(
        "\n  MemPalace now uses IndentiaGraph as its storage backend.\n"
        "  ChromaDB migration is no longer needed.\n"
        "  To rebuild the palace, run: mempalace mine <directory>\n"
    )
    return False


def contains_palace_database(path: str) -> bool:
    """Return True when path looks like a MemPalace palace directory."""
    import os

    return os.path.isdir(path)
