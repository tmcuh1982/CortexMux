"""Portable, non-sensitive machine profiling."""

from __future__ import annotations

import os
import platform

from cortexmux.selection.schemas import MachineProfile


def detect_machine_profile() -> MachineProfile:
    """Detect only characteristics that affect local inference suitability."""
    return MachineProfile(
        system=platform.system(),
        release=platform.release(),
        architecture=platform.machine(),
        processor=platform.processor() or None,
        cpu_count=os.cpu_count(),
        total_memory_bytes=_total_memory_bytes(),
    )


def _total_memory_bytes() -> int | None:
    """Read total physical memory using portable POSIX sysconf when available."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    total = page_size * page_count
    return total if total > 0 else None
