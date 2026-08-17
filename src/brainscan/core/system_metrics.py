"""Lightweight process and CUDA memory snapshots for local profiling/logging."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import ctypes
from ctypes import wintypes
import os
import sys

import torch


@dataclass(frozen=True)
class MemorySnapshot:
    process_rss_mb: float | None
    process_private_mb: float | None
    system_available_mb: float | None
    system_total_mb: float | None
    system_commit_used_mb: float | None
    system_commit_limit_mb: float | None
    cuda_allocated_mb: float | None
    cuda_reserved_mb: float | None
    cuda_peak_allocated_mb: float | None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def _bytes_to_mb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 * 1024.0)


def _collect_windows_process_memory() -> tuple[float | None, float | None]:
    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    success = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not success:
        return None, None
    return _bytes_to_mb(counters.WorkingSetSize), _bytes_to_mb(counters.PrivateUsage)


def _collect_windows_system_memory() -> tuple[float | None, float | None, float | None, float | None]:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    success = kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    if not success:
        return None, None, None, None
    commit_limit = _bytes_to_mb(status.ullTotalPageFile)
    commit_used = _bytes_to_mb(status.ullTotalPageFile - status.ullAvailPageFile)
    total_phys = _bytes_to_mb(status.ullTotalPhys)
    avail_phys = _bytes_to_mb(status.ullAvailPhys)
    return avail_phys, total_phys, commit_used, commit_limit


def capture_memory_snapshot(device: torch.device | None = None) -> MemorySnapshot:
    process_rss_mb: float | None = None
    process_private_mb: float | None = None
    system_available_mb: float | None = None
    system_total_mb: float | None = None
    system_commit_used_mb: float | None = None
    system_commit_limit_mb: float | None = None

    if sys.platform.startswith("win"):
        process_rss_mb, process_private_mb = _collect_windows_process_memory()
        (
            system_available_mb,
            system_total_mb,
            system_commit_used_mb,
            system_commit_limit_mb,
        ) = _collect_windows_system_memory()

    cuda_allocated_mb: float | None = None
    cuda_reserved_mb: float | None = None
    cuda_peak_allocated_mb: float | None = None
    if (
        device is not None
        and device.type == "cuda"
        and torch.cuda.is_available()
    ):
        cuda_allocated_mb = _bytes_to_mb(torch.cuda.memory_allocated(device))
        cuda_reserved_mb = _bytes_to_mb(torch.cuda.memory_reserved(device))
        cuda_peak_allocated_mb = _bytes_to_mb(torch.cuda.max_memory_allocated(device))

    return MemorySnapshot(
        process_rss_mb=process_rss_mb,
        process_private_mb=process_private_mb,
        system_available_mb=system_available_mb,
        system_total_mb=system_total_mb,
        system_commit_used_mb=system_commit_used_mb,
        system_commit_limit_mb=system_commit_limit_mb,
        cuda_allocated_mb=cuda_allocated_mb,
        cuda_reserved_mb=cuda_reserved_mb,
        cuda_peak_allocated_mb=cuda_peak_allocated_mb,
    )
