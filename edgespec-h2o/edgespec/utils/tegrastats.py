# -*- coding: utf-8 -*-
"""Tegrastats log parser for Jetson Nano.

Python-3.6 safe.  Parses the output of `tegrastats --interval 500 --logfile`.

Example line:
RAM 2345/3964MB (lfb 123x4MB) SWAP 0/1978MB (cached 0MB) CPU [23%@1479,18%@1479,...] GR3D_FREQ 76% ...
"""

import re
import os


def parse_tegrastats_line(line):
    """Parse one tegrastats output line into a dict.

    Returns
    -------
    dict or None (if line is unparseable)
    """
    info = {}

    # RAM
    m = re.search(r'RAM (\d+)/(\d+)MB', line)
    if m:
        info['ram_used_mb'] = int(m.group(1))
        info['ram_total_mb'] = int(m.group(2))

    # SWAP
    m = re.search(r'SWAP (\d+)/(\d+)MB', line)
    if m:
        info['swap_used_mb'] = int(m.group(1))
        info['swap_total_mb'] = int(m.group(2))

    # CPU
    m = re.search(r'CPU \[([^\]]+)\]', line)
    if m:
        cpus = m.group(1).split(',')
        cpu_pcts = []
        for c in cpus:
            c = c.strip()
            pm = re.match(r'(\d+)%', c)
            if pm:
                cpu_pcts.append(int(pm.group(1)))
        info['cpu_pcts'] = cpu_pcts
        info['cpu_avg_pct'] = sum(cpu_pcts) / len(cpu_pcts) if cpu_pcts else 0

    # GPU freq
    m = re.search(r'GR3D_FREQ (\d+)%', line)
    if m:
        info['gpu_freq_pct'] = int(m.group(1))

    # Temperatures
    for zone in ['CPU', 'GPU', 'PLL', 'AO', 'thermal', 'PMIC', 'iwlwifi']:
        m = re.search(r'%s@(\d+\.?\d*)C' % zone, line)
        if m:
            info['temp_%s_c' % zone.lower()] = float(m.group(1))

    # Power
    m = re.search(r'POM_5V_IN (\d+)/(\d+)', line)
    if m:
        info['power_in_mw'] = int(m.group(1))
        info['power_in_avg_mw'] = int(m.group(2))

    return info if info else None


def parse_tegrastats_log(filepath):
    """Parse a tegrastats log file into a list of dicts.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    list of dict
    """
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parsed = parse_tegrastats_line(line)
                if parsed:
                    entries.append(parsed)
    return entries


def get_peak_ram(entries):
    """Get peak RAM from parsed tegrastats entries.

    Returns
    -------
    float  (MiB)
    """
    if not entries:
        return 0.0
    return max(e.get('ram_used_mb', 0) for e in entries)


def get_max_temp(entries, zone='gpu'):
    """Get max temperature from parsed entries.

    Parameters
    ----------
    entries : list of dict
    zone : str ('gpu', 'cpu', etc.)

    Returns
    -------
    float (Celsius)
    """
    key = 'temp_%s_c' % zone
    temps = [e.get(key, 0) for e in entries]
    return max(temps) if temps else 0.0


def check_swap_activity(entries):
    """Check if any significant swap activity occurred.

    Returns
    -------
    bool  (True = no swap activity = good)
    """
    for e in entries:
        if e.get('swap_used_mb', 0) > 10:  # >10 MB swap is suspicious
            return False
    return True
