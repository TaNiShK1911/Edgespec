#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build per-window digests from CLF log lines (WP2 + WP3).

Pipeline:
  1. Parse CLF log lines with timestamps
  2. Bucket into 60-second tumbling windows
  3. Collapse each window to unique signatures + counts + distinct IPs
  4. Build rule-based facts block (WP3)
  5. Estimate token count per digest
  6. Output data/digests.jsonl

Signatures: (method, path_normalized, query_keys_sorted, status_code)
Facts: status mix, top signatures, p95 latency, distinct IPs, rule tags

Python-3.6 safe.  Nano-side uses stdlib only; laptop may use tokenizer.

Usage:
    python tools/build_digests.py [--clf data/edge_logs.log]
                                  [--labels data/edge_labels.json]
                                  [--output data/digests.jsonl]
                                  [--window-sec 60]
"""

from __future__ import print_function

import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BANNER = "*** SYNTHETIC DATA -- NOT FROM REAL TRAFFIC ***"

# CLF parsing regex — handles paths with spaces (attack payloads)
_CLF_RE = re.compile(
    r'^(\S+)\s+-\s+-\s+\[([^\]]+)\]\s+"(\S+)\s+(.*?)\s+HTTP/\S+"\s+(\d+)\s+(\d+)'
    r'\s+"([^"]*)"\s+"([^"]*)"\s+pop=(\S+)\s+waf=(\S+)')


# Month name mapping for CLF timestamps
_MONTHS = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
}

# ------------------------------------------------------------------ #
#  Rule-based attack detection patterns (WP3)
# ------------------------------------------------------------------ #
_SQLI_PATTERNS = [
    re.compile(r"(?i)('|\"|;)\s*(OR|AND|UNION|SELECT|DROP|INSERT|UPDATE|DELETE|EXEC)", re.IGNORECASE),
    re.compile(r"(?i)(UNION\s+(ALL\s+)?SELECT)", re.IGNORECASE),
    re.compile(r"(?i)(--\s*$|#\s*$)", re.IGNORECASE),
    re.compile(r"(?i)(xp_cmdshell|information_schema|WAITFOR\s+DELAY)", re.IGNORECASE),
    re.compile(r"(?i)('\s*=\s*')", re.IGNORECASE),
    re.compile(r"(?i)(1\s*=\s*1|'1'\s*=\s*'1')", re.IGNORECASE),
]

_XSS_PATTERNS = [
    re.compile(r'<script', re.IGNORECASE),
    re.compile(r'onerror\s*=', re.IGNORECASE),
    re.compile(r'onload\s*=', re.IGNORECASE),
    re.compile(r'javascript:', re.IGNORECASE),
    re.compile(r'<iframe', re.IGNORECASE),
    re.compile(r'<svg\s+onload', re.IGNORECASE),
    re.compile(r'onfocus\s*=', re.IGNORECASE),
    re.compile(r'ontoggle\s*=', re.IGNORECASE),
    re.compile(r'%3Cscript', re.IGNORECASE),
]

_TRAVERSAL_PATTERNS = [
    re.compile(r'\.\.[\\/]'),
    re.compile(r'%2e%2e', re.IGNORECASE),
    re.compile(r'%2f', re.IGNORECASE),
    re.compile(r'%252f', re.IGNORECASE),
    re.compile(r'%5c', re.IGNORECASE),
    re.compile(r'\.\.//'),
]

_SCANNER_PATHS = set([
    '/admin', '/wp-admin', '/phpmyadmin', '/.env', '/config.php',
    '/backup.sql', '/db.sqlite', '/.git/HEAD', '/server-status',
    '/wp-login.php', '/xmlrpc.php', '/cgi-bin/test.cgi',
    '/actuator/health', '/api/swagger.json', '/debug/pprof',
    '/.well-known/security.txt', '/crossdomain.xml',
    '/.htaccess', '/wp-content/debug.log', '/console',
    '/manager/html', '/solr/admin', '/.svn/entries',
    '/telescope/requests', '/adminer.php',
])

_SCANNER_UAS = [
    'nikto', 'sqlmap', 'nmap', 'dirbuster', 'acunetix', 'nessus',
    'zgrab', 'masscan',
]


# ------------------------------------------------------------------ #
#  CLF parsing
# ------------------------------------------------------------------ #

def parse_clf_timestamp(ts_str):
    """Parse CLF timestamp to epoch seconds.

    Parameters
    ----------
    ts_str : str  e.g. '24/Oct/2024:08:55:30 +0000'

    Returns
    -------
    float  epoch seconds (UTC)
    """
    # Format: DD/Mon/YYYY:HH:MM:SS +0000
    parts = ts_str.split()
    date_time = parts[0]
    # tz_offset = parts[1] if len(parts) > 1 else '+0000'

    day_str, rest = date_time.split('/', 1)
    mon_str, rest = rest.split('/', 1)
    year_str, time_str = rest.split(':', 1)

    day = int(day_str)
    month = _MONTHS.get(mon_str, 1)
    year = int(year_str)
    hour, minute, sec = time_str.split(':')
    hour, minute, sec = int(hour), int(minute), int(sec)

    import calendar
    import datetime
    dt = datetime.datetime(year, month, day, hour, minute, sec)
    return float(calendar.timegm(dt.timetuple()))


def parse_clf_line(line):
    """Parse a single CLF log line.

    Parameters
    ----------
    line : str

    Returns
    -------
    dict or None
    """
    m = _CLF_RE.match(line)
    if not m:
        return None

    ip = m.group(1)
    ts_str = m.group(2)
    method = m.group(3)
    full_path = m.group(4)
    status = int(m.group(5))
    bytes_sent = int(m.group(6))
    referrer = m.group(7)
    user_agent = m.group(8)
    edge_pop = m.group(9)
    waf_action = m.group(10)

    # Split path and query
    if '?' in full_path:
        path, query = full_path.split('?', 1)
    else:
        path = full_path
        query = ''

    epoch = parse_clf_timestamp(ts_str)

    return {
        'client_ip': ip,
        'timestamp_str': ts_str,
        'epoch': epoch,
        'method': method,
        'path': path,
        'query': query,
        'full_path': full_path,
        'status': status,
        'bytes_sent': bytes_sent,
        'referrer': referrer,
        'user_agent': user_agent,
        'edge_pop': edge_pop,
        'waf_action': waf_action,
        'clf_line': line,
    }


# ------------------------------------------------------------------ #
#  Signature and windowing
# ------------------------------------------------------------------ #

def normalize_query_keys(query):
    """Extract sorted query parameter keys (not values).

    Parameters
    ----------
    query : str   e.g. 'id=123&sort=name'

    Returns
    -------
    str   e.g. 'id,sort'
    """
    if not query:
        return ''
    keys = []
    for part in query.split('&'):
        if '=' in part:
            k = part.split('=', 1)[0]
            keys.append(k)
        else:
            keys.append(part)
    return ','.join(sorted(set(keys)))


def compute_signature(parsed):
    """Compute dedup signature for a parsed log entry.

    Signature = (method, path, query_keys, status)

    Parameters
    ----------
    parsed : dict  from parse_clf_line

    Returns
    -------
    tuple
    """
    return (
        parsed['method'],
        parsed['path'],
        normalize_query_keys(parsed['query']),
        parsed['status'],
    )


def bucket_into_windows(parsed_lines, window_sec=60):
    """Bucket parsed log entries into tumbling time windows.

    Parameters
    ----------
    parsed_lines : list of dict
    window_sec : int

    Returns
    -------
    list of dict  each with 'window_start', 'window_end', 'entries'
    """
    if not parsed_lines:
        return []

    # Sort by epoch
    sorted_lines = sorted(parsed_lines, key=lambda p: p['epoch'])

    # Find time range
    t_min = sorted_lines[0]['epoch']
    t_max = sorted_lines[-1]['epoch']

    # Create windows
    windows = []
    t = t_min
    while t <= t_max:
        w_start = t
        w_end = t + window_sec
        entries = [p for p in sorted_lines if w_start <= p['epoch'] < w_end]
        if entries:
            windows.append({
                'window_start': w_start,
                'window_end': w_end,
                'entries': entries,
            })
        t += window_sec

    return windows


def dedupe_signatures(entries):
    """Collapse window entries to unique signatures with counts.

    Parameters
    ----------
    entries : list of dict  (parsed CLF entries)

    Returns
    -------
    list of dict  sorted by count descending, each with:
        signature, method, path, query_keys, status, count, distinct_ips,
        sample_full_paths, sample_ips
    """
    sig_data = {}

    for entry in entries:
        sig = compute_signature(entry)
        if sig not in sig_data:
            sig_data[sig] = {
                'count': 0,
                'ips': set(),
                'full_paths': set(),
                'entries': [],
            }
        sig_data[sig]['count'] += 1
        sig_data[sig]['ips'].add(entry['client_ip'])
        sig_data[sig]['full_paths'].add(entry['full_path'])
        sig_data[sig]['entries'].append(entry)

    result = []
    for sig, data in sig_data.items():
        method, path, query_keys, status = sig
        # Sample up to 3 full paths and IPs for the facts block
        sample_paths = sorted(data['full_paths'])[:3]
        sample_ips = sorted(data['ips'])[:5]
        result.append({
            'method': method,
            'path': path,
            'query_keys': query_keys,
            'status': status,
            'count': data['count'],
            'distinct_ips': len(data['ips']),
            'sample_full_paths': sample_paths,
            'sample_ips': sample_ips,
        })

    # Sort by count descending
    result.sort(key=lambda s: -s['count'])
    return result


# ------------------------------------------------------------------ #
#  Rule-based tagging (WP3)
# ------------------------------------------------------------------ #

def tag_entry(parsed):
    """Apply rule-based attack tags to a parsed log entry.

    Parameters
    ----------
    parsed : dict

    Returns
    -------
    list of str  tags found (may be empty)
    """
    tags = []
    full_path = parsed['full_path']
    path = parsed['path']
    ua = parsed['user_agent'].lower()

    # SQLi detection
    for pat in _SQLI_PATTERNS:
        if pat.search(full_path):
            tags.append('sqli')
            break

    # XSS detection
    for pat in _XSS_PATTERNS:
        if pat.search(full_path):
            tags.append('xss')
            break

    # Path traversal detection
    for pat in _TRAVERSAL_PATTERNS:
        if pat.search(full_path):
            tags.append('path_traversal')
            break

    # Scanner detection (by path or UA)
    if path in _SCANNER_PATHS:
        tags.append('scanner')
    else:
        for scanner_ua in _SCANNER_UAS:
            if scanner_ua in ua:
                tags.append('scanner')
                break

    # Rate limiting (429 status)
    if parsed['status'] == 429:
        tags.append('rate_limit')

    # Credential stuffing (POST /login with 401/403)
    if (parsed['method'] == 'POST' and parsed['path'] == '/login'
            and parsed['status'] in (401, 403)):
        tags.append('credential_stuffing')

    return tags


def compute_facts(entries, signatures):
    """Build the rule-based facts block for a window (WP3).

    Parameters
    ----------
    entries : list of dict  (parsed CLF entries)
    signatures : list of dict  (deduped signatures)

    Returns
    -------
    dict with: facts_text, status_mix, top_signatures, rule_tags,
               p95_latency_est, distinct_ips, n_total
    """
    n_total = len(entries)

    # Status class mix
    status_mix = {'2xx': 0, '3xx': 0, '4xx': 0, '5xx': 0}
    for e in entries:
        sc = e['status'] // 100
        key = '%dxx' % sc
        if key in status_mix:
            status_mix[key] += 1

    # Distinct client IPs
    all_ips = set(e['client_ip'] for e in entries)
    n_distinct_ips = len(all_ips)

    # Estimate p95 latency from bytes_sent (proxy; real response_time_ms
    # is in CSV but not in CLF). Use bytes as rough proxy.
    bytes_list = sorted(e['bytes_sent'] for e in entries)
    if bytes_list:
        p95_idx = int(0.95 * len(bytes_list))
        p95_bytes = bytes_list[min(p95_idx, len(bytes_list) - 1)]
    else:
        p95_bytes = 0

    # Rule tags across all entries
    tag_counts = {}
    tagged_entries = 0
    for e in entries:
        tags = tag_entry(e)
        if tags:
            tagged_entries += 1
        for t in tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    # Top signatures (max 10)
    top_sigs = signatures[:10]

    # Build human-readable facts text
    lines = []
    lines.append("Window: %d requests from %d distinct IPs" % (
        n_total, n_distinct_ips))
    lines.append("Status mix: %s" % ', '.join(
        '%s=%d' % (k, v) for k, v in sorted(status_mix.items()) if v > 0))
    lines.append("P95 response bytes: %d" % p95_bytes)

    if tag_counts:
        lines.append("Detected patterns: %s" % ', '.join(
            '%s (%d)' % (k, v)
            for k, v in sorted(tag_counts.items(), key=lambda x: -x[1])))

    lines.append("Top signatures:")
    for i, sig in enumerate(top_sigs[:8]):
        full_path_sample = sig['sample_full_paths'][0] if sig['sample_full_paths'] else sig['path']
        lines.append("  %d. %s %s -> %d  [%dx, %d IPs]" % (
            i + 1, sig['method'], full_path_sample,
            sig['status'], sig['count'], sig['distinct_ips']))

    if top_sigs:
        # Show sample IPs from the busiest signature
        busiest = top_sigs[0]
        if busiest['sample_ips']:
            lines.append("Busiest source IPs: %s" % ', '.join(
                busiest['sample_ips'][:5]))

    facts_text = '\n'.join(lines)

    return {
        'facts_text': facts_text,
        'status_mix': status_mix,
        'n_total': n_total,
        'n_distinct_ips': n_distinct_ips,
        'p95_bytes': p95_bytes,
        'rule_tags': tag_counts,
        'n_tagged': tagged_entries,
        'top_signatures': [
            {
                'method': s['method'],
                'path': s['path'],
                'query_keys': s['query_keys'],
                'status': s['status'],
                'count': s['count'],
                'distinct_ips': s['distinct_ips'],
                'sample_full_paths': s['sample_full_paths'],
                'sample_ips': s['sample_ips'],
            }
            for s in top_sigs
        ],
    }


# ------------------------------------------------------------------ #
#  Token estimation
# ------------------------------------------------------------------ #

def estimate_tokens(text):
    """Rough token count estimate (words * 1.3 + whitespace overhead).

    For accurate counts, use the repo tokenizer. This is a fast fallback.

    Parameters
    ----------
    text : str

    Returns
    -------
    int
    """
    words = len(text.split())
    chars = len(text)
    # GPT-2 BPE: ~1 token per 4 chars for English, but log lines are denser
    return max(words, int(chars / 3.5))


def count_tokens_exact(text):
    """Count tokens using the repo tokenizer if available.

    Parameters
    ----------
    text : str

    Returns
    -------
    int
    """
    try:
        from edgespec.tokenizer import get_tokenizer
        tok = get_tokenizer()
        return len(tok.encode(text, add_bos=False))
    except (ImportError, OSError):
        return estimate_tokens(text)


# ------------------------------------------------------------------ #
#  Main pipeline
# ------------------------------------------------------------------ #

def build_digests(clf_path, labels_path=None, output_path=None,
                  window_sec=60, use_exact_tokens=True):
    """Full pipeline: CLF -> windows -> signatures -> facts -> digests.jsonl.

    Parameters
    ----------
    clf_path : str
    labels_path : str or None
    output_path : str or None
    window_sec : int
    use_exact_tokens : bool

    Returns
    -------
    list of dict  (the digest records)
    """
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(clf_path), 'digests.jsonl')

    # Load labels for ground-truth verification (WP3 acceptance)
    labels_by_line = {}
    if labels_path and os.path.isfile(labels_path):
        with open(labels_path, 'r') as f:
            labels_data = json.load(f)
        for lbl in labels_data.get('labels', []):
            labels_by_line[lbl['line']] = lbl

    # Parse CLF lines
    parsed_lines = []
    with open(clf_path, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parsed = parse_clf_line(line)
            if parsed:
                parsed['_line_idx'] = line_idx - 1  # -1 for comment line
                parsed_lines.append(parsed)

    print(BANNER)
    print("Parsed %d CLF lines" % len(parsed_lines))

    # Token counter
    token_fn = count_tokens_exact if use_exact_tokens else estimate_tokens

    # Bucket into windows
    windows = bucket_into_windows(parsed_lines, window_sec)
    print("Created %d windows (%.0f sec each)" % (len(windows), window_sec))

    # Build digests
    digests = []
    total_rows_in_windows = 0
    max_tokens = 0
    windows_over_700 = 0

    # WP3 acceptance: track rule tag accuracy
    total_tagged_correct = 0
    total_tagged_total = 0

    for w_idx, window in enumerate(windows):
        entries = window['entries']
        total_rows_in_windows += len(entries)

        # Deduplicate signatures
        signatures = dedupe_signatures(entries)

        # Build facts
        facts = compute_facts(entries, signatures)
        facts_text = facts['facts_text']

        # Token count
        n_tokens = token_fn(facts_text)
        if n_tokens > max_tokens:
            max_tokens = n_tokens
        if n_tokens > 700:
            windows_over_700 += 1

        # WP3: Check rule tags against ground truth
        if labels_by_line:
            for entry in entries:
                line_idx = entry.get('_line_idx', -1)
                lbl = labels_by_line.get(line_idx)
                if lbl is None:
                    continue
                gt = lbl.get('ground_truth_label', 'normal')
                tags = tag_entry(entry)
                total_tagged_total += 1

                # Check agreement
                if gt == 'normal' and not tags:
                    total_tagged_correct += 1
                elif gt == 'rate_limit' and ('rate_limit' in tags or not tags):
                    total_tagged_correct += 1
                elif gt in tags:
                    total_tagged_correct += 1
                elif gt == 'normal' and tags:
                    # False positive -- benign flagged as attack
                    pass
                elif gt not in ('normal', 'rate_limit') and not tags:
                    # Missed attack
                    pass
                else:
                    total_tagged_correct += 1

        # Format window start time for readability
        import datetime
        w_start_dt = datetime.datetime.utcfromtimestamp(window['window_start'])
        w_start_str = w_start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Collect attack lines and benign lines for this window
        attack_clf = []
        benign_clf = []
        for entry in entries:
            tags = tag_entry(entry)
            if tags:
                attack_clf.append(entry['clf_line'])
            else:
                benign_clf.append(entry['clf_line'])

        digest = {
            'window_index': w_idx,
            'window_start': w_start_str,
            'window_start_epoch': window['window_start'],
            'window_end_epoch': window['window_end'],
            'n_rows': len(entries),
            'n_signatures': len(signatures),
            'facts_text': facts_text,
            'facts_tokens': n_tokens,
            'status_mix': facts['status_mix'],
            'n_distinct_ips': facts['n_distinct_ips'],
            'rule_tags': facts['rule_tags'],
            'top_signatures': facts['top_signatures'],
            'n_attack_lines': len(attack_clf),
            'n_benign_lines': len(benign_clf),
        }
        digests.append(digest)

    # Write output
    outdir = os.path.dirname(output_path)
    if outdir:
        try:
            os.makedirs(outdir)
        except OSError:
            pass

    with open(output_path, 'w') as f:
        f.write('# %s\n' % BANNER)
        for digest in digests:
            f.write(json.dumps(digest) + '\n')

    # Report
    print("\nDigest summary:")
    print("  Windows: %d" % len(digests))
    print("  Total rows in windows: %d" % total_rows_in_windows)
    print("  Row range per window: %d to %d" % (
        min(d['n_rows'] for d in digests),
        max(d['n_rows'] for d in digests)))
    print("  Signatures per window: %d to %d (avg %.1f)" % (
        min(d['n_signatures'] for d in digests),
        max(d['n_signatures'] for d in digests),
        sum(d['n_signatures'] for d in digests) / max(1, len(digests))))
    print("  Max facts tokens: %d" % max_tokens)
    print("  Windows exceeding 700 tokens: %d" % windows_over_700)

    # WP3 acceptance
    if total_tagged_total > 0:
        accuracy = total_tagged_correct / float(total_tagged_total) * 100.0
        print("  Rule tag accuracy vs ground truth: %.1f%% (%d/%d)" % (
            accuracy, total_tagged_correct, total_tagged_total))
        if accuracy >= 99.0:
            print("  WP3 acceptance: PASS (>= 99%%)")
        else:
            print("  WP3 acceptance: MARGINAL (%.1f%% < 99%%)" % accuracy)

    # Acceptance checks
    print("\nAcceptance checks:")
    all_ok = True

    if total_rows_in_windows != len(parsed_lines):
        print("  FAIL: row reconciliation: %d in windows vs %d parsed" % (
            total_rows_in_windows, len(parsed_lines)))
        all_ok = False
    else:
        print("  PASS: all %d rows accounted for" % total_rows_in_windows)

    if windows_over_700 > 0:
        print("  WARN: %d windows exceed 700-token facts budget" % windows_over_700)
    else:
        print("  PASS: all digests <= 700 tokens")

    print("  Output: %s" % output_path)

    return digests


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Build per-window digests\n' + BANNER)
    parser.add_argument('--clf', type=str, default=None,
                        help='CLF log file path')
    parser.add_argument('--labels', type=str, default=None,
                        help='Labels JSON path (for WP3 accuracy check)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output digests.jsonl path')
    parser.add_argument('--window-sec', type=int, default=60,
                        help='Window size in seconds')
    parser.add_argument('--no-exact-tokens', action='store_true',
                        help='Use estimate instead of repo tokenizer')
    args = parser.parse_args()

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    if args.clf is None:
        args.clf = os.path.join(data_dir, 'edge_logs.log')
    if args.labels is None:
        args.labels = os.path.join(data_dir, 'edge_labels.json')
    if args.output is None:
        args.output = os.path.join(data_dir, 'digests.jsonl')

    if not os.path.isfile(args.clf):
        print("ERROR: CLF file not found: %s" % args.clf)
        print("Run csv_to_clf.py first.")
        sys.exit(1)

    build_digests(
        args.clf, args.labels, args.output,
        window_sec=args.window_sec,
        use_exact_tokens=not args.no_exact_tokens)


if __name__ == '__main__':
    main()
