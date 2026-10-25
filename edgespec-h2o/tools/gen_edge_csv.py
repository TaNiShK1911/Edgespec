#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate synthetic_edge_logs.csv with 10,000 rows for EdgeSpec-H2O demo.

Distribution:
  6,500 normal traffic
  2,000 high-frequency rate-limit burst (concentrated in one 60s window)
  1,500 attack simulations (SQLi, XSS, path traversal, credential stuffing, scanner)

Timestamps are ISO 8601 with mixed microsecond precision.
Each row gets a unique request_id for label alignment.

All data is SYNTHETIC -- labelled as such.

Python-3.6 safe.

Usage:
    python tools/gen_edge_csv.py [--output data/synthetic_edge_logs.csv] [--seed 42]
"""

from __future__ import print_function

import csv
import datetime
import os
import random
import sys

BANNER = "*** SYNTHETIC DATA -- NOT FROM REAL TRAFFIC ***"

# Base timestamp: 2024-10-24T08:00:00 UTC
BASE_EPOCH = 1729756800
TOTAL_SPAN = 2520  # 42 minutes

# Burst window for high-frequency traffic: minute 20-21
BURST_OFFSET_START = 1200
BURST_OFFSET_END = 1260

# ------------------------------------------------------------------ #
#  Benign resources
# ------------------------------------------------------------------ #
BENIGN_PATHS = [
    '/api/v1/users', '/api/v1/products', '/api/v1/orders',
    '/api/v1/health', '/api/v1/status', '/api/v1/metrics',
    '/api/v2/users', '/api/v2/products', '/api/v2/orders',
    '/index.html', '/about', '/contact', '/privacy', '/terms',
    '/static/css/main.css', '/static/js/app.js', '/static/img/logo.png',
    '/favicon.ico', '/robots.txt', '/sitemap.xml',
    '/login', '/logout', '/register', '/forgot-password',
    '/dashboard', '/settings', '/profile', '/notifications',
    '/cdn/assets/bundle.min.js', '/cdn/assets/style.min.css',
    '/api/v1/cart', '/api/v1/checkout', '/api/v1/reviews',
    '/blog', '/blog/latest', '/docs', '/docs/api-reference',
]

BENIGN_QUERIES = [
    '', '', '', '', '',  # most requests have no query
    'page=1', 'page=2', 'page=3',
    'limit=10', 'limit=25', 'limit=50',
    'sort=name', 'sort=date', 'sort=price',
    'q=laptop', 'q=phone', 'q=tablet', 'q=headphones',
    'category=electronics', 'category=books', 'category=clothing',
    'id=123', 'id=456', 'id=789',
    'format=json', 'format=xml',
]

# High-frequency burst paths (rate-limit targets)
BURST_PATHS = [
    '/api/v1/products', '/api/v1/users', '/api/v1/orders',
    '/api/v1/search', '/api/v1/checkout',
]
BURST_QUERIES = [
    'page=1', 'page=2', 'limit=100', 'q=deal', 'q=sale',
    'sort=price&order=asc', 'category=all',
]

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15',
    'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 Chrome/118.0.0.0 Mobile Safari/537.36',
    'curl/8.4.0',
    'python-requests/2.31.0',
    'PostmanRuntime/7.35.0',
]

SCANNER_UAS = [
    'Nikto/2.1.6', 'sqlmap/1.7', 'Nmap Scripting Engine',
    'DirBuster-1.0-RC1', 'Acunetix', 'Nessus/10.6',
    'Mozilla/5.0 zgrab/0.x', 'masscan/1.3',
]

EDGE_POPS = ['IAD', 'SFO', 'LHR', 'FRA', 'NRT', 'SIN', 'SYD', 'GRU', 'JNB', 'BOM']

REFERRERS = ['-', 'https://example.com', 'https://google.com',
             'https://shop.example.com', '-', '-', '-']

# ------------------------------------------------------------------ #
#  Attack payloads
# ------------------------------------------------------------------ #
SQLI_PAYLOADS = [
    "' OR '1'='1", "' OR '1'='1'--", "1; DROP TABLE users;--",
    "' UNION SELECT username, password FROM users--",
    "admin'--", "1' AND 1=1--", "' OR 1=1#",
    "1; EXEC xp_cmdshell('whoami');--",
    "' UNION ALL SELECT NULL,NULL,table_name FROM information_schema.tables--",
    "1' AND (SELECT COUNT(*) FROM users)>0--",
    "' OR 'x'='x", "1' ORDER BY 1--", "' AND 1=CONVERT(int,(SELECT TOP 1 name FROM sysobjects))--",
    "1; WAITFOR DELAY '0:0:5'--",
]
SQLI_PATHS = ['/login', '/api/v1/users', '/search', '/api/v2/products',
              '/api/v1/orders', '/api/v1/reviews']
SQLI_PARAMS = ['username', 'id', 'q', 'filter', 'order_id', 'user_id']

XSS_PAYLOADS = [
    "<script>alert('XSS')</script>", "<img src=x onerror=alert(1)>",
    "<svg onload=alert(document.cookie)>",
    "javascript:alert('XSS')", "'\"><script>alert(1)</script>",
    "<iframe src='javascript:alert(1)'>",
    "<body onload=alert('XSS')>",
    "<%2Fscript%3E%3Cscript%3Ealert(1)%3C%2Fscript%3E",
    "<input onfocus=alert(1) autofocus>",
    "<details open ontoggle=alert(1)>",
]

PATH_TRAVERSAL_PATHS = [
    '/../../../etc/passwd', '/..%2F..%2F..%2Fetc/passwd',
    '/....//....//....//etc/shadow', '/%2e%2e/%2e%2e/etc/passwd',
    '/static/..%252f..%252f..%252fetc/passwd',
    '/api/../admin/config', '/uploads/../../private/keys',
    '/../../../etc/hosts', '/..%5C..%5C..%5Cwindows/system32/config/sam',
    '/data/../../secrets.json',
]

SCANNER_PATHS = [
    '/admin', '/wp-admin', '/phpmyadmin', '/.env', '/config.php',
    '/backup.sql', '/db.sqlite', '/.git/HEAD', '/server-status',
    '/wp-login.php', '/xmlrpc.php', '/cgi-bin/test.cgi',
    '/actuator/health', '/api/swagger.json', '/debug/pprof',
    '/.well-known/security.txt', '/crossdomain.xml',
    '/.htaccess', '/wp-content/debug.log', '/console',
    '/manager/html', '/solr/admin', '/.svn/entries',
    '/telescope/requests', '/adminer.php',
]

CSV_COLUMNS = [
    'request_id', 'timestamp', 'client_ip', 'method', 'path',
    'query_string', 'http_version', 'status', 'bytes_sent',
    'response_time_ms', 'user_agent', 'referrer', 'edge_pop',
    'waf_action', 'ground_truth_label', 'scenario', 'label_reason',
]


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _random_ip(rng):
    return '%d.%d.%d.%d' % (
        rng.randint(1, 223), rng.randint(0, 255),
        rng.randint(0, 255), rng.randint(1, 254))


def _ts_to_iso(epoch_float, rng):
    """Convert epoch float to ISO 8601 string, randomly with/without microseconds."""
    dt = datetime.datetime.utcfromtimestamp(epoch_float)
    if rng.random() < 0.4:
        # with microseconds
        return dt.strftime('%Y-%m-%dT%H:%M:%S.') + '%06d' % dt.microsecond + 'Z'
    else:
        return dt.strftime('%Y-%m-%dT%H:%M:%SZ')


def _make_ip_pool(rng, size):
    """Create a pool of IPs for reuse."""
    return [_random_ip(rng) for _ in range(size)]


# ------------------------------------------------------------------ #
#  Row generators
# ------------------------------------------------------------------ #

def _gen_normal_rows(rng, count, req_counter):
    """Generate normal traffic rows spread across the full time span."""
    rows = []
    for i in range(count):
        offset = rng.uniform(0, TOTAL_SPAN)
        ts = BASE_EPOCH + offset + rng.random()
        path = rng.choice(BENIGN_PATHS)
        query = rng.choice(BENIGN_QUERIES)
        method = rng.choice(['GET', 'GET', 'GET', 'GET', 'POST', 'HEAD', 'PUT'])
        status = rng.choice([200, 200, 200, 200, 200, 201, 204, 301, 302, 304])
        rows.append({
            'request_id': 'REQ-%05d' % req_counter[0],
            'timestamp': _ts_to_iso(ts, rng),
            'client_ip': _random_ip(rng),
            'method': method,
            'path': path,
            'query_string': query,
            'http_version': 'HTTP/1.1',
            'status': status,
            'bytes_sent': rng.randint(200, 50000),
            'response_time_ms': rng.randint(5, 250),
            'user_agent': rng.choice(USER_AGENTS),
            'referrer': rng.choice(REFERRERS),
            'edge_pop': rng.choice(EDGE_POPS),
            'waf_action': 'pass',
            'ground_truth_label': 'normal',
            'scenario': 'normal',
            'label_reason': 'Standard %s request to %s' % (method, path),
            '_sort_ts': ts,
        })
        req_counter[0] += 1
    return rows


def _gen_burst_rows(rng, count, req_counter):
    """Generate high-frequency rate-limit burst in one 60s window."""
    rows = []
    burst_ips = _make_ip_pool(rng, 30)
    for i in range(count):
        offset = rng.uniform(BURST_OFFSET_START, BURST_OFFSET_END)
        ts = BASE_EPOCH + offset + rng.random()
        path = rng.choice(BURST_PATHS)
        query = rng.choice(BURST_QUERIES)
        ip = rng.choice(burst_ips)
        # Most get 429, some get 200/503
        status = rng.choice([429, 429, 429, 429, 429, 429, 200, 503])
        waf = 'pass' if status == 200 else rng.choice(['block', 'log', 'challenge'])
        rows.append({
            'request_id': 'REQ-%05d' % req_counter[0],
            'timestamp': _ts_to_iso(ts, rng),
            'client_ip': ip,
            'method': rng.choice(['GET', 'GET', 'POST']),
            'path': path,
            'query_string': query,
            'http_version': 'HTTP/1.1',
            'status': status,
            'bytes_sent': rng.randint(0, 500),
            'response_time_ms': rng.randint(1, 20),
            'user_agent': rng.choice(USER_AGENTS[:5]),
            'referrer': '-',
            'edge_pop': rng.choice(EDGE_POPS[:3]),
            'waf_action': waf,
            'ground_truth_label': 'rate_limit',
            'scenario': 'high_frequency',
            'label_reason': 'High-frequency burst to %s (status %d)' % (path, status),
            '_sort_ts': ts,
        })
        req_counter[0] += 1
    return rows


def _gen_sqli_rows(rng, count, req_counter):
    """Generate SQL injection attack rows."""
    rows = []
    ips = _make_ip_pool(rng, 40)
    for i in range(count):
        offset = rng.uniform(60, TOTAL_SPAN - 60)
        ts = BASE_EPOCH + offset + rng.random()
        payload = rng.choice(SQLI_PAYLOADS)
        path = rng.choice(SQLI_PATHS)
        param = rng.choice(SQLI_PARAMS)
        query = '%s=%s' % (param, payload)
        status = rng.choice([200, 400, 403, 500])
        rows.append({
            'request_id': 'REQ-%05d' % req_counter[0],
            'timestamp': _ts_to_iso(ts, rng),
            'client_ip': rng.choice(ips),
            'method': rng.choice(['GET', 'POST']),
            'path': path,
            'query_string': query,
            'http_version': 'HTTP/1.1',
            'status': status,
            'bytes_sent': rng.randint(100, 5000),
            'response_time_ms': rng.randint(50, 2000),
            'user_agent': rng.choice(USER_AGENTS),
            'referrer': '-',
            'edge_pop': rng.choice(EDGE_POPS),
            'waf_action': rng.choice(['block', 'log']),
            'ground_truth_label': 'sqli',
            'scenario': 'attack',
            'label_reason': 'SQL injection: %s in %s' % (payload[:30], param),
            '_sort_ts': ts,
        })
        req_counter[0] += 1
    return rows


def _gen_xss_rows(rng, count, req_counter):
    """Generate XSS attack rows."""
    rows = []
    ips = _make_ip_pool(rng, 30)
    for i in range(count):
        offset = rng.uniform(60, TOTAL_SPAN - 60)
        ts = BASE_EPOCH + offset + rng.random()
        payload = rng.choice(XSS_PAYLOADS)
        path = rng.choice(['/search', '/api/v1/reviews', '/blog', '/contact'])
        query = 'q=%s' % payload
        status = rng.choice([200, 400, 403])
        rows.append({
            'request_id': 'REQ-%05d' % req_counter[0],
            'timestamp': _ts_to_iso(ts, rng),
            'client_ip': rng.choice(ips),
            'method': 'GET',
            'path': path,
            'query_string': query,
            'http_version': 'HTTP/1.1',
            'status': status,
            'bytes_sent': rng.randint(100, 3000),
            'response_time_ms': rng.randint(20, 500),
            'user_agent': rng.choice(USER_AGENTS),
            'referrer': '-',
            'edge_pop': rng.choice(EDGE_POPS),
            'waf_action': rng.choice(['block', 'log']),
            'ground_truth_label': 'xss',
            'scenario': 'attack',
            'label_reason': 'XSS payload: %s' % payload[:30],
            '_sort_ts': ts,
        })
        req_counter[0] += 1
    return rows


def _gen_traversal_rows(rng, count, req_counter):
    """Generate path traversal attack rows."""
    rows = []
    ips = _make_ip_pool(rng, 30)
    for i in range(count):
        offset = rng.uniform(60, TOTAL_SPAN - 60)
        ts = BASE_EPOCH + offset + rng.random()
        path = rng.choice(PATH_TRAVERSAL_PATHS)
        status = rng.choice([200, 400, 403, 404])
        rows.append({
            'request_id': 'REQ-%05d' % req_counter[0],
            'timestamp': _ts_to_iso(ts, rng),
            'client_ip': rng.choice(ips),
            'method': 'GET',
            'path': path,
            'query_string': '',
            'http_version': 'HTTP/1.1',
            'status': status,
            'bytes_sent': rng.randint(0, 1000),
            'response_time_ms': rng.randint(10, 300),
            'user_agent': rng.choice(USER_AGENTS + SCANNER_UAS),
            'referrer': '-',
            'edge_pop': rng.choice(EDGE_POPS),
            'waf_action': rng.choice(['block', 'log']),
            'ground_truth_label': 'path_traversal',
            'scenario': 'attack',
            'label_reason': 'Path traversal: %s' % path[:40],
            '_sort_ts': ts,
        })
        req_counter[0] += 1
    return rows


def _gen_credential_rows(rng, count, req_counter):
    """Generate credential stuffing bursts."""
    rows = []
    # Several burst IPs, each with many attempts
    n_ips = max(5, count // 25)
    ips = _make_ip_pool(rng, n_ips)
    per_ip = count // n_ips
    remainder = count - per_ip * n_ips

    for ip_idx, ip in enumerate(ips):
        n = per_ip + (1 if ip_idx < remainder else 0)
        # Each IP's burst is concentrated in a short time window
        burst_center = rng.uniform(120, TOTAL_SPAN - 120)
        for j in range(n):
            offset = burst_center + rng.uniform(-30, 30)
            ts = BASE_EPOCH + offset + rng.random()
            status = rng.choice([401, 401, 401, 403, 200])
            rows.append({
                'request_id': 'REQ-%05d' % req_counter[0],
                'timestamp': _ts_to_iso(ts, rng),
                'client_ip': ip,
                'method': 'POST',
                'path': '/login',
                'query_string': '',
                'http_version': 'HTTP/1.1',
                'status': status,
                'bytes_sent': rng.randint(100, 500),
                'response_time_ms': rng.randint(50, 500),
                'user_agent': rng.choice(USER_AGENTS[:3]),
                'referrer': '-',
                'edge_pop': rng.choice(EDGE_POPS[:3]),
                'waf_action': rng.choice(['log', 'challenge', 'block']),
                'ground_truth_label': 'credential_stuffing',
                'scenario': 'attack',
                'label_reason': 'Credential stuffing from %s' % ip,
                '_sort_ts': ts,
            })
            req_counter[0] += 1
    return rows


def _gen_scanner_rows(rng, count, req_counter):
    """Generate scanner sweep rows."""
    rows = []
    n_scanners = max(3, count // 25)
    ips = _make_ip_pool(rng, n_scanners)
    per_scanner = count // n_scanners
    remainder = count - per_scanner * n_scanners

    for sc_idx, ip in enumerate(ips):
        n = per_scanner + (1 if sc_idx < remainder else 0)
        ua = rng.choice(SCANNER_UAS)
        sweep_start = rng.uniform(60, TOTAL_SPAN - 180)
        for j in range(n):
            offset = sweep_start + j * rng.uniform(2, 8)
            ts = BASE_EPOCH + offset + rng.random()
            path = SCANNER_PATHS[j % len(SCANNER_PATHS)]
            status = rng.choice([404, 404, 404, 403, 200])
            rows.append({
                'request_id': 'REQ-%05d' % req_counter[0],
                'timestamp': _ts_to_iso(ts, rng),
                'client_ip': ip,
                'method': 'GET',
                'path': path,
                'query_string': '',
                'http_version': 'HTTP/1.1',
                'status': status,
                'bytes_sent': rng.randint(0, 1000),
                'response_time_ms': rng.randint(5, 100),
                'user_agent': ua,
                'referrer': '-',
                'edge_pop': rng.choice(EDGE_POPS),
                'waf_action': rng.choice(['log', 'block']),
                'ground_truth_label': 'scanner',
                'scenario': 'attack',
                'label_reason': 'Scanner sweep (%s)' % ua.split('/')[0],
                '_sort_ts': ts,
            })
            req_counter[0] += 1
    return rows


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def generate_edge_csv(output_path, seed=42):
    """Generate the full 10,000-row CSV."""
    rng = random.Random(seed)
    req_counter = [1]  # mutable counter

    print(BANNER)
    print("Generating synthetic_edge_logs.csv ...")

    # Generate all rows
    rows = []
    rows.extend(_gen_normal_rows(rng, 6500, req_counter))
    rows.extend(_gen_burst_rows(rng, 2000, req_counter))
    rows.extend(_gen_sqli_rows(rng, 400, req_counter))
    rows.extend(_gen_xss_rows(rng, 300, req_counter))
    rows.extend(_gen_traversal_rows(rng, 300, req_counter))
    rows.extend(_gen_credential_rows(rng, 300, req_counter))
    rows.extend(_gen_scanner_rows(rng, 200, req_counter))

    # Sort by timestamp
    rows.sort(key=lambda r: r['_sort_ts'])

    # Re-assign request IDs in sorted order
    for i, row in enumerate(rows):
        row['request_id'] = 'REQ-%05d' % (i + 1)
        del row['_sort_ts']

    # Verify distribution
    counts = {}
    for row in rows:
        s = row['scenario']
        counts[s] = counts.get(s, 0) + 1

    attack_types = {}
    for row in rows:
        gt = row['ground_truth_label']
        attack_types[gt] = attack_types.get(gt, 0) + 1

    print("Total rows: %d" % len(rows))
    print("Scenario distribution: %s" % counts)
    print("Label distribution: %s" % attack_types)

    # Write CSV
    outdir = os.path.dirname(output_path)
    if outdir:
        try:
            os.makedirs(outdir)
        except OSError:
            pass

    with open(output_path, 'w', newline='') as f:
        f.write('# %s\n' % BANNER)
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("Output: %s (%d rows)" % (output_path, len(rows)))
    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate synthetic_edge_logs.csv\n' + BANNER)
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'synthetic_edge_logs.csv')

    generate_edge_csv(args.output, seed=args.seed)


if __name__ == '__main__':
    main()
