#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Synthetic edge-server log generator (P1-12).

Generates W3C/CLF-style access logs with injected, ground-truth-labelled attacks:
- SQLi (SQL injection)
- XSS (cross-site scripting)
- Path traversal
- Credential stuffing bursts
- Scanner sweeps

All data is SYNTHETIC — banner printed on every output.

Usage:
    python tools/gen_logs.py --n 1000 --attack-ratio 0.15 --output data/synthetic_logs.jsonl
"""

from __future__ import print_function

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #

BANNER = "*** SYNTHETIC DATA — NOT FROM REAL TRAFFIC ***"

METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'PATCH']
STATUS_CODES = [200, 200, 200, 200, 200, 201, 204, 301, 302, 304, 400, 401, 403, 404, 500, 502, 503]

BENIGN_PATHS = [
    '/api/v1/users', '/api/v1/products', '/api/v1/orders',
    '/api/v1/health', '/api/v1/status', '/api/v1/metrics',
    '/api/v2/users/{id}', '/api/v2/products/{id}',
    '/index.html', '/about', '/contact', '/privacy',
    '/static/css/main.css', '/static/js/app.js', '/static/img/logo.png',
    '/favicon.ico', '/robots.txt', '/sitemap.xml',
    '/login', '/logout', '/register', '/forgot-password',
    '/dashboard', '/settings', '/profile',
    '/search?q=laptop', '/search?q=phone', '/search?q=tablet',
    '/cdn/assets/bundle.min.js', '/cdn/assets/style.min.css',
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
WAF_ACTIONS = ['pass', 'block', 'challenge', 'log']

# Attack patterns
SQLI_PAYLOADS = [
    "' OR '1'='1", "' OR '1'='1'--", "1; DROP TABLE users;--",
    "' UNION SELECT username, password FROM users--",
    "admin'--", "1' AND 1=1--", "' OR 1=1#",
    "1; EXEC xp_cmdshell('whoami');--",
    "' UNION ALL SELECT NULL,NULL,table_name FROM information_schema.tables--",
    "1' AND (SELECT COUNT(*) FROM users)>0--",
]

XSS_PAYLOADS = [
    "<script>alert('XSS')</script>", "<img src=x onerror=alert(1)>",
    "<svg onload=alert(document.cookie)>",
    "javascript:alert('XSS')", "'\"><script>alert(1)</script>",
    "<iframe src='javascript:alert(1)'>",
    "<body onload=alert('XSS')>",
    "<%2Fscript%3E%3Cscript%3Ealert(1)%3C%2Fscript%3E",
]

PATH_TRAVERSAL = [
    '/../../../etc/passwd', '/..%2F..%2F..%2Fetc/passwd',
    '/....//....//....//etc/shadow', '/%2e%2e/%2e%2e/etc/passwd',
    '/static/..%252f..%252f..%252fetc/passwd',
    '/api/../admin/config', '/uploads/../../private/keys',
]

CREDENTIAL_USERNAMES = [
    'admin', 'root', 'test', 'user', 'administrator', 'support',
    'info', 'marketing', 'sales', 'hr', 'finance', 'dev',
]


def _random_ip():
    return '%d.%d.%d.%d' % (
        random.randint(1, 223), random.randint(0, 255),
        random.randint(0, 255), random.randint(1, 254)
    )


def _random_timestamp(base_ts, offset_range=3600):
    ts = base_ts + random.randint(0, offset_range)
    t = time.gmtime(ts)
    return time.strftime('%d/%b/%Y:%H:%M:%S +0000', t), ts


def generate_benign_entry(base_ts):
    """Generate one benign log entry."""
    ts_str, ts = _random_timestamp(base_ts)
    return {
        'timestamp': ts_str,
        'client_ip': _random_ip(),
        'method': random.choice(['GET', 'GET', 'GET', 'POST', 'HEAD']),
        'path': random.choice(BENIGN_PATHS),
        'status': random.choice([200, 200, 200, 200, 301, 304]),
        'bytes': random.randint(200, 50000),
        'user_agent': random.choice(USER_AGENTS),
        'referrer': random.choice(['-', 'https://example.com', 'https://google.com']),
        'edge_pop': random.choice(EDGE_POPS),
        'waf_action': 'pass',
        'attack_type': None,
        'attack_label': False,
        '_ts': ts,
    }


def generate_sqli_entry(base_ts):
    ts_str, ts = _random_timestamp(base_ts)
    payload = random.choice(SQLI_PAYLOADS)
    path = random.choice([
        '/login?username=%s' % payload,
        '/api/v1/users?id=%s' % payload,
        '/search?q=%s' % payload,
    ])
    return {
        'timestamp': ts_str,
        'client_ip': _random_ip(),
        'method': random.choice(['GET', 'POST']),
        'path': path,
        'status': random.choice([200, 400, 403, 500]),
        'bytes': random.randint(100, 5000),
        'user_agent': random.choice(USER_AGENTS),
        'referrer': '-',
        'edge_pop': random.choice(EDGE_POPS),
        'waf_action': random.choice(['block', 'log']),
        'attack_type': 'sqli',
        'attack_label': True,
        '_ts': ts,
    }


def generate_xss_entry(base_ts):
    ts_str, ts = _random_timestamp(base_ts)
    payload = random.choice(XSS_PAYLOADS)
    path = '/search?q=%s' % payload
    return {
        'timestamp': ts_str,
        'client_ip': _random_ip(),
        'method': 'GET',
        'path': path,
        'status': random.choice([200, 400]),
        'bytes': random.randint(100, 3000),
        'user_agent': random.choice(USER_AGENTS),
        'referrer': '-',
        'edge_pop': random.choice(EDGE_POPS),
        'waf_action': random.choice(['block', 'log']),
        'attack_type': 'xss',
        'attack_label': True,
        '_ts': ts,
    }


def generate_traversal_entry(base_ts):
    ts_str, ts = _random_timestamp(base_ts)
    return {
        'timestamp': ts_str,
        'client_ip': _random_ip(),
        'method': 'GET',
        'path': random.choice(PATH_TRAVERSAL),
        'status': random.choice([200, 400, 403, 404]),
        'bytes': random.randint(0, 1000),
        'user_agent': random.choice(USER_AGENTS + SCANNER_UAS),
        'referrer': '-',
        'edge_pop': random.choice(EDGE_POPS),
        'waf_action': random.choice(['block', 'log']),
        'attack_type': 'path_traversal',
        'attack_label': True,
        '_ts': ts,
    }


def generate_credential_stuffing_burst(base_ts, burst_size=20):
    """Generate a burst of credential stuffing attempts from one IP."""
    entries = []
    ip = _random_ip()
    for i in range(burst_size):
        ts_str, ts = _random_timestamp(base_ts, offset_range=60)
        entries.append({
            'timestamp': ts_str,
            'client_ip': ip,
            'method': 'POST',
            'path': '/login',
            'status': random.choice([401, 401, 401, 403, 200]),
            'bytes': random.randint(100, 500),
            'user_agent': random.choice(USER_AGENTS[:3]),
            'referrer': '-',
            'edge_pop': random.choice(EDGE_POPS[:3]),
            'waf_action': random.choice(['log', 'challenge', 'block']),
            'attack_type': 'credential_stuffing',
            'attack_label': True,
            '_ts': ts,
        })
    return entries


def generate_scanner_sweep(base_ts, sweep_size=30):
    """Generate a scanner sweep (Nikto/sqlmap/etc.)."""
    entries = []
    ip = _random_ip()
    ua = random.choice(SCANNER_UAS)
    scan_paths = [
        '/admin', '/wp-admin', '/phpmyadmin', '/.env', '/config.php',
        '/backup.sql', '/db.sqlite', '/.git/HEAD', '/server-status',
        '/wp-login.php', '/xmlrpc.php', '/cgi-bin/test.cgi',
        '/actuator/health', '/api/swagger.json', '/debug/pprof',
        '/.well-known/security.txt', '/crossdomain.xml',
    ]
    for i in range(min(sweep_size, len(scan_paths))):
        ts_str, ts = _random_timestamp(base_ts, offset_range=120)
        entries.append({
            'timestamp': ts_str,
            'client_ip': ip,
            'method': 'GET',
            'path': scan_paths[i],
            'status': random.choice([404, 404, 404, 403, 200]),
            'bytes': random.randint(0, 1000),
            'user_agent': ua,
            'referrer': '-',
            'edge_pop': random.choice(EDGE_POPS),
            'waf_action': random.choice(['log', 'block']),
            'attack_type': 'scanner',
            'attack_label': True,
            '_ts': ts,
        })
    return entries


def entry_to_clf(entry):
    """Convert entry dict to CLF (Common Log Format) string."""
    return '%s - - [%s] "%s %s HTTP/1.1" %d %d "%s" "%s" pop=%s waf=%s' % (
        entry['client_ip'], entry['timestamp'], entry['method'],
        entry['path'], entry['status'], entry['bytes'],
        entry['referrer'], entry['user_agent'],
        entry['edge_pop'], entry['waf_action'],
    )


def generate_logs(n=1000, attack_ratio=0.15, seed=42):
    """Generate synthetic log entries.

    Parameters
    ----------
    n : int  total entries
    attack_ratio : float
    seed : int

    Returns
    -------
    list of dict
    """
    random.seed(seed)
    base_ts = 1729760130  # 2024-10-24 fixed timestamp

    n_attacks = int(n * attack_ratio)
    n_benign = n - n_attacks

    entries = []

    # Benign traffic
    for _ in range(n_benign):
        entries.append(generate_benign_entry(base_ts))

    # Attacks (distributed among types)
    attack_budget = n_attacks
    generators = [generate_sqli_entry, generate_xss_entry, generate_traversal_entry]

    # A few bursts
    n_bursts = max(1, n_attacks // 30)
    for _ in range(n_bursts):
        burst = generate_credential_stuffing_burst(base_ts)
        entries.extend(burst[:min(len(burst), attack_budget // 4)])
        attack_budget -= len(burst)

    # A scanner sweep
    if attack_budget > 10:
        sweep = generate_scanner_sweep(base_ts)
        entries.extend(sweep[:min(len(sweep), attack_budget // 3)])
        attack_budget -= len(sweep)

    # Individual attacks
    while attack_budget > 0:
        gen = random.choice(generators)
        entries.append(gen(base_ts))
        attack_budget -= 1

    # Sort by timestamp
    entries.sort(key=lambda e: e.get('_ts', 0))

    return entries


def main():
    parser = argparse.ArgumentParser(
        description='Synthetic edge-server log generator\n' + BANNER)
    parser.add_argument('--n', type=int, default=1000,
                        help='Number of log entries')
    parser.add_argument('--attack-ratio', type=float, default=0.15,
                        help='Fraction of attack entries')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--format', type=str, default='jsonl',
                        choices=['jsonl', 'clf', 'both'])
    args = parser.parse_args()

    print(BANNER)
    entries = generate_logs(n=args.n, attack_ratio=args.attack_ratio, seed=args.seed)

    n_attacks = sum(1 for e in entries if e['attack_label'])
    n_benign = len(entries) - n_attacks
    attack_types = {}
    for e in entries:
        if e['attack_label']:
            t = e.get('attack_type', 'unknown')
            attack_types[t] = attack_types.get(t, 0) + 1

    print("\nGenerated %d entries: %d benign, %d attacks" % (
        len(entries), n_benign, n_attacks))
    print("Attack breakdown: %s" % json.dumps(attack_types))

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'synthetic_logs'
        )

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    if args.format in ('jsonl', 'both'):
        jsonl_path = args.output + '.jsonl'
        with open(jsonl_path, 'w') as f:
            f.write('# %s\n' % BANNER)
            for entry in entries:
                out = dict(entry)
                out.pop('_ts', None)
                f.write(json.dumps(out) + '\n')
        print("JSONL output: %s" % jsonl_path)

    if args.format in ('clf', 'both'):
        clf_path = args.output + '.log'
        with open(clf_path, 'w') as f:
            f.write('# %s\n' % BANNER)
            for entry in entries:
                f.write(entry_to_clf(entry) + '\n')
        print("CLF output: %s" % clf_path)

    # Ground truth labels
    labels_path = args.output + '_labels.json'
    labels = []
    for i, entry in enumerate(entries):
        if entry['attack_label']:
            labels.append({
                'line': i,
                'attack_type': entry['attack_type'],
                'client_ip': entry['client_ip'],
                'path': entry['path'],
            })

    with open(labels_path, 'w') as f:
        json.dump({
            'banner': BANNER,
            'total_entries': len(entries),
            'n_attacks': n_attacks,
            'attack_types': attack_types,
            'labels': labels,
        }, f, indent=2)
    print("Labels: %s" % labels_path)


if __name__ == '__main__':
    main()
