#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Akamai edge-security demo: prompts, validator, and evaluator (P3-6).

Provides:
- Few-shot prompt templates for Task A (log triage) and Task B (WAF rule drafting)
- Output validators (regex compile, recall, FPR)
- Evaluation against ground-truth labels from gen_logs.py

All data is SYNTHETIC — banner embedded in every prompt.

Python-3.6 safe.
"""

from __future__ import print_function

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BANNER = "*** SYNTHETIC DATA — NOT FROM REAL TRAFFIC ***"


# --------------------------------------------------------------------------- #
#  Prompt templates
# --------------------------------------------------------------------------- #

TRIAGE_SYSTEM = """You are a security analyst reviewing edge-server access logs.
%s
Analyse the log window below and report:
1. Top 3 offending client IPs with attack counts
2. Most targeted paths
3. Attack types observed (SQLi, XSS, path traversal, credential stuffing, scanner)
4. Recommended actions

### LOGS
""" % BANNER

TRIAGE_FEW_SHOT = """### EXAMPLE INPUT
192.168.1.100 - - [24/Oct/2024:12:00:01 +0000] "POST /login HTTP/1.1" 401 312 "-" "python-requests/2.31.0" pop=IAD waf=log
192.168.1.100 - - [24/Oct/2024:12:00:02 +0000] "POST /login HTTP/1.1" 401 312 "-" "python-requests/2.31.0" pop=IAD waf=log
10.0.0.5 - - [24/Oct/2024:12:00:03 +0000] "GET /search?q=' OR '1'='1 HTTP/1.1" 400 156 "-" "curl/8.4.0" pop=SFO waf=block
10.0.0.5 - - [24/Oct/2024:12:00:04 +0000] "GET /api/v1/users?id=' UNION SELECT username, password FROM users-- HTTP/1.1" 403 0 "-" "curl/8.4.0" pop=SFO waf=block

### FINDINGS
1. **Top offending IPs:**
   - 192.168.1.100 (2 hits) — credential stuffing (rapid POST /login with 401s)
   - 10.0.0.5 (2 hits) — SQL injection attempts (union/OR payloads)
2. **Most targeted paths:** /login (credential stuffing), /search, /api/v1/users (SQLi)
3. **Attack types:** Credential stuffing, SQL injection
4. **Recommended actions:** Block 192.168.1.100 and 10.0.0.5; add WAF rules for SQLi patterns in query params

"""

WAF_SYSTEM = """You are a WAF rule engineer. Given attack log lines, write regex patterns to detect them.
%s
For each attack line, output a regex that would match the malicious part of the request path/query.
The regex must be valid Python re syntax.

### FORMAT
Attack: <log line>
Regex: <python regex pattern>

""" % BANNER

WAF_FEW_SHOT = """### EXAMPLE
Attack: GET /search?q=' OR '1'='1 HTTP/1.1
Regex: ('\\s*OR\\s+'\\d+'\\s*=\\s*'\\d+)

Attack: GET /search?q=<script>alert('XSS')</script> HTTP/1.1
Regex: (<script[^>]*>.*?</script>)

Attack: GET /../../../etc/passwd HTTP/1.1
Regex: (\\.\\.[\\/]+)

"""


def build_triage_prompt(log_lines, max_lines=200):
    """Build a triage prompt from log lines.

    Parameters
    ----------
    log_lines : list of str
        CLF-format log lines.
    max_lines : int

    Returns
    -------
    str
    """
    window = log_lines[:max_lines]
    logs_text = '\n'.join(window)
    prompt = TRIAGE_SYSTEM + TRIAGE_FEW_SHOT + "### LOGS\n" + logs_text + "\n\n### FINDINGS\n"
    return prompt


def build_waf_prompt(attack_lines, max_attacks=10):
    """Build a WAF rule drafting prompt from attack lines.

    Parameters
    ----------
    attack_lines : list of str
        CLF-format attack log lines.
    max_attacks : int

    Returns
    -------
    str
    """
    selected = attack_lines[:max_attacks]
    attacks_text = '\n'.join('Attack: %s' % line for line in selected)
    prompt = WAF_SYSTEM + WAF_FEW_SHOT + "### ATTACK LINES\n" + attacks_text + "\n\nRegex: "
    return prompt


# --------------------------------------------------------------------------- #
#  Validators
# --------------------------------------------------------------------------- #

def validate_waf_regexes(output_text):
    """Extract and validate regex patterns from model output.

    Parameters
    ----------
    output_text : str

    Returns
    -------
    results : list of dict
        Each dict: {regex_str, compiled, error}
    compile_rate : float
        Fraction that compiled successfully.
    """
    # Extract regex patterns from "Regex: <pattern>" lines
    pattern = re.compile(r'Regex:\s*(.+?)(?:\n|$)')
    matches = pattern.findall(output_text)

    results = []
    compiled_count = 0

    for regex_str in matches:
        regex_str = regex_str.strip()
        try:
            re.compile(regex_str)
            results.append({
                'regex_str': regex_str,
                'compiled': True,
                'error': None,
            })
            compiled_count += 1
        except re.error as e:
            results.append({
                'regex_str': regex_str,
                'compiled': False,
                'error': str(e),
            })

    total = len(results)
    compile_rate = (compiled_count / total * 100.0) if total > 0 else 0.0

    return results, compile_rate


def evaluate_waf_recall(regexes, attack_lines, benign_lines):
    """Evaluate WAF regex recall and false positive rate.

    Parameters
    ----------
    regexes : list of str
        Compiled regex pattern strings.
    attack_lines : list of str
    benign_lines : list of str

    Returns
    -------
    dict with recall, fpr, true_positives, false_positives, etc.
    """
    compiled = []
    for r in regexes:
        try:
            compiled.append(re.compile(r))
        except re.error:
            continue

    if not compiled:
        return {
            'recall': 0.0,
            'fpr': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'n_attack': len(attack_lines),
            'n_benign': len(benign_lines),
        }

    # True positives: attack lines matched by at least one regex
    tp = 0
    for line in attack_lines:
        for pat in compiled:
            if pat.search(line):
                tp += 1
                break

    # False positives: benign lines matched by at least one regex
    fp = 0
    for line in benign_lines:
        for pat in compiled:
            if pat.search(line):
                fp += 1
                break

    recall = (tp / len(attack_lines) * 100.0) if attack_lines else 0.0
    fpr = (fp / len(benign_lines) * 100.0) if benign_lines else 0.0

    return {
        'recall': recall,
        'fpr': fpr,
        'true_positives': tp,
        'false_positives': fp,
        'n_attack': len(attack_lines),
        'n_benign': len(benign_lines),
    }


def evaluate_triage(output_text, ground_truth_labels):
    """Evaluate triage output against ground truth.

    Checks if the model named the top offending IP correctly.

    Parameters
    ----------
    output_text : str
    ground_truth_labels : dict
        From gen_logs.py labels JSON.

    Returns
    -------
    dict with accuracy metrics.
    """
    # Find the actual top offender from ground truth
    ip_counts = {}
    for label in ground_truth_labels.get('labels', []):
        ip = label.get('client_ip', '')
        ip_counts[ip] = ip_counts.get(ip, 0) + 1

    if not ip_counts:
        return {'top_offender_found': False, 'top_offender_ip': None}

    top_ip = max(ip_counts, key=ip_counts.get)

    # Check if the model mentioned this IP
    found = top_ip in output_text

    return {
        'top_offender_found': found,
        'top_offender_ip': top_ip,
        'top_offender_count': ip_counts[top_ip],
        'output_mentions_ip': found,
    }


def load_demo_data(data_dir=None):
    """Load synthetic log data and labels.

    Parameters
    ----------
    data_dir : str or None

    Returns
    -------
    clf_lines : list of str
    labels : dict
    attack_lines : list of str
    benign_lines : list of str
    """
    if data_dir is None:
        from edgespec.config import DATA_DIR
        data_dir = DATA_DIR

    clf_path = os.path.join(data_dir, 'synthetic_logs.log')
    labels_path = os.path.join(data_dir, 'synthetic_logs_labels.json')
    jsonl_path = os.path.join(data_dir, 'synthetic_logs.jsonl')

    # Load CLF lines
    clf_lines = []
    if os.path.isfile(clf_path):
        with open(clf_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    clf_lines.append(line)

    # Load labels
    labels = {}
    if os.path.isfile(labels_path):
        with open(labels_path, 'r') as f:
            labels = json.load(f)

    # Separate attack and benign lines from JSONL
    attack_lines = []
    benign_lines = []
    if os.path.isfile(jsonl_path):
        with open(jsonl_path, 'r') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    entry = json.loads(line)
                    clf_line = clf_lines[line_idx] if line_idx < len(clf_lines) else ''
                    if entry.get('attack_label'):
                        attack_lines.append(clf_line)
                    else:
                        benign_lines.append(clf_line)
                except (json.JSONDecodeError, ValueError):
                    continue

    return clf_lines, labels, attack_lines, benign_lines


def create_demo_windows(clf_lines, window_size=200, n_windows=20):
    """Create sliding windows of log lines for triage evaluation.

    Parameters
    ----------
    clf_lines : list of str
    window_size : int
    n_windows : int

    Returns
    -------
    list of list of str
    """
    windows = []
    step = max(1, (len(clf_lines) - window_size) // max(1, n_windows - 1))
    for start in range(0, len(clf_lines) - window_size + 1, step):
        windows.append(clf_lines[start:start + window_size])
        if len(windows) >= n_windows:
            break
    # If not enough windows, pad with the last window
    while len(windows) < n_windows and clf_lines:
        windows.append(clf_lines[-window_size:])
    return windows


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main():
    """Generate demo prompts and validate sample output."""
    import argparse

    parser = argparse.ArgumentParser(description='Akamai demo prompt/validator')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--task', choices=['triage', 'waf', 'both'], default='both')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    clf_lines, labels, attack_lines, benign_lines = load_demo_data(args.data_dir)
    print(BANNER)
    print("Loaded %d log lines (%d attack, %d benign)" % (
        len(clf_lines), len(attack_lines), len(benign_lines)))

    if args.task in ('triage', 'both'):
        prompt = build_triage_prompt(clf_lines)
        print("\n=== TRIAGE PROMPT (%d chars) ===" % len(prompt))
        print(prompt[:500] + '...\n')

    if args.task in ('waf', 'both'):
        prompt = build_waf_prompt(
            [l for l in clf_lines if 'waf=block' in l or 'waf=log' in l][:10]
        )
        print("\n=== WAF PROMPT (%d chars) ===" % len(prompt))
        print(prompt[:500] + '...\n')

    if args.output:
        windows = create_demo_windows(clf_lines)
        out = {
            'banner': BANNER,
            'n_windows': len(windows),
            'n_attack_lines': len(attack_lines),
            'n_benign_lines': len(benign_lines),
            'sample_triage_prompt': build_triage_prompt(windows[0])[:2000],
            'sample_waf_prompt': build_waf_prompt(attack_lines[:5])[:2000],
        }
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print("Output: %s" % args.output)


if __name__ == '__main__':
    main()
