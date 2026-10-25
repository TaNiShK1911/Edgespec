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
#  WP4: Digest-based prompt builder (facts + held-out few-shot)
# --------------------------------------------------------------------------- #

# Held-out few-shot examples — use DIFFERENT attack types from the CSV
# (command injection, SSRF) so SQLi/XSS/traversal payloads in the CSV
# are not leaked into the few-shot template.  See plan section WP4.

DIGEST_TRIAGE_SYSTEM = """You are a security analyst reviewing edge-server access logs.
%s
You receive a facts block summarising a 60-second traffic window.
Analyse it and fill in each section below.  Use ONLY data from the facts block.
Do NOT invent IPs, paths, or numbers that are not in the facts.

### FORMAT
Traffic overview: [one-sentence summary of volume and status mix]
Notable signatures: [top 3 request patterns by count]
Attack families: [attack types detected, or "none detected"]
Recommended action: [brief action item]
### END REPORT
""" % BANNER

DIGEST_TRIAGE_FEW_SHOT = """### EXAMPLE FACTS
Window: 312 requests from 45 distinct IPs
Status mix: 2xx=280, 4xx=30, 5xx=2
Detected patterns: credential_stuffing (15), scanner (8)
Top signatures:
  1. GET /api/v1/health -> 200  [120x, 35 IPs]
  2. POST /login -> 401  [15x, 1 IP]
  3. GET /admin -> 404  [8x, 1 IP]
Busiest source IPs: 10.0.0.50, 192.168.1.1

### ANALYSIS REPORT
Traffic overview: 312 requests from 45 IPs; 90% success (2xx=280), 30 client errors, 2 server errors.
Notable signatures: /api/v1/health (120 hits, normal health checks), POST /login 401 (15 hits from 1 IP, credential stuffing), GET /admin 404 (8 hits from 1 IP, scanner probe).
Attack families: credential_stuffing (15 attempts from 10.0.0.50), scanner (8 probes targeting /admin).
Recommended action: Block 10.0.0.50 for credential stuffing; add rate limit on /login; monitor scanner IP.
### END REPORT

"""

DIGEST_WAF_SYSTEM = """You are a WAF rule engineer.  Given attack patterns from log analysis,
write a Python-compatible regex to detect each attack family.
%s
The regex must be valid Python re syntax and match the malicious part of the request.
Output one regex per attack family.

### FORMAT
Attack family: <name>
Sample: <example request path>
Regex: <python regex pattern>

""" % BANNER

# Held-out few-shot: command injection and SSRF (NOT in the CSV data)
DIGEST_WAF_FEW_SHOT = """### EXAMPLE
Attack family: command_injection
Sample: GET /api/run?cmd=;cat /etc/passwd
Regex: ([;|`]\\s*(cat|ls|whoami|id|wget|curl)\\s)

Attack family: ssrf
Sample: GET /proxy?url=http://169.254.169.254/latest/meta-data/
Regex: (https?://(169\\.254\\.169\\.254|localhost|127\\.0\\.0\\.1|0\\.0\\.0\\.0|\\[::1\\]))

"""

# Template fallback regexes (WP5: visible "FALLBACK" tag)
FALLBACK_REGEXES = {
    'sqli': r"(?i)('\\s*(OR|AND|UNION|SELECT|DROP|INSERT|DELETE)\\s|--\\s*$|#\\s*$)",
    'xss': r"(?i)(<script|onerror\\s*=|onload\\s*=|javascript:|<iframe|<svg\\s+on)",
    'path_traversal': r"(\\.\\./|\\.\\.\\\\/|%2e%2e|%252f)",
    'credential_stuffing': r"(POST\\s+/login\\s.*\\s40[13]\\s)",
    'scanner': r"(?i)(/\\.env|/\\.git/|/wp-admin|/phpmyadmin|/backup\\.sql|/server-status)",
    'rate_limit': r"\\s429\\s",
}


def _count_tokens_fast(text):
    """Quick token estimate without loading the tokenizer."""
    return max(len(text.split()), int(len(text) / 3.5))


def _count_tokens(text):
    """Count tokens using the repo tokenizer if available."""
    try:
        from edgespec.tokenizer import get_tokenizer
        tok = get_tokenizer()
        return len(tok.encode(text, add_bos=False))
    except (ImportError, OSError):
        return _count_tokens_fast(text)


def build_digest_summary_prompt(digest, max_gen=256, max_total=2048):
    """Build a Task A (summary) prompt from a digest record.

    Parameters
    ----------
    digest : dict
        One record from digests.jsonl.
    max_gen : int
        Token budget reserved for generation.
    max_total : int
        Maximum total tokens (OPT context window).

    Returns
    -------
    prompt : str
    fits : bool
        Whether the prompt fits within budget.
    token_count : int
    """
    system = DIGEST_TRIAGE_SYSTEM + DIGEST_TRIAGE_FEW_SHOT
    facts = digest.get('facts_text', '')

    prompt = system + "### FACTS\n" + facts + "\n\n### ANALYSIS REPORT\n"

    n_tokens = _count_tokens(prompt)
    fits = (n_tokens + max_gen) <= max_total

    return prompt, fits, n_tokens


def build_digest_waf_prompt(digest, max_gen=256, max_total=2048):
    """Build a Task B (WAF rule) prompt from a digest record.

    One prompt per attack family found in the window.

    Parameters
    ----------
    digest : dict
        One record from digests.jsonl.
    max_gen : int
    max_total : int

    Returns
    -------
    list of (prompt_str, attack_family, fits, token_count)
    """
    rule_tags = digest.get('rule_tags', {})
    top_sigs = digest.get('top_signatures', [])

    results = []
    system = DIGEST_WAF_SYSTEM + DIGEST_WAF_FEW_SHOT

    for family in sorted(rule_tags.keys()):
        if family in ('rate_limit',):
            # Rate limiting is not a WAF regex target
            continue

        # Find a sample request for this family
        sample = ''
        for sig in top_sigs:
            if sig.get('sample_full_paths'):
                sample = sig['sample_full_paths'][0]
                break

        prompt = system + ("### ATTACK FAMILY: %s\n" % family)
        prompt += "Sample: %s\n" % sample
        prompt += "Regex: "

        n_tokens = _count_tokens(prompt)
        fits = (n_tokens + 60) <= max_total  # WAF regexes are short (~40 tokens)
        results.append((prompt, family, fits, n_tokens))

    return results


def load_digests(digests_path=None):
    """Load digests.jsonl.

    Parameters
    ----------
    digests_path : str or None

    Returns
    -------
    list of dict
    """
    if digests_path is None:
        from edgespec.config import DATA_DIR
        digests_path = os.path.join(DATA_DIR, 'digests.jsonl')

    digests = []
    if os.path.isfile(digests_path):
        with open(digests_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    digests.append(json.loads(line))
                except (ValueError, KeyError):
                    continue
    return digests


def select_demo_windows(digests):
    """Select 3 demo windows per section 6 of the plan.

    Picks: (1) quietest normal window, (2) rate-limit burst window,
    (3) highest attack count window.

    Parameters
    ----------
    digests : list of dict

    Returns
    -------
    list of dict  (3 digests)
    """
    if not digests:
        return []

    # (1) Quietest: fewest rows, no attacks
    normal_only = [d for d in digests if not d.get('rule_tags')]
    if normal_only:
        quietest = min(normal_only, key=lambda d: d['n_rows'])
    else:
        quietest = min(digests, key=lambda d: d['n_rows'])

    # (2) Burst: most rows (the rate-limit burst window)
    burst = max(digests, key=lambda d: d['n_rows'])

    # (3) Highest attack count
    highest_attack = max(digests, key=lambda d: d.get('n_attack_lines', 0))

    return [quietest, burst, highest_attack]


# --------------------------------------------------------------------------- #
#  WP5: Grounding validator, regex-with-fallback, and scorer
# --------------------------------------------------------------------------- #

def validate_grounding(output_text, facts_text):
    """Check that every IP, path, and number in the output appears in the facts.

    Parameters
    ----------
    output_text : str
        Model-generated summary.
    facts_text : str
        Facts block that was in the prompt.

    Returns
    -------
    dict with:
        grounded_count, total_entities, grounded_fraction,
        ungrounded_entities, stripped_output
    """
    # Extract IPs from output
    ip_re = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
    output_ips = set(ip_re.findall(output_text))
    facts_ips = set(ip_re.findall(facts_text))

    # Extract paths from output (starting with /)
    path_re = re.compile(r'(/[a-zA-Z0-9_./-]+)')
    output_paths = set(path_re.findall(output_text))
    facts_paths = set(path_re.findall(facts_text))

    # Extract numbers (counts, percentages)
    num_re = re.compile(r'\b(\d{2,})\b')
    output_nums = set(num_re.findall(output_text))
    facts_nums = set(num_re.findall(facts_text))

    grounded = 0
    total = 0
    ungrounded = []

    for ip in output_ips:
        total += 1
        if ip in facts_ips:
            grounded += 1
        else:
            ungrounded.append('IP: %s' % ip)

    for path in output_paths:
        total += 1
        if path in facts_paths:
            grounded += 1
        else:
            ungrounded.append('path: %s' % path)

    for num in output_nums:
        total += 1
        if num in facts_nums:
            grounded += 1
        # Numbers are less strict -- model may compute sums
        # Don't count as ungrounded

    fraction = (grounded / float(total)) if total > 0 else 1.0

    # Strip ungrounded entities from output
    stripped = output_text
    for entity_desc in ungrounded:
        if entity_desc.startswith('IP: '):
            ip = entity_desc[4:]
            stripped = stripped.replace(ip, '[UNGROUNDED]')

    return {
        'grounded_count': grounded,
        'total_entities': total,
        'grounded_fraction': fraction,
        'ungrounded_entities': ungrounded,
        'stripped_output': stripped,
    }


def validate_regex_with_fallback(regex_str, family=None):
    """Validate a regex string and fall back to template if it fails.

    Parameters
    ----------
    regex_str : str
    family : str or None
        Attack family name for fallback lookup.

    Returns
    -------
    dict with: regex, compiled, used_fallback, error
    """
    regex_str = regex_str.strip()

    # Try to compile
    try:
        re.compile(regex_str)
        return {
            'regex': regex_str,
            'compiled': True,
            'used_fallback': False,
            'error': None,
        }
    except re.error as e:
        error_msg = str(e)

    # Fallback
    if family and family in FALLBACK_REGEXES:
        fallback = FALLBACK_REGEXES[family]
        try:
            re.compile(fallback)
            return {
                'regex': fallback,
                'compiled': True,
                'used_fallback': True,
                'error': 'Original failed: %s. FALLBACK USED.' % error_msg,
            }
        except re.error:
            pass

    return {
        'regex': regex_str,
        'compiled': False,
        'used_fallback': False,
        'error': error_msg,
    }


def evaluate_waf_on_window(regexes, attack_clf_lines, benign_clf_lines):
    """Evaluate WAF regexes on a window's attack and benign lines.

    Parameters
    ----------
    regexes : list of dict
        From validate_regex_with_fallback, each with 'regex' key.
    attack_clf_lines : list of str
    benign_clf_lines : list of str

    Returns
    -------
    dict with recall, fpr, true_positives, false_positives, etc.
    """
    compiled = []
    for r in regexes:
        if r.get('compiled'):
            try:
                compiled.append(re.compile(r['regex']))
            except re.error:
                continue

    if not compiled:
        return {
            'recall': 0.0, 'fpr': 0.0,
            'true_positives': 0, 'false_positives': 0,
            'n_attack': len(attack_clf_lines),
            'n_benign': len(benign_clf_lines),
            'n_regexes': 0,
        }

    tp = 0
    for line in attack_clf_lines:
        for pat in compiled:
            if pat.search(line):
                tp += 1
                break

    fp = 0
    for line in benign_clf_lines:
        for pat in compiled:
            if pat.search(line):
                fp += 1
                break

    recall = (tp / float(len(attack_clf_lines)) * 100.0) if attack_clf_lines else 0.0
    fpr = (fp / float(len(benign_clf_lines)) * 100.0) if benign_clf_lines else 0.0

    return {
        'recall': recall,
        'fpr': fpr,
        'true_positives': tp,
        'false_positives': fp,
        'n_attack': len(attack_clf_lines),
        'n_benign': len(benign_clf_lines),
        'n_regexes': len(compiled),
    }


def score_window(summary_output, waf_outputs, digest,
                 attack_clf_lines=None, benign_clf_lines=None):
    """Comprehensive scoring for one demo window.

    Parameters
    ----------
    summary_output : str or None
        Model output for Task A (summary).
    waf_outputs : list of dict or None
        Each dict: {family, regex_str, ...}
    digest : dict
        The digest record for this window.
    attack_clf_lines : list of str or None
    benign_clf_lines : list of str or None

    Returns
    -------
    dict with all scores
    """
    scores = {
        'window_index': digest.get('window_index', -1),
        'window_start': digest.get('window_start', ''),
        'n_rows': digest.get('n_rows', 0),
    }

    # Task A: grounding check
    if summary_output:
        grounding = validate_grounding(summary_output, digest.get('facts_text', ''))
        scores['grounding'] = grounding
        scores['grounded_fraction'] = grounding['grounded_fraction']
    else:
        scores['grounding'] = None
        scores['grounded_fraction'] = None

    # Task B: regex validation
    if waf_outputs:
        validated_regexes = []
        n_compiled = 0
        n_fallback = 0
        for waf in waf_outputs:
            family = waf.get('family', '')
            regex_str = waf.get('regex_str', '')
            result = validate_regex_with_fallback(regex_str, family)
            result['family'] = family
            validated_regexes.append(result)
            if result['compiled']:
                n_compiled += 1
            if result.get('used_fallback'):
                n_fallback += 1

        compile_rate = (n_compiled / float(len(validated_regexes)) * 100.0
                        if validated_regexes else 0.0)
        scores['regex_compile_rate'] = compile_rate
        scores['regex_fallback_count'] = n_fallback
        scores['validated_regexes'] = validated_regexes

        # Evaluate recall/FPR if lines are provided
        if attack_clf_lines is not None and benign_clf_lines is not None:
            waf_eval = evaluate_waf_on_window(
                validated_regexes, attack_clf_lines, benign_clf_lines)
            scores['waf_eval'] = waf_eval
        else:
            scores['waf_eval'] = None
    else:
        scores['regex_compile_rate'] = None
        scores['regex_fallback_count'] = None
        scores['validated_regexes'] = None
        scores['waf_eval'] = None

    return scores


def save_demo_results(results, output_dir=None):
    """Save demo scoring results to results/demo_<ts>.json.

    Parameters
    ----------
    results : dict or list
        Scoring results.
    output_dir : str or None

    Returns
    -------
    str  path to saved file
    """
    import time

    if output_dir is None:
        from edgespec.config import RESULTS_DIR
        output_dir = RESULTS_DIR

    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    ts = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(output_dir, 'demo_%s.json' % ts)

    output = {
        'banner': BANNER,
        'timestamp': ts,
        'results': results,
    }

    with open(path, 'w') as f:
        json.dump(output, f, indent=2)

    return path


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main():
    """Generate demo prompts and validate sample output."""
    import argparse

    parser = argparse.ArgumentParser(description='Akamai demo prompt/validator')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--task', choices=['triage', 'waf', 'digest', 'both', 'all'],
                        default='both')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--digests', type=str, default=None,
                        help='Path to digests.jsonl for digest-based prompts')
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

    if args.task in ('digest', 'all'):
        digests = load_digests(args.digests)
        if digests:
            demo_windows = select_demo_windows(digests)
            print("\n=== DIGEST-BASED PROMPTS (WP4) ===")
            print("Selected %d demo windows" % len(demo_windows))

            all_fit = True
            for dw in demo_windows:
                prompt, fits, n_tok = build_digest_summary_prompt(dw)
                status = "FITS" if fits else "OVER BUDGET"
                if not fits:
                    all_fit = False
                print("  Window %d (%s): %d rows, %d tokens (%s)" % (
                    dw.get('window_index', -1), dw.get('window_start', '?'),
                    dw.get('n_rows', 0), n_tok, status))

                waf_prompts = build_digest_waf_prompt(dw)
                for wp, family, wf, wt in waf_prompts:
                    ws = "FITS" if wf else "OVER BUDGET"
                    if not wf:
                        all_fit = False
                    print("    WAF %s: %d tokens (%s)" % (family, wt, ws))

            print("\nWP4 acceptance: %s (all prompts fit in 2048)" % (
                "PASS" if all_fit else "FAIL"))
        else:
            print("No digests found. Run build_digests.py first.")

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
