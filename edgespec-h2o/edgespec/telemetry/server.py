# -*- coding: utf-8 -*-
"""SSE telemetry server for the EdgeSpec-H2O live dashboard (P3-5).

Runs on the Nano, streams JSON events over Server-Sent Events (SSE).
The browser UI runs on the laptop connecting to the Nano's IP.

Uses stdlib only (ThreadingMixIn + BaseHTTPRequestHandler) because
Flask 2.0.x is the last line supporting Py3.6 and may not be available.

Python-3.6 safe.
"""

from __future__ import print_function

import json
import os
import sys
import time
import threading

# Python 2/3 compat for HTTP server
try:
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    from socketserver import ThreadingMixIn
except ImportError:
    from BaseHTTPServer import HTTPServer
    from SimpleHTTPServer import SimpleHTTPRequestHandler
    from SocketServer import ThreadingMixIn


class TelemetryStore(object):
    """Thread-safe telemetry data store."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            'tokens_per_sec': 0.0,
            'peak_ram_mib': 0.0,
            'kv_bytes': 0,
            'kv_mib': 0.0,
            'alpha_hat': 0.0,
            'mean_gamma': 0.0,
            'total_tokens': 0,
            'total_steps': 0,
            'cache_stats': {},
            'gamma_ctrl_stats': {},
            'recent_tokens': [],       # list of {id, text, status}
            'heatmap_data': [],        # H2O score heatmap
            'config': {},              # current config (budget, lambda, lenience)
        }
        self._subscribers = []

    def update(self, data):
        """Update telemetry data and notify subscribers."""
        with self._lock:
            self._data.update(data)
            event_data = json.dumps(self._data)

        # Notify SSE subscribers
        dead = []
        for i, q in enumerate(self._subscribers):
            try:
                q.append(event_data)
            except Exception:
                dead.append(i)
        for i in reversed(dead):
            self._subscribers.pop(i)

    def get(self):
        """Get current telemetry snapshot."""
        with self._lock:
            return dict(self._data)

    def subscribe(self):
        """Subscribe to SSE updates. Returns a list that will be appended to."""
        q = []
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q):
        """Unsubscribe from SSE updates."""
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass


# Global store instance
_store = TelemetryStore()


def get_store():
    """Get the global telemetry store."""
    return _store


# Static HTML/JS/CSS for the dashboard
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')


class TelemetryHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the telemetry dashboard."""

    def log_message(self, format, *args):
        """Suppress default request logging."""
        pass

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self._serve_dashboard()
        elif self.path == '/api/stream':
            self._serve_sse()
        elif self.path == '/api/snapshot':
            self._serve_json(_store.get())
        elif self.path.startswith('/static/'):
            self._serve_static(self.path[8:])
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/api/config':
            self._handle_config_update()
        else:
            self.send_error(404)

    def _serve_dashboard(self):
        """Serve the main dashboard HTML."""
        html = _get_dashboard_html()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(html.encode('utf-8'))))
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def _serve_sse(self):
        """Server-Sent Events stream."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        q = _store.subscribe()
        try:
            while True:
                if q:
                    data = q.pop(0)
                    msg = 'data: %s\n\n' % data
                    self.wfile.write(msg.encode('utf-8'))
                    self.wfile.flush()
                else:
                    time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            _store.unsubscribe(q)

    def _serve_json(self, data):
        """Serve a JSON response."""
        body = json.dumps(data)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body.encode('utf-8'))

    def _serve_static(self, path):
        """Serve static files."""
        filepath = os.path.join(_STATIC_DIR, path)
        if not os.path.isfile(filepath):
            self.send_error(404)
            return

        content_types = {
            '.html': 'text/html',
            '.css': 'text/css',
            '.js': 'application/javascript',
            '.json': 'application/json',
            '.png': 'image/png',
            '.svg': 'image/svg+xml',
        }
        ext = os.path.splitext(path)[1]
        ct = content_types.get(ext, 'application/octet-stream')

        with open(filepath, 'rb') as f:
            data = f.read()
        self.send_response(200)
        self.send_header('Content-Type', ct)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_config_update(self):
        """Handle config update from the UI sliders."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            config = json.loads(body.decode('utf-8'))
            _store.update({'config_request': config})
            self._serve_json({'status': 'ok', 'config': config})
        except Exception as e:
            self._serve_json({'status': 'error', 'message': str(e)})


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server."""
    daemon_threads = True


def start_server(host='0.0.0.0', port=8765):
    """Start the telemetry server in a background thread.

    Parameters
    ----------
    host : str
    port : int

    Returns
    -------
    server : ThreadedHTTPServer
    thread : threading.Thread
    """
    server = ThreadedHTTPServer((host, port), TelemetryHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    print("[Telemetry] Server started on http://%s:%d" % (host, port))
    return server, thread


def _get_dashboard_html():
    """Return the full dashboard HTML as a string."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EdgeSpec-H2O Live Dashboard</title>
<style>
:root {
    --bg: #0d1117;
    --card-bg: #161b22;
    --border: #30363d;
    --text: #c9d1d9;
    --text-bright: #f0f6fc;
    --accent: #58a6ff;
    --green: #3fb950;
    --blue: #58a6ff;
    --orange: #d29922;
    --red: #f85149;
    --purple: #bc8cff;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text);
    padding: 16px; line-height: 1.5;
}
h1 { color: var(--text-bright); font-size: 1.5em; margin-bottom: 8px; }
h2 { color: var(--accent); font-size: 1.1em; margin-bottom: 8px; }
.banner {
    background: linear-gradient(90deg, #58a6ff22, #bc8cff22);
    border: 1px solid var(--accent); border-radius: 8px;
    padding: 12px 16px; margin-bottom: 16px; display: flex;
    align-items: center; gap: 12px;
}
.banner .status-dot {
    width: 12px; height: 12px; border-radius: 50%;
    background: var(--green); animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
}
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
.card {
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px;
}
.metric { font-size: 2em; font-weight: 700; color: var(--text-bright); }
.metric-label { font-size: 0.85em; color: var(--text); opacity: 0.7; }
.metric-unit { font-size: 0.5em; color: var(--accent); }
.tokens-stream {
    font-family: "Fira Code", "Consolas", monospace; font-size: 0.9em;
    background: #0d1117; border: 1px solid var(--border); border-radius: 4px;
    padding: 12px; min-height: 120px; max-height: 300px;
    overflow-y: auto; word-wrap: break-word; line-height: 1.8;
}
.tok-accepted { color: var(--green); }
.tok-corrected { color: var(--blue); }
.tok-rejected { color: var(--red); text-decoration: line-through; }
.slider-group { margin: 8px 0; }
.slider-group label { display: block; font-size: 0.85em; margin-bottom: 4px; }
.slider-group input[type=range] { width: 100%; }
.slider-val { float: right; color: var(--accent); font-weight: 600; }
.heatmap { display: flex; flex-wrap: wrap; gap: 2px; margin-top: 8px; }
.heatmap-cell {
    width: 8px; height: 8px; border-radius: 2px;
    transition: background 0.3s;
}
.bar-chart { display: flex; align-items: flex-end; gap: 4px; height: 60px; margin-top: 8px; }
.bar {
    flex: 1; background: var(--accent); border-radius: 2px 2px 0 0;
    min-width: 16px; transition: height 0.3s;
}
.bar-label { font-size: 0.6em; text-align: center; color: var(--text); opacity: 0.6; }
table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
td, th { padding: 4px 8px; border-bottom: 1px solid var(--border); text-align: left; }
th { color: var(--accent); }
</style>
</head>
<body>
<div class="banner">
    <div class="status-dot" id="statusDot"></div>
    <div>
        <h1>EdgeSpec-H2O</h1>
        <span style="opacity:0.7">Speculative Decoding + H2O KV Cache &mdash; Live Telemetry</span>
    </div>
</div>

<div class="grid">
    <div class="card">
        <div class="metric-label">Tokens/sec</div>
        <div class="metric" id="tokPerSec">0.0</div>
    </div>
    <div class="card">
        <div class="metric-label">Acceptance Rate (&alpha;&#770;)</div>
        <div class="metric" id="alphaHat">0.0<span class="metric-unit">%</span></div>
    </div>
    <div class="card">
        <div class="metric-label">Mean Accepted / Step</div>
        <div class="metric" id="meanAccepted">0.0</div>
    </div>
    <div class="card">
        <div class="metric-label">KV Cache</div>
        <div class="metric" id="kvMib">0.0<span class="metric-unit">MiB</span></div>
    </div>
    <div class="card">
        <div class="metric-label">Mean &gamma;</div>
        <div class="metric" id="meanGamma">0.0</div>
    </div>
    <div class="card">
        <div class="metric-label">Total Tokens</div>
        <div class="metric" id="totalTokens">0</div>
    </div>
</div>

<div class="grid" style="margin-top:16px">
    <div class="card" style="grid-column: span 2">
        <h2>Token Stream</h2>
        <div class="tokens-stream" id="tokenStream"></div>
    </div>
    <div class="card">
        <h2>Recent Acceptance</h2>
        <div class="bar-chart" id="acceptanceChart"></div>
    </div>
</div>

<div class="grid" style="margin-top:16px">
    <div class="card">
        <h2>H2O Score Heatmap</h2>
        <div class="heatmap" id="heatmap"></div>
    </div>
    <div class="card">
        <h2>Controls</h2>
        <div class="slider-group">
            <label>KV Budget: <span class="slider-val" id="budgetVal">20%</span></label>
            <input type="range" id="budgetSlider" min="10" max="100" value="20" step="5">
        </div>
        <div class="slider-group">
            <label>&lambda; (decay): <span class="slider-val" id="decayVal">0.98</span></label>
            <input type="range" id="decaySlider" min="90" max="100" value="98" step="1">
        </div>
        <div class="slider-group">
            <label>Lenience (l): <span class="slider-val" id="lenienceVal">1.0</span></label>
            <input type="range" id="lenienceSlider" min="30" max="100" value="100" step="5">
        </div>
    </div>
    <div class="card">
        <h2>Cache Stats</h2>
        <table id="cacheTable">
            <tr><th>Metric</th><th>Value</th></tr>
        </table>
    </div>
</div>

<script>
var evtSource = new EventSource("/api/stream");
var tokenStreamEl = document.getElementById("tokenStream");
var maxTokens = 500;

evtSource.onmessage = function(e) {
    var d = JSON.parse(e.data);

    document.getElementById("tokPerSec").textContent =
        (d.tokens_per_sec || 0).toFixed(1);
    document.getElementById("alphaHat").innerHTML =
        ((d.alpha_hat || 0) * 100).toFixed(1) + '<span class="metric-unit">%</span>';
    document.getElementById("meanAccepted").textContent =
        (d.mean_accepted_per_step || 0).toFixed(2);
    document.getElementById("kvMib").innerHTML =
        (d.kv_mib || 0).toFixed(1) + '<span class="metric-unit">MiB</span>';
    document.getElementById("meanGamma").textContent =
        (d.mean_gamma || 0).toFixed(1);
    document.getElementById("totalTokens").textContent =
        d.total_tokens || 0;

    // Token stream
    if (d.recent_tokens && d.recent_tokens.length > 0) {
        d.recent_tokens.forEach(function(t) {
            var span = document.createElement("span");
            span.className = "tok-" + (t.status || "accepted");
            span.textContent = t.text || (" " + t.id);
            tokenStreamEl.appendChild(span);
        });
        // Trim
        while (tokenStreamEl.childNodes.length > maxTokens) {
            tokenStreamEl.removeChild(tokenStreamEl.firstChild);
        }
        tokenStreamEl.scrollTop = tokenStreamEl.scrollHeight;
    }

    // Acceptance chart
    var chart = document.getElementById("acceptanceChart");
    var recent = d.recent_accepted || [];
    chart.innerHTML = "";
    var maxVal = Math.max.apply(null, recent.concat([1]));
    recent.forEach(function(v) {
        var bar = document.createElement("div");
        bar.className = "bar";
        bar.style.height = Math.max(4, (v / maxVal) * 56) + "px";
        bar.style.background = v > 0 ? "var(--green)" : "var(--red)";
        chart.appendChild(bar);
    });

    // Heatmap
    if (d.heatmap_data && d.heatmap_data.length > 0) {
        var hm = document.getElementById("heatmap");
        hm.innerHTML = "";
        d.heatmap_data.forEach(function(score) {
            var cell = document.createElement("div");
            cell.className = "heatmap-cell";
            var intensity = Math.min(1, score / (d.cache_stats && d.cache_stats.score_stats
                ? d.cache_stats.score_stats.max || 1 : 1));
            var r = Math.round(88 + intensity * 167);
            var g = Math.round(166 - intensity * 100);
            var b = Math.round(255 - intensity * 106);
            cell.style.background = "rgb(" + r + "," + g + "," + b + ")";
            hm.appendChild(cell);
        });
    }

    // Cache table
    if (d.cache_stats) {
        var ct = document.getElementById("cacheTable");
        ct.innerHTML = "<tr><th>Metric</th><th>Value</th></tr>";
        var rows = [
            ["Valid slots", d.cache_stats.n_valid || 0],
            ["Budget", d.cache_stats.budget || 0],
            ["Occupancy", ((d.cache_stats.occupancy_pct || 0).toFixed(1)) + "%"],
            ["KV MiB", ((d.cache_stats.kv_mib || 0).toFixed(2))],
            ["Decay", d.cache_stats.decay || 0],
        ];
        rows.forEach(function(row) {
            var tr = document.createElement("tr");
            tr.innerHTML = "<td>" + row[0] + "</td><td>" + row[1] + "</td>";
            ct.appendChild(tr);
        });
    }
};

evtSource.onerror = function() {
    document.getElementById("statusDot").style.background = "var(--red)";
};

// Slider controls
["budget", "decay", "lenience"].forEach(function(name) {
    var slider = document.getElementById(name + "Slider");
    var val = document.getElementById(name + "Val");
    slider.addEventListener("input", function() {
        var v = parseFloat(slider.value);
        if (name === "budget") val.textContent = v + "%";
        else if (name === "decay") val.textContent = (v / 100).toFixed(2);
        else val.textContent = (v / 100).toFixed(2);
    });
    slider.addEventListener("change", function() {
        var config = {};
        config[name] = parseFloat(slider.value);
        if (name === "decay" || name === "lenience") config[name] /= 100;
        fetch("/api/config", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(config)
        });
    });
});
</script>
</body>
</html>'''
