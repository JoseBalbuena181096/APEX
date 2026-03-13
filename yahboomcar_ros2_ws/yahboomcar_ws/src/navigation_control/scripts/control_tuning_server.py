#!/usr/bin/env python3
"""
Control parameter tuning web UI for APEX Master v2.
Port 8082. 6 sliders + START/STOP.
Saves/loads from ~/.config/apex/control_params.json
"""

import rclpy
from rclpy.node import Node
import json
import os
import shutil
import subprocess
import threading
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

CONFIG_FILE = os.path.expanduser('~/.config/apex/control_params.json')
ros_node = None

HTML_PAGE = '''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>APEX Control Tuning</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', monospace;
       display: flex; flex-direction: column; align-items: center; min-height: 100vh; }
h1 { color: #58a6ff; font-size: 1.3em; padding: 18px 0 10px; }
.nav-panel { width: 100%; max-width: 600px; padding: 0 12px; margin-bottom: 14px; }
.nav-bar { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 14px; display: flex; align-items: center; gap: 16px;
           justify-content: center; }
.nav-status { font-size: 1.1em; font-weight: bold; padding: 0 20px; min-width: 140px;
              text-align: center; }
.nav-status.stopped { color: #f85149; }
.nav-status.running { color: #3fb950; }
.start-btn { padding: 12px 36px; background: #238636; color: #fff; border: 2px solid #2ea043;
             border-radius: 8px; font-size: 1.05em; cursor: pointer; font-weight: bold;
             min-width: 150px; }
.start-btn:hover { background: #2ea043; }
.stop-btn { padding: 12px 36px; background: #da3633; color: #fff; border: 2px solid #f85149;
            border-radius: 8px; font-size: 1.05em; cursor: pointer; font-weight: bold;
            min-width: 150px; }
.stop-btn:hover { background: #f85149; }
.panel { width: 100%; max-width: 600px; background: #161b22;
         border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin: 0 12px; }
.panel h2 { color: #58a6ff; font-size: 0.95em; margin-bottom: 14px;
            border-bottom: 1px solid #30363d; padding-bottom: 8px; }
.section-label { color: #f0883e; font-size: 0.8em; margin: 12px 0 4px; font-weight: bold; }
.param-group { margin-bottom: 14px; }
.param-group label { display: flex; justify-content: space-between;
                     font-size: 0.85em; margin-bottom: 4px; color: #8b949e; }
.param-group label span.val { color: #58a6ff; font-weight: bold; font-size: 1.1em; }
.slider-row { display: flex; align-items: center; gap: 8px; }
.slider-row input[type=range] { flex: 1; height: 6px; -webkit-appearance: none;
                    background: #30363d; border-radius: 3px; outline: none; }
.slider-row input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 16px;
    height: 16px; border-radius: 50%; background: #58a6ff; cursor: pointer; }
.slider-row input[type=number] { width: 80px; background: #0d1117; color: #58a6ff;
    border: 1px solid #30363d; border-radius: 4px; text-align: center;
    font-size: 0.85em; padding: 4px; }
.status { font-size: 0.85em; color: #3fb950; margin-top: 10px; min-height: 1.2em;
          text-align: center; font-weight: bold; }
.save-btn { width: 100%; margin-top: 14px; padding: 10px; background: #238636;
            color: #fff; border: 1px solid #2ea043; border-radius: 6px;
            font-size: 0.9em; cursor: pointer; font-weight: bold; }
.save-btn:hover { background: #2ea043; }
.save-btn.saved { background: #1f6feb; border-color: #388bfd; }
.hint { color: #484f58; font-size: 0.7em; margin-top: 2px; font-style: italic; }
</style></head>
<body>
<h1>APEX Control Tuning v2</h1>

<div class="nav-panel">
  <div class="nav-bar">
    <button class="start-btn" onclick="navControl('start')">START</button>
    <div class="nav-status stopped" id="navStatus">STOPPED</div>
    <button class="stop-btn" onclick="navControl('stop')">STOP</button>
  </div>
</div>

<div class="panel">
  <h2>Predictive Controller - 6 Parameters</h2>

  <div class="section-label">Steering Gains</div>

  <div class="param-group">
    <label>Kp CTE (lateral correction) <span class="val" id="v_kp_cte">0.50</span></label>
    <div class="slider-row">
      <input type="range" id="kp_cte" min="0.0" max="1.5" step="0.05" value="0.50"
             oninput="updateParam(this,'kp_cte')">
      <input type="number" id="n_kp_cte" min="0.0" max="1.5" step="0.01" value="0.50"
             onchange="syncParam('kp_cte',this.value)">
    </div>
    <div class="hint">No se centra? subir. Zigzaguea? bajar.</div>
  </div>

  <div class="param-group">
    <label>Kp Heading (angular alignment) <span class="val" id="v_kp_heading">0.30</span></label>
    <div class="slider-row">
      <input type="range" id="kp_heading" min="0.0" max="1.0" step="0.05" value="0.30"
             oninput="updateParam(this,'kp_heading')">
      <input type="number" id="n_kp_heading" min="0.0" max="1.0" step="0.01" value="0.30"
             onchange="syncParam('kp_heading',this.value)">
    </div>
    <div class="hint">Reemplaza Kd. Anti-zigzag natural.</div>
  </div>

  <div class="param-group">
    <label>Kff (curvature feedforward) <span class="val" id="v_kff">0.20</span></label>
    <div class="slider-row">
      <input type="range" id="kff" min="0.0" max="0.8" step="0.05" value="0.20"
             oninput="updateParam(this,'kff')">
      <input type="number" id="n_kff" min="0.0" max="0.8" step="0.01" value="0.20"
             onchange="syncParam('kff',this.value)">
    </div>
    <div class="hint">Entra tarde a curvas? subir. Gira antes de tiempo? bajar.</div>
  </div>

  <div class="section-label">Speed</div>

  <div class="param-group">
    <label>Max Speed (m/s) <span class="val" id="v_max_speed">0.20</span></label>
    <div class="slider-row">
      <input type="range" id="max_speed" min="0.05" max="0.5" step="0.01" value="0.20"
             oninput="updateParam(this,'max_speed')">
      <input type="number" id="n_max_speed" min="0.05" max="0.5" step="0.01" value="0.20"
             onchange="syncParam('max_speed',this.value)">
    </div>
  </div>

  <div class="param-group">
    <label>Kv Curve (speed reduction in curves) <span class="val" id="v_kv_curve">0.50</span></label>
    <div class="slider-row">
      <input type="range" id="kv_curve" min="0.0" max="1.0" step="0.05" value="0.50"
             oninput="updateParam(this,'kv_curve')">
      <input type="number" id="n_kv_curve" min="0.0" max="1.0" step="0.01" value="0.50"
             onchange="syncParam('kv_curve',this.value)">
    </div>
    <div class="hint">Se sale en curvas? subir. Casi se detiene? bajar.</div>
  </div>

  <div class="section-label">Limits</div>

  <div class="param-group">
    <label>Max Angular (steering clamp) <span class="val" id="v_max_angular">0.80</span></label>
    <div class="slider-row">
      <input type="range" id="max_angular" min="0.1" max="2.0" step="0.05" value="0.80"
             oninput="updateParam(this,'max_angular')">
      <input type="number" id="n_max_angular" min="0.1" max="2.0" step="0.05" value="0.80"
             onchange="syncParam('max_angular',this.value)">
    </div>
  </div>

  <div class="status" id="status"></div>
  <button class="save-btn" id="saveBtn" onclick="saveConfig()">Save Configuration</button>
</div>

<script>
let debounceTimers = {};

function navControl(action) {
  let st = document.getElementById('navStatus');
  fetch('/nav_control', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({action: action})
  })
  .then(r => r.json())
  .then(d => {
    if (d.ok) {
      let running = (action === 'start');
      st.textContent = running ? 'RUNNING' : 'STOPPED';
      st.className = 'nav-status ' + (running ? 'running' : 'stopped');
    } else { st.textContent = 'ERROR'; st.className = 'nav-status stopped'; }
  })
  .catch(() => { st.textContent = 'ERROR'; st.className = 'nav-status stopped'; });
}

function syncParam(name, value) {
  let slider = document.getElementById(name);
  let numInput = document.getElementById('n_' + name);
  let valSpan = document.getElementById('v_' + name);
  if (slider) slider.value = value;
  if (numInput) numInput.value = value;
  if (valSpan) valSpan.textContent = value;
  clearTimeout(debounceTimers[name]);
  debounceTimers[name] = setTimeout(() => {
    let val = parseFloat(value);
    fetch('/set_param', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: name, value: val})
    })
    .then(r => r.json())
    .then(d => {
      let st = document.getElementById('status');
      if (d.ok) {
        st.textContent = '\\u2713 ' + name + ' = ' + val; st.style.color = '#3fb950';
        fetch('/save_config', { method: 'POST' });
      } else {
        st.textContent = '\\u2717 ' + (d.message || 'error'); st.style.color = '#f85149';
      }
    }).catch(() => { document.getElementById('status').textContent = 'Connection error'; });
  }, 150);
}
function updateParam(el, name) { syncParam(name, el.value); }

function saveConfig() {
  let btn = document.getElementById('saveBtn');
  btn.className = 'save-btn'; btn.textContent = 'Saving...';
  fetch('/save_config', { method: 'POST' })
    .then(r => r.json())
    .then(d => {
      btn.className = d.ok ? 'save-btn saved' : 'save-btn error';
      btn.textContent = d.ok ? '\\u2713 Saved!' : '\\u2717 Failed';
      setTimeout(() => { btn.className = 'save-btn'; btn.textContent = 'Save Configuration'; }, 3000);
    });
}

fetch('/get_params').then(r => r.json()).then(params => {
  for (const [name, val] of Object.entries(params)) {
    let el = document.getElementById(name);
    if (el) { el.value = val; document.getElementById('v_' + name).textContent = val; }
    let n = document.getElementById('n_' + name);
    if (n) n.value = val;
  }
});
fetch('/get_nav_state').then(r => r.json()).then(d => {
  let st = document.getElementById('navStatus');
  st.textContent = d.enabled ? 'RUNNING' : 'STOPPED';
  st.className = 'nav-status ' + (d.enabled ? 'running' : 'stopped');
});
</script>
</body></html>'''


class TuningHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200); self.send_header('Content-Type', 'text/html'); self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        elif self.path == '/get_params':
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers()
            self.wfile.write(json.dumps(ros_node.get_current_params()).encode())
        elif self.path == '/get_nav_state':
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers()
            self.wfile.write(json.dumps({'enabled': ros_node.nav_enabled}).encode())
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length) if length else b'{}'
        try:
            data = json.loads(body) if body else {}
        except:
            data = {}

        if self.path == '/set_param':
            try:
                ok, msg = ros_node.set_remote_param(data['name'], data['value'])
                self._json_response({'ok': ok, 'message': msg})
            except Exception as e:
                self._json_response({'ok': False, 'error': str(e)}, 400)
        elif self.path == '/nav_control':
            try:
                ok, msg = ros_node.set_nav_enabled(data['action'] == 'start')
                self._json_response({'ok': ok, 'message': msg})
            except Exception as e:
                self._json_response({'ok': False, 'error': str(e)}, 400)
        elif self.path == '/save_config':
            try:
                params = ros_node.get_current_params()
                os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(params, f, indent=2)
                self._json_response({'ok': True, 'message': f'Saved to {CONFIG_FILE}'})
            except Exception as e:
                self._json_response({'ok': False, 'message': str(e)}, 500)
        else:
            self.send_response(404); self.end_headers()

    def _json_response(self, data, code=200):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json'); self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass


class ControlTuningNode(Node):
    PARAM_NAMES = ['kp_cte', 'kp_heading', 'kff', 'max_speed', 'kv_curve', 'max_angular']

    def __init__(self):
        super().__init__('control_tuning_server')
        self.nav_enabled = False
        self._param_cache = {
            'kp_cte': 0.50, 'kp_heading': 0.30, 'kff': 0.20,
            'max_speed': 0.20, 'kv_curve': 0.50, 'max_angular': 0.80,
        }
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE) as f:
                    saved = json.load(f)
                for k in self.PARAM_NAMES:
                    if k in saved:
                        self._param_cache[k] = saved[k]
                self.get_logger().info(f'Loaded config from {CONFIG_FILE}')
            except Exception as e:
                self.get_logger().error(f'Error loading config: {e}')

        self.create_timer(3.0, self._apply_saved_config_once)
        self._config_applied = False
        self.get_logger().info('Control tuning server v2 on port 8082')

    def _apply_saved_config_once(self):
        if self._config_applied: return
        self._config_applied = True
        for name, value in self._param_cache.items():
            ok, msg = self.set_remote_param(name, value)
            if ok: self.get_logger().info(f'  Restored {name} = {value}')

    def _ros2_param_set(self, node, name, val_str):
        ros2_bin = shutil.which('ros2') or '/opt/ros/humble/bin/ros2'
        env = os.environ.copy()
        if '/opt/ros/humble/bin' not in env.get('PATH', ''):
            env['PATH'] = '/opt/ros/humble/bin:' + env.get('PATH', '')
        r = subprocess.run([ros2_bin, 'param', 'set', node, name, val_str],
                          capture_output=True, text=True, timeout=5.0, env=env)
        return r.returncode == 0, (r.stdout.strip() if r.returncode == 0 else r.stderr.strip())

    def set_remote_param(self, name, value):
        if name not in set(self.PARAM_NAMES):
            return False, f'Unknown: {name}'
        try:
            ok, msg = self._ros2_param_set('/master_control', name, str(value))
            if ok: self._param_cache[name] = value
            return ok, msg
        except Exception as e:
            return False, str(e)

    def set_nav_enabled(self, enabled):
        try:
            ok, msg = self._ros2_param_set('/master_control', 'enabled',
                                           'true' if enabled else 'false')
            if ok: self.nav_enabled = enabled
            return ok, msg
        except Exception as e:
            return False, str(e)

    def get_current_params(self):
        return dict(self._param_cache)


def main(args=None):
    global ros_node
    rclpy.init(args=args)
    ros_node = ControlTuningNode()

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        allow_reuse_address = True
        daemon_threads = True
        def server_bind(self):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            super().server_bind()

    server = ThreadedHTTPServer(('0.0.0.0', 8082), TuningHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        rclpy.spin(ros_node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown(); ros_node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
