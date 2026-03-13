#!/usr/bin/env python3
"""
MJPEG web server + parameter tuning UI for lane detection.
Subscribes to /lane_detection/debug_image and serves MJPEG on port 8081.
Provides sliders that update ROS2 parameters on the line_detection node in real time.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import json
import os
import shutil
import subprocess
import threading
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

CONFIG_FILE = os.path.expanduser('~/.config/apex/lane_detection_params.json')

bridge = CvBridge()
latest_frame = None
frame_lock = threading.Lock()
ros_node = None  # Global ref so HTTP handler can access it

# ============================================================
# HTML + JS for the web UI with parameter sliders
# ============================================================
HTML_PAGE = '''<!DOCTYPE html>
<html><head><title>APEX Lane Detection</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', monospace;
       display: flex; flex-direction: column; align-items: center; min-height: 100vh; }
h1 { color: #58a6ff; font-size: 1.1em; padding: 10px 0 5px; }
.container { display: flex; flex-wrap: wrap; justify-content: center;
             gap: 12px; width: 100%; max-width: 1100px; padding: 0 10px; }
.video-panel { flex: 1; min-width: 300px; }
.video-panel img { width: 100%; border: 2px solid #30363d; border-radius: 6px; }
.controls { flex: 0 0 350px; background: #161b22; border: 1px solid #30363d;
            border-radius: 6px; padding: 12px; max-height: 90vh; overflow-y: auto; }
.controls h2 { color: #58a6ff; font-size: 0.9em; margin-bottom: 10px;
               border-bottom: 1px solid #30363d; padding-bottom: 6px; }
.param-group { margin-bottom: 8px; }
.param-group label { display: flex; justify-content: space-between;
                     font-size: 0.8em; margin-bottom: 2px; color: #8b949e; }
.param-group label span.val { color: #58a6ff; font-weight: bold; }
.slider-row { display: flex; align-items: center; gap: 6px; }
.slider-row input[type=range] { flex: 1; height: 6px; -webkit-appearance: none;
                    background: #30363d; border-radius: 3px; outline: none; }
.slider-row input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 14px;
    height: 14px; border-radius: 50%; background: #58a6ff; cursor: pointer; }
.slider-row input[type=number] { width: 58px; background: #0d1117; color: #58a6ff;
    border: 1px solid #30363d; border-radius: 3px; text-align: center;
    font-size: 0.8em; padding: 2px; }
.section-label { color: #f0883e; font-size: 0.8em; margin: 10px 0 4px;
                 font-weight: bold; }
.save-btn { width: 100%; margin-top: 12px; padding: 8px; background: #238636;
            color: #fff; border: 1px solid #2ea043; border-radius: 6px;
            font-size: 0.85em; cursor: pointer; font-weight: bold;
            transition: all 0.3s ease; }
.save-btn:hover { background: #2ea043; }
.save-btn.saving { background: #6e40c9; border-color: #8957e5; }
.save-btn.saved { background: #1f6feb; border-color: #388bfd; }
.save-btn.error { background: #da3633; border-color: #f85149; }
.status { font-size: 0.85em; color: #3fb950; margin-top: 8px; min-height: 1.4em;
          padding: 4px 8px; border-radius: 4px; text-align: center; font-weight: bold; }
.status.show { background: #161b22; border: 1px solid #30363d; }
</style></head>
<body>
<h1>APEX Lane Detection</h1>
<div class="container">
  <div class="video-panel">
    <img src="/stream" alt="Lane Detection Stream" />
  </div>
  <div class="controls">
    <h2>Parameter Tuning</h2>

    <div class="section-label">Threshold</div>
    <div class="param-group">
      <label>CLAHE Clip Limit <span class="val" id="v_clahe_clip">3.0</span></label>
      <div class="slider-row">
        <input type="range" id="clahe_clip" min="1.0" max="8.0" step="0.5" value="3.0"
               oninput="updateParam(this, 'clahe_clip', 'double')">
        <input type="number" id="n_clahe_clip" min="1.0" max="8.0" step="0.5" value="3.0"
               onchange="syncParam('clahe_clip', this.value, 'double')">
      </div>
    </div>
    <div class="param-group">
      <label>Adaptive Block Size <span class="val" id="v_adaptive_block">51</span></label>
      <div class="slider-row">
        <input type="range" id="adaptive_block" min="11" max="101" step="2" value="51"
               oninput="updateParam(this, 'adaptive_block', 'int')">
        <input type="number" id="n_adaptive_block" min="11" max="101" step="2" value="51"
               onchange="syncParam('adaptive_block', this.value, 'int')">
      </div>
    </div>
    <div class="param-group">
      <label>Adaptive C <span class="val" id="v_adaptive_c">-25</span></label>
      <div class="slider-row">
        <input type="range" id="adaptive_c" min="-50" max="-5" step="1" value="-25"
               oninput="updateParam(this, 'adaptive_c', 'int')">
        <input type="number" id="n_adaptive_c" min="-50" max="-5" step="1" value="-25"
               onchange="syncParam('adaptive_c', this.value, 'int')">
      </div>
    </div>

    <div class="section-label">Robustness</div>
    <div class="param-group">
      <label>Max Curvature Radius (px) <span class="val" id="v_max_curvature_radius">80.0</span></label>
      <div class="slider-row">
        <input type="range" id="max_curvature_radius" min="20" max="200" step="5" value="80"
               oninput="updateParam(this, 'max_curvature_radius', 'double')">
        <input type="number" id="n_max_curvature_radius" min="20" max="200" step="5" value="80"
               onchange="syncParam('max_curvature_radius', this.value, 'double')">
      </div>
    </div>
    <div class="param-group">
      <label>Color Filter (0=off, 1=on) <span class="val" id="v_use_color_filter">1</span></label>
      <div class="slider-row">
        <input type="range" id="use_color_filter" min="0" max="1" step="1" value="1"
               oninput="updateParam(this, 'use_color_filter', 'int')">
        <input type="number" id="n_use_color_filter" min="0" max="1" step="1" value="1"
               onchange="syncParam('use_color_filter', this.value, 'int')">
      </div>
    </div>
    <div class="param-group">
      <label>Sat Max White <span class="val" id="v_sat_max_white">60</span></label>
      <div class="slider-row">
        <input type="range" id="sat_max_white" min="20" max="150" step="5" value="60"
               oninput="updateParam(this, 'sat_max_white', 'int')">
        <input type="number" id="n_sat_max_white" min="20" max="150" step="1" value="60"
               onchange="syncParam('sat_max_white', this.value, 'int')">
      </div>
    </div>

    <div class="section-label">Regression</div>
    <div class="param-group">
      <label>RANSAC Min Points <span class="val" id="v_ransac_min_points">30</span></label>
      <div class="slider-row">
        <input type="range" id="ransac_min_points" min="5" max="100" step="5" value="30"
               oninput="updateParam(this, 'ransac_min_points', 'int')">
        <input type="number" id="n_ransac_min_points" min="5" max="100" step="1" value="30"
               onchange="syncParam('ransac_min_points', this.value, 'int')">
      </div>
    </div>
    <div class="param-group">
      <label>RANSAC Min Inliers <span class="val" id="v_ransac_min_inliers">8</span></label>
      <div class="slider-row">
        <input type="range" id="ransac_min_inliers" min="3" max="50" step="1" value="8"
               oninput="updateParam(this, 'ransac_min_inliers', 'int')">
        <input type="number" id="n_ransac_min_inliers" min="3" max="50" step="1" value="8"
               onchange="syncParam('ransac_min_inliers', this.value, 'int')">
      </div>
    </div>
    <div class="param-group">
      <label>Poly Search Min Pts <span class="val" id="v_poly_search_min">30</span></label>
      <div class="slider-row">
        <input type="range" id="poly_search_min" min="5" max="100" step="5" value="30"
               oninput="updateParam(this, 'poly_search_min', 'int')">
        <input type="number" id="n_poly_search_min" min="5" max="100" step="1" value="30"
               onchange="syncParam('poly_search_min', this.value, 'int')">
      </div>
    </div>

    <div class="section-label">Sliding Window</div>
    <div class="param-group">
      <label>Num Windows <span class="val" id="v_nwindows">9</span></label>
      <div class="slider-row">
        <input type="range" id="nwindows" min="4" max="20" step="1" value="9"
               oninput="updateParam(this, 'nwindows', 'int')">
        <input type="number" id="n_nwindows" min="4" max="20" step="1" value="9"
               onchange="syncParam('nwindows', this.value, 'int')">
      </div>
    </div>
    <div class="param-group">
      <label>Window Margin (px) <span class="val" id="v_sliding_margin">60</span></label>
      <div class="slider-row">
        <input type="range" id="sliding_margin" min="10" max="160" step="5" value="60"
               oninput="updateParam(this, 'sliding_margin', 'int')">
        <input type="number" id="n_sliding_margin" min="10" max="160" step="1" value="60"
               onchange="syncParam('sliding_margin', this.value, 'int')">
      </div>
    </div>
    <div class="param-group">
      <label>Min Pixels to Recenter <span class="val" id="v_sliding_minpix">15</span></label>
      <div class="slider-row">
        <input type="range" id="sliding_minpix" min="1" max="80" step="1" value="15"
               oninput="updateParam(this, 'sliding_minpix', 'int')">
        <input type="number" id="n_sliding_minpix" min="1" max="80" step="1" value="15"
               onchange="syncParam('sliding_minpix', this.value, 'int')">
      </div>
    </div>

    <div class="section-label">Smoothing</div>
    <div class="param-group">
      <label>EMA Alpha (0=smooth, 1=reactive) <span class="val" id="v_smooth_alpha">0.6</span></label>
      <div class="slider-row">
        <input type="range" id="smooth_alpha" min="0.1" max="1.0" step="0.05" value="0.6"
               oninput="updateParam(this, 'smooth_alpha', 'double')">
        <input type="number" id="n_smooth_alpha" min="0.1" max="1.0" step="0.05" value="0.6"
               onchange="syncParam('smooth_alpha', this.value, 'double')">
      </div>
    </div>

    <div class="section-label">Geometry (40cm = lane width)</div>
    <div class="param-group">
      <label>Lane Width (bird-eye px) <span class="val" id="v_lane_width_px">90</span></label>
      <div class="slider-row">
        <input type="range" id="lane_width_px" min="30" max="200" step="5" value="90"
               oninput="updateParam(this, 'lane_width_px', 'int')">
        <input type="number" id="n_lane_width_px" min="30" max="200" step="5" value="90"
               onchange="syncParam('lane_width_px', this.value, 'int')">
      </div>
    </div>

    <div class="section-label">Navigation (No-Lines)</div>
    <div class="param-group">
      <label>Recovery Blend Frames <span class="val" id="v_recovery_blend_frames">10</span></label>
      <div class="slider-row">
        <input type="range" id="recovery_blend_frames" min="3" max="30" step="1" value="10"
               oninput="updateParam(this, 'recovery_blend_frames', 'int')">
        <input type="number" id="n_recovery_blend_frames" min="3" max="30" step="1" value="10"
               onchange="syncParam('recovery_blend_frames', this.value, 'int')">
      </div>
    </div>
    <div class="param-group">
      <label>Max Inertial Frames <span class="val" id="v_max_inertial_frames">90</span></label>
      <div class="slider-row">
        <input type="range" id="max_inertial_frames" min="15" max="300" step="5" value="90"
               oninput="updateParam(this, 'max_inertial_frames', 'int')">
        <input type="number" id="n_max_inertial_frames" min="15" max="300" step="5" value="90"
               onchange="syncParam('max_inertial_frames', this.value, 'int')">
      </div>
    </div>
    <div class="param-group">
      <label>Inertial Straighten Rate <span class="val" id="v_inertial_straighten_rate">0.02</span></label>
      <div class="slider-row">
        <input type="range" id="inertial_straighten_rate" min="0.005" max="0.1" step="0.005" value="0.02"
               oninput="updateParam(this, 'inertial_straighten_rate', 'double')">
        <input type="number" id="n_inertial_straighten_rate" min="0.005" max="0.1" step="0.005" value="0.02"
               onchange="syncParam('inertial_straighten_rate', this.value, 'double')">
      </div>
    </div>

    <div class="section-label">Perspective Trapezoid</div>
    <div class="param-group">
      <label>Top-Left X (%) <span class="val" id="v_persp_top_left_pct">28</span></label>
      <div class="slider-row">
        <input type="range" id="persp_top_left_pct" min="0" max="50" step="1" value="28"
               oninput="updateParam(this, 'persp_top_left_pct', 'int')">
        <input type="number" id="n_persp_top_left_pct" min="0" max="50" step="1" value="28"
               onchange="syncParam('persp_top_left_pct', this.value, 'int')">
      </div>
    </div>
    <div class="param-group">
      <label>Top-Right X (%) <span class="val" id="v_persp_top_right_pct">71</span></label>
      <div class="slider-row">
        <input type="range" id="persp_top_right_pct" min="50" max="100" step="1" value="71"
               oninput="updateParam(this, 'persp_top_right_pct', 'int')">
        <input type="number" id="n_persp_top_right_pct" min="50" max="100" step="1" value="71"
               onchange="syncParam('persp_top_right_pct', this.value, 'int')">
      </div>
    </div>
    <div class="param-group">
      <label>Top Y (%) <span class="val" id="v_persp_top_y_pct">50</span></label>
      <div class="slider-row">
        <input type="range" id="persp_top_y_pct" min="20" max="80" step="1" value="50"
               oninput="updateParam(this, 'persp_top_y_pct', 'int')">
        <input type="number" id="n_persp_top_y_pct" min="20" max="80" step="1" value="50"
               onchange="syncParam('persp_top_y_pct', this.value, 'int')">
      </div>
    </div>
    <div class="param-group">
      <label>Bot-Left X (%) <span class="val" id="v_persp_bot_left_pct">5</span></label>
      <div class="slider-row">
        <input type="range" id="persp_bot_left_pct" min="0" max="40" step="1" value="5"
               oninput="updateParam(this, 'persp_bot_left_pct', 'int')">
        <input type="number" id="n_persp_bot_left_pct" min="0" max="40" step="1" value="5"
               onchange="syncParam('persp_bot_left_pct', this.value, 'int')">
      </div>
    </div>
    <div class="param-group">
      <label>Bot-Right X (%) <span class="val" id="v_persp_bot_right_pct">95</span></label>
      <div class="slider-row">
        <input type="range" id="persp_bot_right_pct" min="60" max="100" step="1" value="95"
               oninput="updateParam(this, 'persp_bot_right_pct', 'int')">
        <input type="number" id="n_persp_bot_right_pct" min="60" max="100" step="1" value="95"
               onchange="syncParam('persp_bot_right_pct', this.value, 'int')">
      </div>
    </div>
    <div class="param-group">
      <label>Bot Y (%) <span class="val" id="v_persp_bot_y_pct">100</span></label>
      <div class="slider-row">
        <input type="range" id="persp_bot_y_pct" min="50" max="100" step="1" value="100"
               oninput="updateParam(this, 'persp_bot_y_pct', 'int')">
        <input type="number" id="n_persp_bot_y_pct" min="50" max="100" step="1" value="100"
               onchange="syncParam('persp_bot_y_pct', this.value, 'int')">
      </div>
    </div>

    <div class="status" id="status"></div>
    <button class="save-btn" id="saveBtn" onclick="saveConfig()">Save Configuration</button>
  </div>
</div>

<script>
let debounceTimers = {};

function syncParam(name, value, type) {
  let slider = document.getElementById(name);
  let numInput = document.getElementById('n_' + name);
  let valSpan = document.getElementById('v_' + name);
  if (slider) slider.value = value;
  if (numInput) numInput.value = value;
  if (valSpan) valSpan.textContent = value;

  clearTimeout(debounceTimers[name]);
  debounceTimers[name] = setTimeout(() => {
    let val = type === 'double' ? parseFloat(value) : parseInt(value);
    fetch('/set_param', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: name, value: val, type: type})
    })
    .then(r => r.json())
    .then(d => {
      let st = document.getElementById('status');
      if (d.ok) {
        st.textContent = '\\u2713 ' + name + ' = ' + val;
        st.style.color = '#3fb950';
        // Auto-save after each successful change
        fetch('/save_config', { method: 'POST' });
      } else {
        st.textContent = '\\u2717 ' + (d.message || d.error || 'unknown error');
        st.style.color = '#f85149';
      }
    })
    .catch(e => {
      document.getElementById('status').textContent = 'Connection error';
    });
  }, 150);
}

function updateParam(el, name, type) {
  syncParam(name, el.value, type);
}

function saveConfig() {
  let st = document.getElementById('status');
  let btn = document.getElementById('saveBtn');
  btn.className = 'save-btn saving';
  btn.textContent = 'Saving...';
  st.textContent = '';
  st.className = 'status';
  fetch('/save_config', { method: 'POST' })
    .then(r => r.json())
    .then(d => {
      if (d.ok) {
        btn.className = 'save-btn saved';
        btn.textContent = '\\u2713 Saved!';
        st.textContent = 'Configuration saved successfully';
        st.style.color = '#3fb950';
        st.className = 'status show';
      } else {
        btn.className = 'save-btn error';
        btn.textContent = '\\u2717 Failed';
        st.textContent = d.message || 'Unknown error';
        st.style.color = '#f85149';
        st.className = 'status show';
      }
      setTimeout(() => {
        btn.className = 'save-btn';
        btn.textContent = 'Save Configuration';
      }, 3000);
    })
    .catch(e => {
      btn.className = 'save-btn error';
      btn.textContent = '\\u2717 Error';
      st.textContent = 'Connection error';
      st.style.color = '#f85149';
      st.className = 'status show';
      setTimeout(() => {
        btn.className = 'save-btn';
        btn.textContent = 'Save Configuration';
      }, 3000);
    });
}

// On load, fetch current parameter values from the node
fetch('/get_params')
  .then(r => r.json())
  .then(params => {
    for (const [name, val] of Object.entries(params)) {
      let el = document.getElementById(name);
      if (el) {
        el.value = val;
        document.getElementById('v_' + name).textContent = val;
      }
      let numEl = document.getElementById('n_' + name);
      if (numEl) numEl.value = val;
    }
  });
</script>
</body></html>'''


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())

        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        frame = latest_frame
                    if frame is None:
                        import time; time.sleep(0.03)
                        continue
                    _, jpeg = cv2.imencode('.jpg', frame,
                                           [cv2.IMWRITE_JPEG_QUALITY, 70])
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b'\r\n')
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif self.path == '/get_params':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            params = ros_node.get_current_params()
            self.wfile.write(json.dumps(params).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/set_param':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                name = data['name']
                value = data['value']
                ptype = data.get('type', 'int')
                ok, msg = ros_node.set_remote_param(name, value, ptype)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': ok, 'message': msg}).encode())
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': str(e)}).encode())
        elif self.path == '/save_config':
            try:
                params = ros_node.get_current_params()
                os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(params, f, indent=2)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': True, 'message': f'Saved to {CONFIG_FILE}'}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'message': str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


class WebVideoNode(Node):
    def __init__(self):
        super().__init__('web_video_server')
        self.subscription = self.create_subscription(
            Image, '/lane_detection/debug_image', self.image_cb, 5)
        self.get_logger().info('Web video server listening on port 8081')
        # In-memory cache of param values
        self._param_cache = {
            'clahe_clip': 3.0, 'adaptive_block': 51, 'adaptive_c': -25,
            'sliding_margin': 60, 'sliding_minpix': 15,
            'ransac_min_points': 30, 'ransac_min_inliers': 8,
            'poly_search_min': 30, 'nwindows': 9, 'smooth_alpha': 0.6,
            'lane_width_px': 90,
            'persp_top_left_pct': 28, 'persp_top_right_pct': 71,
            'persp_top_y_pct': 50,
            'persp_bot_left_pct': 5, 'persp_bot_right_pct': 95,
            'persp_bot_y_pct': 100,
            # New params
            'max_curvature_radius': 80.0,
            'use_color_filter': 1,
            'sat_max_white': 60,
            'recovery_blend_frames': 10,
            'max_inertial_frames': 90,
            'inertial_straighten_rate': 0.02,
        }
        # Load saved config after 3 seconds
        self.create_timer(3.0, self._load_saved_config_once)
        self._config_loaded = False

    def _load_saved_config_once(self):
        if self._config_loaded:
            return
        self._config_loaded = True
        if not os.path.exists(CONFIG_FILE):
            self.get_logger().info(f'No saved config at {CONFIG_FILE}')
            return
        try:
            with open(CONFIG_FILE) as f:
                saved = json.load(f)
            self.get_logger().info(f'Loading saved config from {CONFIG_FILE}')
            type_map = {
                'clahe_clip': 'double', 'smooth_alpha': 'double',
                'max_curvature_radius': 'double',
                'inertial_straighten_rate': 'double',
            }
            for name, value in saved.items():
                ptype = type_map.get(name, 'int')
                ok, msg = self.set_remote_param(name, value, ptype)
                if ok:
                    self.get_logger().info(f'  Restored {name} = {value}')
                else:
                    self.get_logger().warn(f'  Failed to restore {name}: {msg}')
        except Exception as e:
            self.get_logger().error(f'Error loading saved config: {e}')

    def image_cb(self, msg):
        global latest_frame
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, 'bgr8')
            with frame_lock:
                latest_frame = cv_image
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')

    def set_remote_param(self, name, value, ptype='int'):
        """Set a parameter on the line_detection node via ros2 CLI."""
        allowed = {
            'clahe_clip', 'adaptive_block', 'adaptive_c', 'sliding_margin',
            'sliding_minpix', 'ransac_min_points', 'ransac_min_inliers',
            'poly_search_min', 'nwindows', 'smooth_alpha',
            'lane_width_px', 'persp_top_left_pct', 'persp_top_right_pct',
            'persp_top_y_pct', 'persp_bot_left_pct', 'persp_bot_right_pct',
            'persp_bot_y_pct',
            # New params
            'max_curvature_radius', 'use_color_filter', 'sat_max_white',
            'recovery_blend_frames', 'max_inertial_frames',
            'inertial_straighten_rate',
        }
        if name not in allowed:
            return False, f'Unknown param: {name}'
        try:
            val_str = str(value)
            ros2_bin = shutil.which('ros2') or '/opt/ros/humble/bin/ros2'
            env = os.environ.copy()
            if '/opt/ros/humble/bin' not in env.get('PATH', ''):
                env['PATH'] = '/opt/ros/humble/bin:' + env.get('PATH', '')
            result = subprocess.run(
                [ros2_bin, 'param', 'set', '/line_detection', name, val_str],
                capture_output=True, text=True, timeout=5.0, env=env)
            ok = result.returncode == 0
            msg = result.stdout.strip() if ok else result.stderr.strip()
            if ok:
                self.get_logger().info(f'param OK: {name} = {val_str}')
                self._param_cache[name] = value
            else:
                self.get_logger().warn(f'param FAIL: {name} -> {msg}')
            return ok, msg
        except subprocess.TimeoutExpired:
            return False, 'Command timed out'
        except Exception as e:
            self.get_logger().error(f'set_remote_param error: {e}')
            return False, str(e)

    def get_current_params(self):
        """Return in-memory param cache."""
        return dict(self._param_cache)


def main(args=None):
    global ros_node
    rclpy.init(args=args)
    ros_node = WebVideoNode()

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        allow_reuse_address = True
        daemon_threads = True
        def server_bind(self):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            super().server_bind()

    server = ThreadedHTTPServer(('0.0.0.0', 8081), MJPEGHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        rclpy.spin(ros_node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
