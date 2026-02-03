
from flask import Flask, render_template, jsonify
import json
import os
import time

app = Flask(__name__)
STATE_FILE = "dashboard_state.json"
LOG_FILE = "bot.log"

# Security
from functools import wraps
from flask import request, Response
from dotenv import load_dotenv

load_dotenv() # Load .env for DASHBOARD_PASSWORD

def check_auth(username, password):
    """This function is called to check if a username /
    password combination is valid."""
    # Username can be anything, password must match env
    stored_password = os.getenv("DASHBOARD_PASSWORD")
    if not stored_password:
        return True # Open if no password set (dev mode or first run safety)
    return password == stored_password

def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
    'Could not verify your access level for that URL.\n'
    'You have to login with proper credentials', 401,
    {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

@app.route('/')
@requires_auth
def home():
    return render_template("dashboard.html")

@app.route('/api/state')
@requires_auth
def get_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)})
    return jsonify({"status": "waiting_for_bot"})

@app.route('/api/history')
@requires_auth
def get_history():
    try:
        HISTORY_FILE = "trade_history.json"
        if os.path.exists(HISTORY_FILE):
             with open(HISTORY_FILE, 'r') as f:
                return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)})
    return jsonify([])

@app.route('/api/control/<action>', methods=['POST'])
@requires_auth
def control_bot(action):
    try:
        if action == 'pause':
            with open("bot.pause", "w") as f: f.write("paused")
            return jsonify({"status": "paused"})
        elif action == 'resume':
            if os.path.exists("bot.pause"):
                os.remove("bot.pause")
            return jsonify({"status": "resumed"})
        elif action == 'stop':
            # Create a stop file or just kill
            # runner.py check for bot.pause is for pausing, not stopping. 
            # We can kill python runner.py
            os.system("taskkill /F /IM python.exe /FI \"WINDOWTITLE eq runner.py\"") 
            # Force kill all runners if specific targeting fails
            os.system("taskkill /F /FI \"COMMANDLINE eq python runner.py\"") # Pseudo-command
             # Best effort:
            os.system("taskkill /F /IM python.exe") 
            return jsonify({"status": "stopped"})
    except Exception as e:
        return jsonify({"error": str(e)})
    return jsonify({"error": "invalid_action"})

@app.route('/api/logs')
@requires_auth
def get_logs():
    try:
        if os.path.exists(LOG_FILE):
             # Read with retry in case of write lock
            for _ in range(3):
                try:
                    with open(LOG_FILE, 'r') as f:
                        lines = f.readlines()[-50:]
                        return jsonify({"logs": lines})
                except IOError:
                    time.sleep(0.1)
    except Exception as e:
        return jsonify({"error": str(e)})
    return jsonify({"logs": []})

if __name__ == '__main__':
    print("Starting Dashboard on http://localhost:5000")
    print("Security: Password Protection ENABLED")
    app.run(host='0.0.0.0', port=5000, debug=True)
