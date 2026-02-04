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

load_dotenv()  # Load .env for DASHBOARD_PASSWORD


def check_auth(username, password):
    """Username can be anything; password must match DASHBOARD_PASSWORD."""
    stored_password = os.getenv("DASHBOARD_PASSWORD")
    if not stored_password:
        # If no password is set, dashboard is open (not recommended for LAN access)
        return True
    return password == stored_password


def authenticate():
    return Response(
        'Could not verify your access level for that URL.\n'
        'You have to login with proper credentials',
        401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )


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
        history_file = "trade_history.json"
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)})
    return jsonify([])


@app.route('/api/logs')
@requires_auth
def get_logs():
    try:
        if os.path.exists(LOG_FILE):
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
    port = int(os.getenv('DASHBOARD_PORT', '5000'))
    host = os.getenv('DASHBOARD_HOST', '0.0.0.0')
    print(f"Starting Dashboard on http://{host}:{port}")
    if os.getenv('DASHBOARD_PASSWORD'):
        print("Security: Password Protection ENABLED")
    else:
        print("Security: NO PASSWORD SET (set DASHBOARD_PASSWORD in .env)")

    # IMPORTANT: debug/reloader off for 24/7 scheduled-task runs
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
