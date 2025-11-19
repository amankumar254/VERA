from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import sqlite3
import bcrypt
import os
from datetime import timedelta

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
app.secret_key = os.environ.get('FLASK_SECRET', 'super_secret_key')
app.permanent_session_lifetime = timedelta(days=7)

# Database setup
DB_PATH = 'users.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def create_table():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fullname TEXT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

create_table()

@app.context_processor
def inject_user_and_date():
    from datetime import datetime
    user = session.get('user_email')
    return {
        'current_date': datetime.now().strftime('%B %d, %Y'),
        'user_email': user,
        'current_category': 'Home'
    }

# Render homepage (index.html must be inside templates/)
@app.route('/')
def index():
    return render_template('index.html', categories=['Home','Business','Sports','Science','Entertainment'])

# ✅ Register API
@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json() or {}
    fullname = data.get('fullname', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not email or not password:
        return jsonify({'success': False, 'message': 'Email and password required.'}), 400
    if len(password) < 6:
        return jsonify({'success': False, 'message': 'Password must be at least 6 characters.'}), 400

    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO users (fullname, email, password) VALUES (?, ?, ?)", (fullname, email, hashed))
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'message': 'Email already registered.'}), 409

    return jsonify({'success': True, 'message': 'Registration successful. Please log in.'}), 201

# ✅ Login API
@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json() or {}
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not email or not password:
        return jsonify({'success': False, 'message': 'Email and password required.'}), 400

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return jsonify({'success': False, 'message': 'Invalid email or password.'}), 401

    if bcrypt.checkpw(password.encode('utf-8'), row['password']):
        session.permanent = True
        session['user_email'] = email
        session['user_fullname'] = row['fullname'] or email
        return jsonify({'success': True, 'message': 'Login successful.'}), 200
    else:
        return jsonify({'success': False, 'message': 'Invalid email or password.'}), 401

# ✅ Logout
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
