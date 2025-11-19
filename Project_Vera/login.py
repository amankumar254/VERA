import hashlib
import sys
import sqlite3 # Import the SQLite library

# --- Configuration ---
# This will create a file named 'user_data.db' to store the user table.
DB_NAME = 'user_data.db'
# --- End Configuration ---

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    # Allows accessing columns by name: row['username']
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database and creates the user table if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    # Create the 'users' table with a unique username constraint
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print(f"Database '{DB_NAME}' initialized and table 'users' ensured.")

def hash_password(password):
    """Hashes the password using SHA-256."""
    # We encode the string to bytes, as hashlib works with bytes
    # NOTE: For production security, use dedicated libraries like 'bcrypt' or 'argon2'.
    return hashlib.sha256(password.encode()).hexdigest()

def sign_up():
    """Allows a new user to register and stores credentials in the SQL database."""
    print("\n--- Sign Up ---")

    conn = get_db_connection()
    cursor = conn.cursor()

    while True:
        username = input("Enter new username: ").strip()
        if not username:
            print("Username cannot be empty.")
            continue

        # Check if username already exists in the database
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            print("Username already exists. Please choose another one.")
            continue
        break

    while True:
        password = input("Enter new password (must be at least 6 characters): ")
        if len(password) < 6:
            print("Password is too short. Please try again.")
            continue
        break

    # Hash the password
    hashed_password = hash_password(password)

    # Insert the new user into the database
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, hashed_password)
        )
        conn.commit()
        print(f"\n✅ Success! User '{username}' has been signed up and saved to the database.")
    except sqlite3.IntegrityError:
        print("\n❌ Error: Could not save user (Integrity constraint violation).")
    finally:
        conn.close()


def login():
    """Allows an existing user to log in by verifying credentials against the SQL database."""
    print("\n--- Log In ---")
    username = input("Enter username: ").strip()
    password = input("Enter password: ")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Retrieve the user record by username
    cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    user_record = cursor.fetchone()
    conn.close()

    if user_record is None:
        print("\n❌ Error: User not found.")
        return

    stored_hashed_password = user_record['password_hash']
    input_hashed_password = hash_password(password) # Hash the input password for comparison

    if input_hashed_password == stored_hashed_password:
        print(f"\n🎉 Welcome back, {username}! Login successful.")
    else:
        print("\n❌ Error: Incorrect password.")

def main_menu():
    """The main application loop and menu."""

    # 1. Initialize the database connection and tables
    init_db()

    while True:
        print("\n" + "="*30)
        print("PYTHON AUTHENTICATION SYSTEM (SQLite)")
        print("="*30)
        print("1. Sign Up (Register)")
        print("2. Log In")
        print("3. Exit")
        print("-"*30)

        choice = input("Enter your choice (1-3): ").strip()

        if choice == '1':
            sign_up()
        elif choice == '2':
            login()
        elif choice == '3':
            print("Goodbye! The system is now exiting.")
            sys.exit(0)
        else:
            print("\n🚨 Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting.")
        sys.exit(0)