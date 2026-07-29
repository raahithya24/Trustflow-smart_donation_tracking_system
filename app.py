# ============================================================
#  TrustFlow - Smart Donation Platform
#  app.py  (complete, updated)
#
#  Changes from original:
#   1. Credentials moved to .env  (python-dotenv)
#   2. trust_badge column handled safely
#   3. /ranked_campaigns fixed  (includes pending + approved)
#   4. /fund_alerts fixed        (includes pending + approved)
#   5. Fraud detection added inside /donate
#        - Duplicate transaction check  (same amount, same campaign, 2 min)
#        - Velocity check               (5+ donations in 10 min)
#        - Unusual amount check         (3x donor average)
#   6. Admin API routes added
#        - GET  /admin/stats
#        - GET  /admin/users
#        - GET  /admin/campaigns
#        - POST /admin/campaign/<id>/status   (approve / reject)
#        - GET  /admin/fraud_logs
#        - POST /admin/fraud_logs/<id>/review
#        - POST /admin/verify_ngo/<ngo_id>
#   7. Donations table now accepts payment_method & transaction_id
#   8. Audit log helper (write_audit_log) used throughout
#   9. Missing NGO routes added
#        - GET  /get_campaign/<id>            (for edit modal)
#        - PUT  /update_campaign/<id>         (edit title/desc/target)
#        - DELETE /delete_campaign/<id>       (soft-delete if has donations)
#        - POST /add_beneficiary              (add beneficiary story)
#        - GET  /get_beneficiaries/<id>       (list beneficiaries)
#        - POST /add_fund_allocation          (record fund allocation)
#        - GET  /get_fund_allocations/<id>    (list allocations)
#        - GET  /donor_statement/<id>/<year>  (year-end PDF statement)
# ============================================================

import io
import os
from datetime import datetime
from functools import wraps

from dotenv import load_dotenv
import threading
from datetime import datetime as dt

# AI/ML imports - graceful fallback if packages not installed
try:
    import numpy as np
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    import warnings
    warnings.warn('numpy/scikit-learn not installed. Run: pip install numpy scikit-learn joblib')
from flask import (Flask, jsonify, redirect, render_template,
                   request, send_file, session, url_for)
from flask_mysqldb import MySQL
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from werkzeug.security import check_password_hash, generate_password_hash

# ── Load environment variables ─────────────────────────────
load_dotenv()

app = Flask(__name__)
APP_NAME = "TrustFlow"

app.secret_key = os.getenv("SECRET_KEY", "trustflow_fallback_secret")

# ── MySQL Configuration ────────────────────────────────────
app.config['MYSQL_HOST']     = os.getenv("MYSQL_HOST", "localhost")
app.config['MYSQL_USER']     = os.getenv("MYSQL_USER", "root")
app.config['MYSQL_PASSWORD'] = os.getenv("MYSQL_PASSWORD", "")
app.config['MYSQL_DB']       = os.getenv("MYSQL_DB", "smart_donation")

mysql = MySQL(app)


# ============================================================
#  AI ENGINE — Fraud Detection + Campaign Intelligence
# ============================================================

class TrustFlowAI:
    """
    Wraps all ML models used across TrustFlow.

    Models:
      - IsolationForest  : unsupervised anomaly detection on donor behaviour
      - RandomForest     : supervised fraud classifier (retrains as labels accumulate)
      - Priority scoring : gradient-based campaign urgency ranking
      - Recommendation   : category-affinity scoring per donor
    """

    def __init__(self):
        self.iso_model      = None   # IsolationForest
        self.rf_model       = None   # RandomForestClassifier
        self.scaler         = StandardScaler() if ML_AVAILABLE else None
        self.is_trained     = False
        self.rf_trained     = False
        self._lock          = threading.Lock()

        # Synthetic seed data — realistic Indian donation patterns.
        # The model retrains automatically once real data is available.
        self._seed_X = np.array([
            # [amount, donations_last_24h, ratio, acct_age_days, total_don, is_new, amount_24h_total]
            [500,   1, 1.0, 120, 10, 0,   500],
            [1000,  1, 1.2, 200, 20, 0,  1000],
            [200,   2, 0.8,  90,  8, 0,   400],
            [2000,  1, 1.5, 365, 35, 0,  2000],
            [500,   1, 1.0,  45,  3, 0,   500],
            [750,   2, 1.1, 180, 14, 0,  1500],
            [100,   1, 0.5, 730, 80, 0,   100],
            [5000,  1, 2.0, 500, 45, 0,  5000],
            [300,   3, 0.9,  60,  5, 0,   900],
            [1500,  1, 1.3, 150, 12, 0,  1500],
            # --- fraud patterns ---
            [50000, 18, 40.0,  1, 1, 1,  900000],
            [25000, 12, 35.0,  2, 2, 1,  300000],
            [10000,  9, 20.0,  1, 1, 1,   90000],
            [8000,  15, 18.0,  3, 3, 1,  120000],
            [500,   20,  1.0,  1, 1, 1,   10000]
        ])
        self._seed_y = [0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1]

        # Train immediately on seed data (only if ML packages available)
        if ML_AVAILABLE:
            self._train_isolation_forest(self._seed_X)
            self._train_random_forest(self._seed_X, self._seed_y)

    # ── Feature engineering ────────────────────────────────
    def build_features(self, amount, donor_id, cur):
        """
        Builds a 6-dimensional feature vector for a transaction.
        All features are computable from existing MySQL tables.
        """
        features = {}

        # 1. Raw amount
        features['amount'] = amount

        # 2. Rolling 24-hour donation count (excludes fraud records)
        #    FIX: was CURDATE() which resets at midnight.
        cur.execute("""
            SELECT COUNT(*) FROM donations
            WHERE donor_id = %s
              AND donation_date >= NOW() - INTERVAL 24 HOUR
              AND status != 'fraud'
        """, (donor_id,))
        features['donations_today'] = int(cur.fetchone()[0])

        # 3. Amount ratio with cold-start fix + exclude fraud donations
        #    FIX 1: Only meaningful with >= 3 prior donations.
        #    FIX 2: Exclude fraud donations from the average.
        cur.execute("""
            SELECT AVG(amount), COUNT(*) FROM donations
            WHERE donor_id = %s AND status != 'fraud'
        """, (donor_id,))
        row = cur.fetchone()
        avg_amount      = float(row[0]) if row[0] else 0
        total_donations = int(row[1])
        if total_donations >= 3 and avg_amount > 0:
            features['amount_ratio'] = amount / avg_amount
        else:
            features['amount_ratio'] = 1.0  # neutral: not enough history
        features['total_donations'] = total_donations

        # 4. Account age in days
        cur.execute("SELECT DATEDIFF(NOW(), created_at) FROM users WHERE id=%s", (donor_id,))
        age_row = cur.fetchone()
        features['account_age_days'] = int(age_row[0]) if age_row and age_row[0] else 0

        # 5. Is this a new account? (registered within 7 days)
        features['is_new_account'] = 1 if features['account_age_days'] <= 7 else 0

        # 6. Rolling 24h total amount (excludes fraud)
        cur.execute("""
            SELECT COALESCE(SUM(amount), 0) FROM donations
            WHERE donor_id = %s
              AND donation_date >= NOW() - INTERVAL 24 HOUR
              AND status != 'fraud'
        """, (donor_id,))
        features['amount_24h_total'] = float(cur.fetchone()[0])

        vec = np.array([[
            features['amount'],
            features['donations_today'],
            features['amount_ratio'],
            features['account_age_days'],
            features['total_donations'],
            features['is_new_account'],
            features['amount_24h_total'],
        ]])
        return vec, features

    # ── Isolation Forest (unsupervised anomaly) ────────────
    def _train_isolation_forest(self, X):
        with self._lock:
            try:
                iso = IsolationForest(
                    n_estimators=100,
                    contamination=0.15,   # expect ~15% anomalies
                    random_state=42,
                    n_jobs=-1
                )
                iso.fit(X)
                self.iso_model   = iso
                self.is_trained  = True
            except Exception as e:
                print(f"[AI] IsolationForest training failed: {e}")

    def anomaly_score(self, feature_vec):
        """
        Returns a 0-100 risk score.
        Isolation Forest raw score is in [-0.5, 0.5].
        We map it to [0, 100] where 100 = most anomalous.
        """
        if not self.is_trained:
            return 0.0
        raw = self.iso_model.decision_function(feature_vec)[0]
        # raw: positive = normal, negative = anomalous
        # map to 0-100 where higher = more suspicious
        score = max(0.0, min(100.0, (-raw + 0.5) * 100))
        return round(score, 1)

    # ── Random Forest (supervised classifier) ─────────────
    def _train_random_forest(self, X, y):
        with self._lock:
            try:
                if len(set(y)) < 2:
                    return   # need both classes to train
                rf = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf',    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=6,
                        random_state=42,
                        class_weight='balanced',
                        n_jobs=-1
                    ))
                ])
                rf.fit(X, y)
                self.rf_model    = rf
                self.rf_trained  = True
            except Exception as e:
                print(f"[AI] RandomForest training failed: {e}")

    def fraud_probability(self, feature_vec):
        """Returns 0.0-1.0 fraud probability from RandomForest."""
        if not self.rf_trained:
            return 0.0
        proba = self.rf_model.predict_proba(feature_vec)[0]
        return round(float(proba[1]), 3)   # probability of class 1 (fraud)

    def retrain(self, cur):
        """
        Retrain both models using all real labelled donations from the DB.
        Called in a background thread after each donation.
        Fraud label = 1 if the donor has any reviewed fraud_log entries.
        """
        try:
            cur.execute("""
                SELECT d.amount,
                       COUNT(d2.id)  AS donations_that_day,
                       d.amount / NULLIF(AVG(d3.amount),0),
                       DATEDIFF(d.donation_date, u.created_at),
                       (SELECT COUNT(*) FROM donations WHERE donor_id=d.donor_id AND donation_date < d.donation_date),
                       IF(DATEDIFF(d.donation_date, u.created_at) <= 7, 1, 0),
                       IF(fl.donor_id IS NOT NULL, 1, 0) AS is_fraud
                FROM donations d
                JOIN users u ON d.donor_id = u.id
                LEFT JOIN donations d2 ON d2.donor_id=d.donor_id
                    AND DATE(d2.donation_date)=DATE(d.donation_date)
                LEFT JOIN donations d3 ON d3.donor_id=d.donor_id
                    AND d3.donation_date < d.donation_date
                LEFT JOIN fraud_logs fl ON fl.donor_id=d.donor_id AND fl.reviewed=1
                GROUP BY d.id
                LIMIT 5000
            """)
            rows = cur.fetchall()
            if len(rows) < 10:
                return   # not enough real data yet — keep using seed

            X = np.array([[
                float(r[0] or 0),
                int(r[1] or 0),
                float(r[2] or 1.0),
                int(r[3] or 0),
                int(r[4] or 0),
                int(r[5] or 0)
            ] for r in rows])
            y = [int(r[6] or 0) for r in rows]

            # Combine with seed to avoid catastrophic forgetting on small datasets
            X_combined = np.vstack([self._seed_X, X])
            y_combined  = self._seed_y + y

            self._train_isolation_forest(X_combined)
            self._train_random_forest(X_combined, y_combined)
            print(f"[AI] Retrained on {len(rows)} real donations")
        except Exception as e:
            print(f"[AI] Retrain failed: {e}")

    # ── Priority Score ─────────────────────────────────────
    @staticmethod
    def compute_priority_score(urgency, severity, completion_rate, deadline):
        """
        Campaign priority score 0-100.
        Higher = needs attention more urgently.

        Formula:
          need_factor    = urgency (1-10) × severity (1-10)         → 0-100
          gap_factor     = 1 - completion_rate/100                  → 0-1
          deadline_factor= exponential urgency as deadline approaches → 1.0-2.0
          score          = need × gap × deadline, normalised to 0-100
        """
        try:
            need_factor = (urgency * severity) / 100.0   # normalise to 0-1

            gap_factor  = max(0.0, 1.0 - (completion_rate / 100.0))

            if deadline:
                today     = dt.today().date()
                if isinstance(deadline, str):
                    deadline = dt.strptime(deadline, '%Y-%m-%d').date()
                days_left = (deadline - today).days
                if   days_left <= 0:   deadline_factor = 2.0
                elif days_left <= 7:   deadline_factor = 1.8
                elif days_left <= 14:  deadline_factor = 1.5
                elif days_left <= 30:  deadline_factor = 1.2
                elif days_left <= 60:  deadline_factor = 1.05
                else:                  deadline_factor = 1.0
            else:
                deadline_factor = 1.0

            raw   = need_factor * gap_factor * deadline_factor * 100
            score = round(min(100.0, max(0.0, raw)), 2)
            return score
        except Exception:
            return 0.0

    # ── Donor Recommendation ───────────────────────────────
    def recommend_campaigns(self, donor_id, all_campaigns, cur, top_n=6):
        """
        Category-affinity collaborative filtering.

        1. Count how many times this donor donated per category.
        2. Score each campaign by (category_affinity × urgency × gap).
        3. Return top_n campaigns sorted by score.

        Falls back to priority-score ranking for new donors.
        """
        try:
            cur.execute("""
                SELECT c.category, COUNT(*) as cnt
                FROM donations d
                JOIN campaigns c ON d.campaign_id = c.id
                WHERE d.donor_id = %s
                GROUP BY c.category
                ORDER BY cnt DESC
            """, (donor_id,))
            history = {r[0]: int(r[1]) for r in cur.fetchall()}

            if not history:
                # New donor — return by priority score
                return sorted(all_campaigns,
                               key=lambda c: c.get('priority_score', 0),
                               reverse=True)[:top_n]

            total_donations = sum(history.values())

            def affinity_score(campaign):
                cat      = campaign.get('category', '')
                affinity = history.get(cat, 0) / total_donations   # 0.0-1.0
                urgency  = campaign.get('urgency_level', 5) / 10.0
                gap      = 1 - min(campaign.get('completion_rate', 0), 100) / 100
                return affinity * 0.5 + urgency * 0.3 + gap * 0.2

            scored = [(c, affinity_score(c)) for c in all_campaigns]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [c for c, _ in scored[:top_n]]
        except Exception:
            return all_campaigns[:top_n]


# Instantiate once at module level — shared across all requests
ai = TrustFlowAI()


# ============================================================
#  DATABASE AUTO-MIGRATION
#  Runs once at startup — creates any missing tables/columns.
#  Safe on existing databases (uses IF NOT EXISTS).
# ============================================================

def init_db():
    """Create all tables and add missing columns on startup."""
    DDL_STATEMENTS = [
        # ── Core tables ──────────────────────────────────
        """CREATE TABLE IF NOT EXISTS users (
            id         INT AUTO_INCREMENT PRIMARY KEY,
            name       VARCHAR(150),
            email      VARCHAR(200) UNIQUE NOT NULL,
            phone      VARCHAR(20),
            password   VARCHAR(512) NOT NULL,
            role       ENUM('donor','ngo','admin') NOT NULL DEFAULT 'donor',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS ngos (
            id             INT AUTO_INCREMENT PRIMARY KEY,
            user_id        INT UNIQUE NOT NULL,
            document_score DECIMAL(5,2) DEFAULT 0,
            update_score   DECIMAL(5,2) DEFAULT 0,
            is_verified    TINYINT(1)   DEFAULT 0,
            trust_badge    VARCHAR(20)  DEFAULT NULL,
            created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )""",
        """CREATE TABLE IF NOT EXISTS campaigns (
            id               INT AUTO_INCREMENT PRIMARY KEY,
            ngo_id           INT NOT NULL,
            title            VARCHAR(300),
            description      TEXT,
            category         VARCHAR(100) DEFAULT 'General',
            target_amount    DECIMAL(12,2) DEFAULT 0,
            collected_amount DECIMAL(12,2) DEFAULT 0,
            total_donors     INT           DEFAULT 0,
            urgency_level    INT           DEFAULT 5,
            severity_score   INT           DEFAULT 5,
            deadline         DATE,
            status           ENUM('pending','approved','rejected','completed') DEFAULT 'pending',
            completion_rate  DECIMAL(6,2)  DEFAULT 0,
            priority_score   DECIMAL(8,2)  DEFAULT 0,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ngo_id) REFERENCES ngos(id) ON DELETE CASCADE
        )""",
        """CREATE TABLE IF NOT EXISTS donations (
            id             INT AUTO_INCREMENT PRIMARY KEY,
            donor_id       INT NOT NULL,
            campaign_id    INT NOT NULL,
            amount         DECIMAL(12,2) NOT NULL,
            payment_method VARCHAR(50)  DEFAULT 'online',
            transaction_id VARCHAR(100),
            status         ENUM('success','failed','pending') DEFAULT 'success',
            donation_date  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (donor_id)    REFERENCES users(id)     ON DELETE CASCADE,
            FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
        )""",
        # ── campaign_expenses (was missing) ───────────────
        """CREATE TABLE IF NOT EXISTS campaign_expenses (
            id           INT AUTO_INCREMENT PRIMARY KEY,
            campaign_id  INT NOT NULL,
            title        VARCHAR(300) NOT NULL,
            amount       DECIMAL(12,2) NOT NULL,
            description  TEXT,
            expense_date DATE NOT NULL,
            bill_image   VARCHAR(500),
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
        )""",
        """CREATE TABLE IF NOT EXISTS campaign_updates (
            id             INT AUTO_INCREMENT PRIMARY KEY,
            campaign_id    INT NOT NULL,
            update_text    TEXT,
            proof_document VARCHAR(500),
            created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
        )""",
        """CREATE TABLE IF NOT EXISTS beneficiaries (
            id           INT AUTO_INCREMENT PRIMARY KEY,
            campaign_id  INT NOT NULL,
            name         VARCHAR(200),
            story        TEXT,
            age          INT,
            location     VARCHAR(200),
            image        VARCHAR(500),
            helped_date  DATE,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
        )""",
        """CREATE TABLE IF NOT EXISTS fund_allocation (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            campaign_id     INT NOT NULL,
            category        VARCHAR(200),
            amount          DECIMAL(12,2) DEFAULT 0,
            description     TEXT,
            allocation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
        )""",
        """CREATE TABLE IF NOT EXISTS notifications (
            id         INT AUTO_INCREMENT PRIMARY KEY,
            user_id    INT NOT NULL,
            message    TEXT,
            type       VARCHAR(100),
            is_read    TINYINT(1) DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )""",
        """CREATE TABLE IF NOT EXISTS fraud_logs (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            campaign_id INT,
            donor_id    INT,
            reason      TEXT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reviewed    TINYINT(1) DEFAULT 0,
            admin_note  TEXT,
            reviewed_at TIMESTAMP  NULL,
            reviewed_by INT        DEFAULT NULL,
            FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE SET NULL,
            FOREIGN KEY (donor_id)    REFERENCES users(id)     ON DELETE SET NULL
        )""",
        """CREATE TABLE IF NOT EXISTS audit_log (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            table_name  VARCHAR(50),
            record_id   INT,
            changed_by  INT DEFAULT NULL,
            change_type ENUM('insert','update','delete'),
            old_value   TEXT,
            new_value   TEXT,
            changed_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (changed_by) REFERENCES users(id) ON DELETE SET NULL
        )""",
        # ── campaign_questions (donor Q&A) ────────────────
        """CREATE TABLE IF NOT EXISTS campaign_questions (
            id           INT AUTO_INCREMENT PRIMARY KEY,
            campaign_id  INT NOT NULL,
            donor_id     INT NOT NULL,
            question     TEXT NOT NULL,
            answer       TEXT,
            answered_by  INT DEFAULT NULL,
            answered_at  TIMESTAMP NULL,
            is_public    TINYINT(1) DEFAULT 1,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE,
            FOREIGN KEY (donor_id)    REFERENCES users(id)     ON DELETE CASCADE
        )""",
        # ── campaign_milestones ────────────────────────────
        """CREATE TABLE IF NOT EXISTS campaign_milestones (
            id           INT AUTO_INCREMENT PRIMARY KEY,
            campaign_id  INT NOT NULL,
            amount       DECIMAL(12,2) NOT NULL,
            title        VARCHAR(300)  NOT NULL,
            description  TEXT,
            reached      TINYINT(1)    DEFAULT 0,
            reached_at   TIMESTAMP     NULL,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
        )""",
        # ── ngo_documents ─────────────────────────────────
        """CREATE TABLE IF NOT EXISTS ngo_documents (
            id           INT AUTO_INCREMENT PRIMARY KEY,
            ngo_id       INT NOT NULL,
            doc_type     VARCHAR(100) NOT NULL,
            doc_url      VARCHAR(1000) NOT NULL,
            description  TEXT,
            status       ENUM('pending','approved','rejected') DEFAULT 'pending',
            admin_note   TEXT,
            reviewed_by  INT DEFAULT NULL,
            reviewed_at  TIMESTAMP NULL,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ngo_id) REFERENCES ngos(id) ON DELETE CASCADE
        )""",
    ]

    # Column migrations — compatible with ALL MySQL versions (no IF NOT EXISTS on ALTER)
    # Each entry: (table, column, definition)
    COLUMN_MIGRATIONS = [
        ("donations",        "payment_method",  "VARCHAR(50)    DEFAULT 'online'"),
        ("donations",        "transaction_id",  "VARCHAR(100)   DEFAULT NULL"),
        ("ngos",             "trust_badge",     "VARCHAR(20)    DEFAULT NULL"),
        ("ngos",             "is_verified",     "TINYINT(1)     DEFAULT 0"),
        ("fraud_logs",       "reviewed",        "TINYINT(1)     DEFAULT 0"),
        ("fraud_logs",       "admin_note",      "TEXT           DEFAULT NULL"),
        ("fraud_logs",       "reviewed_at",     "TIMESTAMP      NULL"),
        ("fraud_logs",       "reviewed_by",     "INT            DEFAULT NULL"),
        ("campaigns",        "completion_rate", "DECIMAL(6,2)   DEFAULT 0"),
        ("campaigns",        "priority_score",  "DECIMAL(8,2)   DEFAULT 0"),
        ("campaigns",        "total_donors",    "INT            DEFAULT 0"),
        ("campaigns",        "collected_amount","DECIMAL(12,2)  DEFAULT 0"),
        ("campaigns",        "donation_cap",    "DECIMAL(12,2)  DEFAULT NULL"),
        ("campaigns",        "close_message",   "TEXT           DEFAULT NULL"),
        ("campaign_expenses","bill_image",      "VARCHAR(500)   DEFAULT NULL"),
        ("campaign_expenses","receipt_url",     "VARCHAR(1000)  DEFAULT NULL"),
        ("campaign_updates", "photo_url",       "VARCHAR(1000)  DEFAULT NULL"),
        ("beneficiaries",    "age",             "INT            DEFAULT NULL"),
        ("beneficiaries",    "location",        "VARCHAR(200)   DEFAULT NULL"),
        ("users",            "profile_photo",   "LONGTEXT       DEFAULT NULL"),
        ("users",            "is_blocked",      "TINYINT(1)     DEFAULT 0"),
        ("users",            "block_reason",    "TEXT           DEFAULT NULL"),
    ]

    try:
        with app.app_context():
            cur = mysql.connection.cursor()

            # Run CREATE TABLE statements
            for stmt in DDL_STATEMENTS:
                try:
                    cur.execute(stmt)
                    mysql.connection.commit()
                except Exception as e:
                    print(f"[init_db] Skipped: {str(e)[:120]}")

            # Run column migrations — check INFORMATION_SCHEMA first
            db_name = app.config.get("MYSQL_DB", "smart_donation")
            for table, column, definition in COLUMN_MIGRATIONS:
                try:
                    cur.execute("""
                        SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_SCHEMA = %s
                          AND TABLE_NAME   = %s
                          AND COLUMN_NAME  = %s
                    """, (db_name, table, column))
                    exists = cur.fetchone()[0]
                    if not exists:
                        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                        mysql.connection.commit()
                        print(f"[init_db] Added column {table}.{column}")
                except Exception as e:
                    print(f"[init_db] Column {table}.{column} skipped: {str(e)[:80]}")

            cur.close()
            print("[init_db] Database schema check complete.")
    except Exception as e:
        print(f"[init_db] Could not connect to DB on startup: {e}")


# ── Run migration at startup ───────────────────────────────
with app.app_context():
    try:
        init_db()
    except Exception as e:
        print(f"[startup] init_db failed: {e}")


# ============================================================
#  HELPERS
# ============================================================

def role_required(required_role):
    """Decorator — restricts a route to one role."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if 'role' not in session:
                return jsonify({"status": "error", "message": "Login required"}), 401
            if session['role'] != required_role:
                return jsonify({"status": "error", "message": "Access denied"}), 403
            return f(*args, **kwargs)
        return wrapper
    return decorator


def write_audit_log(cur, table_name, record_id, change_type, old_value=None, new_value=None):
    """Write a row to audit_log. Safe to call even if the table doesn't exist yet."""
    try:
        changed_by = session.get('user_id')
        cur.execute("""
            INSERT INTO audit_log
                (table_name, record_id, changed_by, change_type, old_value, new_value)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (table_name, record_id, changed_by, change_type,
              str(old_value) if old_value is not None else None,
              str(new_value) if new_value is not None else None))
    except Exception:
        pass  # audit log failure must never break the main flow


# ============================================================
#  PAGE ROUTES
# ============================================================

@app.route('/')
def home():
    return render_template("project.html")

@app.route('/web/login')
def web_login():
    return render_template("login.html")

@app.route('/admin-login')
def admin_login_page():
    # If already logged in as admin, go straight to dashboard
    if session.get('role') == 'admin':
        return redirect(url_for('web_admin'))
    return render_template("admin_login.html")

@app.route('/web/donor')
def web_donor():
    if session.get('role') == 'donor':
        return render_template("donar_dashboard.html")
    return redirect(url_for('web_login') + '?reason=auth&role=donor')

@app.route('/web/admin')
def web_admin():
    if session.get('role') == 'admin':
        return render_template("admin_dashboard.html")
    return redirect('/admin-login')

@app.route('/web/campaign')
def web_campaign():
    return render_template("campaign_detail.html")

@app.route('/web/ngo')
def web_ngo():
    if session.get('role') == 'ngo':
        return render_template("NGO.html")
    return redirect(url_for('web_login') + '?reason=auth&role=ngo')


# ============================================================
#  AUTH
# ============================================================

@app.route('/check_session', methods=['GET'])
def check_session():
    if 'user_id' in session:
        return jsonify({
            "logged_in": True,
            "role":    session.get('role'),
            "user_id": session.get('user_id')
        })
    return jsonify({"logged_in": False})


@app.route('/current_user')
def current_user():
    if 'user_id' in session:
        return jsonify({
            "logged_in": True,
            "role":    session['role'],
            "user_id": session['user_id']
        })
    return jsonify({"logged_in": False})


@app.route('/register', methods=['POST'])
def register():
    try:
        data     = request.get_json()
        name     = data['name']
        email    = data['email']
        password = generate_password_hash(data['password'])
        role     = data['role']
        phone    = data.get('phone', '')

        cur = mysql.connection.cursor()
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            cur.close()
            return jsonify({"status": "error", "message": "Email already registered"})

        cur.execute(
            "INSERT INTO users (name, email, phone, password, role) VALUES (%s,%s,%s,%s,%s)",
            (name, email, phone, password, role)
        )
        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": "User registered successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/register_ngo', methods=['POST'])
def register_ngo():
    try:
        data     = request.get_json()
        name     = data['name']
        email    = data['email']
        password = generate_password_hash(data['password'])
        phone    = data.get('phone', '')

        cur = mysql.connection.cursor()
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            cur.close()
            return jsonify({"status": "error", "message": "Email already registered"})

        cur.execute(
            "INSERT INTO users (name, email, phone, password, role) VALUES (%s,%s,%s,%s,'ngo')",
            (name, email, phone, password)
        )
        mysql.connection.commit()
        user_id = cur.lastrowid

        cur.execute("INSERT INTO ngos (user_id) VALUES (%s)", (user_id,))
        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": "NGO registered successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/login', methods=['POST'])
def login():
    try:
        data     = request.get_json()
        email    = data['email']
        password = data['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT id, password, role FROM users WHERE email=%s", (email,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['role']    = user[2]
            return jsonify({"status": "success", "message": "Login successful", "role": user[2]})

        return jsonify({"status": "error", "message": "Invalid email or password"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/logout')
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully"})


# ============================================================
#  PROFILE
# ============================================================

@app.route('/users', methods=['GET'])
def get_users():
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, name, email, role FROM users")
    rows = cur.fetchall()
    cur.close()
    return jsonify([{"id": r[0], "name": r[1], "email": r[2], "role": r[3]} for r in rows])


@app.route('/get_user_profile', methods=['GET'])
def get_user_profile():
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "Not logged in"})
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            "SELECT id, name, email, phone, created_at, profile_photo FROM users WHERE id = %s",
            (session['user_id'],)
        )
        user = cur.fetchone()
        cur.close()
        if user:
            return jsonify({
                "status": "success",
                "user": {
                    "id":           user[0],
                    "name":         user[1] or "",
                    "email":        user[2] or "",
                    "phone":        user[3] or "",
                    "created_at":   user[4],
                    "profile_photo": user[5]
                }
            })
        return jsonify({"status": "error", "message": "User not found"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "Not logged in"})
    try:
        data  = request.get_json()
        name  = data.get('name')
        email = data.get('email')
        phone = data.get('phone')

        cur = mysql.connection.cursor()
        cur.execute(
            "SELECT id FROM users WHERE email = %s AND id != %s",
            (email, session['user_id'])
        )
        if cur.fetchone():
            cur.close()
            return jsonify({"status": "error", "message": "Email already in use"})

        cur.execute(
            "UPDATE users SET name=%s, email=%s, phone=%s WHERE id=%s",
            (name, email, phone, session['user_id'])
        )
        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": "Profile updated successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ============================================================
#  DONOR PROFILE PHOTO UPLOAD
# ============================================================

@app.route('/upload_profile_photo', methods=['POST'])
def upload_profile_photo():
    """
    Upload a profile photo as base64 string.
    Stored directly in the users table (LONGTEXT).
    Max size enforced: 2MB after base64 encoding.
    """
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "Not logged in"})
    try:
        data      = request.get_json()
        photo_b64 = data.get('photo')

        if not photo_b64:
            return jsonify({"status": "error", "message": "No photo provided"})

        # Validate it is a base64 data URI (image/jpeg or image/png)
        if not photo_b64.startswith('data:image/'):
            return jsonify({"status": "error", "message": "Invalid image format. Use JPEG or PNG."})

        # Size check — base64 string length * 0.75 ≈ actual bytes
        max_bytes = 2 * 1024 * 1024   # 2 MB
        if len(photo_b64) * 0.75 > max_bytes:
            return jsonify({"status": "error", "message": "Image too large. Maximum size is 2MB."})

        cur = mysql.connection.cursor()
        cur.execute(
            "UPDATE users SET profile_photo = %s WHERE id = %s",
            (photo_b64, session['user_id'])
        )
        mysql.connection.commit()
        cur.close()

        return jsonify({"status": "success", "message": "Profile photo updated!", "photo": photo_b64})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/delete_profile_photo', methods=['POST'])
def delete_profile_photo():
    """Remove the donor's profile photo."""
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "Not logged in"})
    try:
        cur = mysql.connection.cursor()
        cur.execute("UPDATE users SET profile_photo = NULL WHERE id = %s", (session['user_id'],))
        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": "Photo removed"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ============================================================
#  NGO DOCUMENTS
# ============================================================

# Document types and their score contributions
DOC_SCORE_MAP = {
    'registration':   30,   # NGO registration certificate
    'tax_exemption':  25,   # 80G / FCRA certificate
    'audited_report': 25,   # Latest audited financial report
    'board_resolution': 10, # Board resolution
    'pan_card':        10,  # PAN card copy
}

@app.route('/submit_document', methods=['POST'])
@role_required('ngo')
def submit_document():
    """NGO submits a document URL for admin review."""
    try:
        data    = request.get_json()
        doc_type    = data.get('doc_type', '').strip()
        doc_url     = data.get('doc_url', '').strip()
        description = data.get('description', '').strip()

        if not doc_type or not doc_url:
            return jsonify({'status': 'error', 'message': 'Document type and URL are required'})

        if not (doc_url.startswith('http://') or doc_url.startswith('https://')):
            return jsonify({'status': 'error', 'message': 'Please enter a valid URL starting with http:// or https://'})

        cur = mysql.connection.cursor()
        cur.execute('SELECT id FROM ngos WHERE user_id=%s', (session['user_id'],))
        ngo = cur.fetchone()
        if not ngo:
            cur.close()
            return jsonify({'status': 'error', 'message': 'NGO not found'})

        ngo_id = ngo[0]

        # Check if this doc type was already submitted — replace if so
        cur.execute(
            'SELECT id FROM ngo_documents WHERE ngo_id=%s AND doc_type=%s',
            (ngo_id, doc_type)
        )
        existing = cur.fetchone()

        if existing:
            cur.execute("""
                UPDATE ngo_documents
                SET doc_url=%s, description=%s, status='pending',
                    admin_note=NULL, reviewed_by=NULL, reviewed_at=NULL,
                    submitted_at=NOW()
                WHERE id=%s
            """, (doc_url, description, existing[0]))
        else:
            cur.execute("""
                INSERT INTO ngo_documents (ngo_id, doc_type, doc_url, description)
                VALUES (%s, %s, %s, %s)
            """, (ngo_id, doc_type, doc_url, description))

        mysql.connection.commit()
        cur.close()
        return jsonify({'status': 'success', 'message': 'Document submitted for review'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/get_my_documents', methods=['GET'])
@role_required('ngo')
def get_my_documents():
    """Return all documents submitted by this NGO."""
    try:
        cur = mysql.connection.cursor()
        cur.execute('SELECT id FROM ngos WHERE user_id=%s', (session['user_id'],))
        ngo = cur.fetchone()
        if not ngo:
            cur.close()
            return jsonify([])

        cur.execute("""
            SELECT id, doc_type, doc_url, description, status,
                   admin_note, submitted_at, reviewed_at
            FROM ngo_documents
            WHERE ngo_id=%s
            ORDER BY submitted_at DESC
        """, (ngo[0],))
        rows = cur.fetchall()
        cur.close()
        return jsonify([{
            'id':           r[0],
            'doc_type':     r[1],
            'doc_url':      r[2],
            'description':  r[3] or '',
            'status':       r[4],
            'admin_note':   r[5] or '',
            'submitted_at': str(r[6]),
            'reviewed_at':  str(r[7]) if r[7] else None
        } for r in rows])
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/admin/ngo_documents/<int:ngo_id>', methods=['GET'])
@role_required('admin')
def admin_get_ngo_documents(ngo_id):
    """Admin fetches all documents for a specific NGO."""
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT d.id, d.doc_type, d.doc_url, d.description,
                   d.status, d.admin_note, d.submitted_at, d.reviewed_at
            FROM ngo_documents d
            WHERE d.ngo_id=%s
            ORDER BY d.submitted_at DESC
        """, (ngo_id,))
        rows = cur.fetchall()
        cur.close()
        return jsonify([{
            'id':           r[0],
            'doc_type':     r[1],
            'doc_url':      r[2],
            'description':  r[3] or '',
            'status':       r[4],
            'admin_note':   r[5] or '',
            'submitted_at': str(r[6]),
            'reviewed_at':  str(r[7]) if r[7] else None
        } for r in rows])
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/admin/review_document/<int:doc_id>', methods=['POST'])
@role_required('admin')
def admin_review_document(doc_id):
    """Admin approves or rejects a document and updates NGO document_score."""
    try:
        data    = request.get_json()
        action  = data.get('action')       # 'approved' or 'rejected'
        note    = data.get('note', '')

        if action not in ('approved', 'rejected'):
            return jsonify({'status': 'error', 'message': 'Invalid action'})

        cur = mysql.connection.cursor()

        # Get document details
        cur.execute("""
            SELECT d.id, d.doc_type, d.ngo_id, d.status
            FROM ngo_documents d WHERE d.id=%s
        """, (doc_id,))
        doc = cur.fetchone()
        if not doc:
            cur.close()
            return jsonify({'status': 'error', 'message': 'Document not found'})

        old_status = doc[3]
        doc_type   = doc[1]
        ngo_id     = doc[2]

        # Update document status
        cur.execute("""
            UPDATE ngo_documents
            SET status=%s, admin_note=%s, reviewed_by=%s, reviewed_at=NOW()
            WHERE id=%s
        """, (action, note, session['user_id'], doc_id))

        # Recalculate document_score from scratch based on all approved docs
        cur.execute("""
            SELECT doc_type FROM ngo_documents
            WHERE ngo_id=%s AND status='approved'
        """, (ngo_id,))
        approved_types = {r[0] for r in cur.fetchall()}

        # Include the current doc if just approved
        if action == 'approved':
            approved_types.add(doc_type)
        elif action == 'rejected' and old_status == 'approved':
            approved_types.discard(doc_type)

        new_score = sum(DOC_SCORE_MAP.get(t, 10) for t in approved_types)
        new_score = min(new_score, 100)

        cur.execute(
            'UPDATE ngos SET document_score=%s WHERE id=%s',
            (new_score, ngo_id)
        )

        # Auto-upgrade trust badge based on score thresholds
        if new_score >= 80:
            badge = 'platinum'
        elif new_score >= 60:
            badge = 'gold'
        elif new_score >= 30:
            badge = 'silver'
        else:
            badge = 'bronze'

        cur.execute(
            'UPDATE ngos SET trust_badge=%s WHERE id=%s',
            (badge, ngo_id)
        )

        write_audit_log(cur, 'ngo_documents', doc_id, 'update',
                        new_value=f'action={action}, doc_type={doc_type}, score={new_score}')
        mysql.connection.commit()
        cur.close()

        return jsonify({
            'status':       'success',
            'message':      f'Document {action}',
            'new_score':    new_score,
            'new_badge':    badge
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/change_password', methods=['POST'])
def change_password():
    """Change the logged-in user's password after verifying current password."""
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "Not logged in"})
    try:
        data         = request.get_json()
        current_pw   = data.get('current_password', '')
        new_pw       = data.get('new_password', '')

        if not current_pw or not new_pw:
            return jsonify({"status": "error", "message": "Both fields are required"})

        if len(new_pw) < 8:
            return jsonify({"status": "error", "message": "New password must be at least 8 characters"})

        cur = mysql.connection.cursor()
        cur.execute("SELECT password FROM users WHERE id = %s", (session['user_id'],))
        row = cur.fetchone()
        if not row:
            cur.close()
            return jsonify({"status": "error", "message": "User not found"})

        if not check_password_hash(row[0], current_pw):
            cur.close()
            return jsonify({"status": "error", "message": "Current password is incorrect"})

        new_hash = generate_password_hash(new_pw)
        cur.execute(
            "UPDATE users SET password = %s WHERE id = %s",
            (new_hash, session['user_id'])
        )
        write_audit_log(cur, 'users', session['user_id'], 'update',
                        old_value="password changed")
        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": "Password changed successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ============================================================
#  DONOR CERTIFICATE
# ============================================================

@app.route('/generate_certificate/<int:donation_id>', methods=['GET'])
def generate_certificate(donation_id):
    """
    Generate a formal, prestigious appreciation certificate PDF.
    Design: cream background, gold borders, serif fonts, hexagonal NGO seal.
    """
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "Not logged in"})
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT d.id, d.amount, d.donation_date,
                   u.name   AS donor_name,
                   c.title  AS campaign_title,
                   c.id     AS campaign_id,
                   n_u.name AS ngo_name,
                   n.trust_badge,
                   n.is_verified
            FROM donations d
            JOIN users    u   ON d.donor_id    = u.id
            JOIN campaigns c  ON d.campaign_id = c.id
            JOIN ngos     n   ON c.ngo_id       = n.id
            JOIN users    n_u ON n.user_id       = n_u.id
            WHERE d.id = %s AND d.donor_id = %s
        """, (donation_id, session['user_id']))
        row = cur.fetchone()
        cur.close()

        if not row:
            return jsonify({"status": "error", "message": "Donation not found"})

        did, amount, date, donor_name, campaign_title, camp_id, ngo_name, trust_badge, is_verified = row

        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas as pdfcanvas
        import math, io

        # ── Color palette ─────────────────────────────────
        CREAM       = (0.99, 0.97, 0.90)
        CREAM_DARK  = (0.96, 0.93, 0.82)
        GOLD_DARK   = (0.62, 0.44, 0.02)
        GOLD_MID    = (0.80, 0.62, 0.10)
        GOLD_LIGHT  = (0.93, 0.82, 0.40)
        TEAL        = (0.05, 0.23, 0.28)
        WARM_GREY   = (0.45, 0.40, 0.32)
        WARM_LIGHT  = (0.65, 0.60, 0.50)

        def sf(cv, rgb): cv.setFillColorRGB(*rgb)
        def ss(cv, rgb): cv.setStrokeColorRGB(*rgb)

        buf = io.BytesIO()
        W, H = A4
        cv = pdfcanvas.Canvas(buf, pagesize=A4)

        # ── 1. Backgrounds ────────────────────────────────
        sf(cv, CREAM)
        cv.rect(0, 0, W, H, fill=1, stroke=0)
        sf(cv, CREAM_DARK)
        cv.rect(30, 30, W-60, H-60, fill=1, stroke=0)

        # ── 2. Gold border frames ─────────────────────────
        ss(cv, GOLD_DARK)
        cv.setLineWidth(5.5)
        cv.rect(22, 22, W-44, H-44, fill=0, stroke=1)
        ss(cv, GOLD_MID)
        cv.setLineWidth(1.2)
        cv.rect(31, 31, W-62, H-62, fill=0, stroke=1)
        ss(cv, GOLD_LIGHT)
        cv.setLineWidth(0.4)
        cv.rect(35, 35, W-70, H-70, fill=0, stroke=1)

        # ── 3. Corner ornaments ───────────────────────────
        for cx, cy in [(42,42),(W-42,42),(42,H-42),(W-42,H-42)]:
            sf(cv, GOLD_DARK); ss(cv, GOLD_DARK)
            cv.setLineWidth(0.4)
            cv.circle(cx, cy, 7, fill=1, stroke=0)
            sf(cv, CREAM_DARK)
            cv.circle(cx, cy, 4.5, fill=1, stroke=0)
            sf(cv, GOLD_MID)
            cv.circle(cx, cy, 2, fill=1, stroke=0)

        # ── 4. Header band ────────────────────────────────
        sf(cv, TEAL)
        cv.rect(40, H-130, W-80, 88, fill=1, stroke=0)
        ss(cv, GOLD_MID); cv.setLineWidth(1.0)
        cv.line(40, H-42,  W-40, H-42)
        cv.line(40, H-130, W-40, H-130)

        # Flanking ornament lines in header
        ss(cv, GOLD_LIGHT); cv.setLineWidth(0.4)
        cv.line(52, H-86, 130, H-86)
        cv.line(W-130, H-86, W-52, H-86)
        sf(cv, GOLD_LIGHT); cv.circle(51, H-86, 2.5, fill=1, stroke=0)
        cv.circle(W-51, H-86, 2.5, fill=1, stroke=0)

        sf(cv, GOLD_LIGHT)
        cv.setFont("Times-Bold", 30)
        cv.drawCentredString(W/2, H-88, "TrustFlow")
        sf(cv, (0.82, 0.76, 0.60))
        cv.setFont("Times-Italic", 10.5)
        cv.drawCentredString(W/2, H-108, "Transparent Donation Platform  ·  India")

        # ── 5. Certificate title ──────────────────────────
        sf(cv, GOLD_DARK)
        cv.setFont("Times-Bold", 21)
        cv.drawCentredString(W/2, H-163, "CERTIFICATE  OF  APPRECIATION")

        # Double rule under title
        ss(cv, GOLD_MID); cv.setLineWidth(0.9)
        cv.line(90, H-173, W-90, H-173)
        ss(cv, GOLD_LIGHT); cv.setLineWidth(0.3)
        cv.line(100, H-177, W-100, H-177)

        # ── 6. Presented to ───────────────────────────────
        sf(cv, WARM_GREY)
        cv.setFont("Times-Italic", 13)
        cv.drawCentredString(W/2, H-212, "This certificate is proudly presented to")

        # ── 7. Donor name ─────────────────────────────────
        dn = donor_name if len(donor_name) <= 42 else donor_name[:39]+"..."
        sf(cv, TEAL)
        cv.setFont("Times-BoldItalic", 34)
        cv.drawCentredString(W/2, H-256, dn)

        # Flourish underline with end dots
        nw = cv.stringWidth(dn, "Times-BoldItalic", 34)
        ss(cv, GOLD_MID); cv.setLineWidth(0.9)
        cv.line(W/2-nw/2-22, H-264, W/2+nw/2+22, H-264)
        sf(cv, GOLD_MID)
        cv.circle(W/2-nw/2-22, H-264, 2.8, fill=1, stroke=0)
        cv.circle(W/2+nw/2+22, H-264, 2.8, fill=1, stroke=0)

        # ── 8. Body text ──────────────────────────────────
        sf(cv, WARM_GREY)
        cv.setFont("Times-Roman", 13)
        cv.drawCentredString(W/2, H-298, "in recognition of a generous contribution of")

        # ── 9. Amount ─────────────────────────────────────
        sf(cv, GOLD_DARK)
        cv.setFont("Times-Bold", 33)
        cv.drawCentredString(W/2, H-338, f"Rs. {float(amount):,.2f}")

        # Subtle amount underline
        aw = cv.stringWidth(f"Rs. {float(amount):,.2f}", "Times-Bold", 33)
        ss(cv, GOLD_LIGHT); cv.setLineWidth(0.5)
        cv.line(W/2-aw/2, H-345, W/2+aw/2, H-345)

        # ── 10. Campaign ──────────────────────────────────
        sf(cv, WARM_GREY)
        cv.setFont("Times-Roman", 13)
        cv.drawCentredString(W/2, H-370, "towards the campaign")

        sf(cv, TEAL)
        cv.setFont("Times-BoldItalic", 15)
        td = campaign_title if len(campaign_title)<=52 else campaign_title[:49]+"..."
        cv.drawCentredString(W/2, H-394, f"“{td}”")

        # ── 11. NGO seal ──────────────────────────────────
        sf(cv, WARM_GREY)
        cv.setFont("Times-Roman", 12)
        cv.drawCentredString(W/2, H-418, "organised by")

        # Hexagonal seal
        bx, by = W/2, H-464
        br = 30
        sf(cv, TEAL); ss(cv, GOLD_MID); cv.setLineWidth(1.5)
        p = cv.beginPath()
        for i in range(6):
            ang = math.pi/2 + i*math.pi/3
            px,py = bx+br*math.cos(ang), by+br*math.sin(ang)
            p.moveTo(px,py) if i==0 else p.lineTo(px,py)
        p.close(); cv.drawPath(p, fill=1, stroke=1)

        # Inner hex ring
        ss(cv, GOLD_LIGHT); cv.setLineWidth(0.6)
        p2 = cv.beginPath()
        for i in range(6):
            ang = math.pi/2 + i*math.pi/3
            px,py = bx+(br-6)*math.cos(ang), by+(br-6)*math.sin(ang)
            p2.moveTo(px,py) if i==0 else p2.lineTo(px,py)
        p2.close(); cv.drawPath(p2, fill=0, stroke=1)

        # Star glyph in seal
        sf(cv, GOLD_LIGHT)
        cv.setFont("Times-Bold", 16)
        cv.drawCentredString(bx, by-6, "★")

        # NGO name + badge text
        sf(cv, TEAL)
        cv.setFont("Times-Bold", 15)
        nn = ngo_name if len(ngo_name)<=40 else ngo_name[:37]+"..."
        cv.drawCentredString(W/2, H-503, nn)

        badge_label = (trust_badge or "bronze").capitalize()
        verified_str = "Verified NGO" if is_verified else "Registered NGO"
        sf(cv, WARM_LIGHT)
        cv.setFont("Times-Italic", 9.5)
        cv.drawCentredString(W/2, H-518, f"{verified_str}  ·  {badge_label} Badge")

        # ── 12. Decorative centre divider ─────────────────
        ss(cv, GOLD_MID); cv.setLineWidth(0.5)
        cv.line(80, H-533, W/2-22, H-533)
        cv.line(W/2+22, H-533, W-80, H-533)
        sf(cv, GOLD_MID)
        cv.circle(W/2, H-533, 3.5, fill=1, stroke=0)
        cv.circle(W/2-16, H-533, 1.8, fill=1, stroke=0)
        cv.circle(W/2+16, H-533, 1.8, fill=1, stroke=0)

        # ── 13. Quote ─────────────────────────────────────
        sf(cv, WARM_GREY)
        cv.setFont("Times-Italic", 11)
        cv.drawCentredString(W/2, H-556,
            "“No act of kindness, no matter how small, is ever wasted.”")
        sf(cv, WARM_LIGHT)
        cv.setFont("Times-Roman", 9.5)
        cv.drawCentredString(W/2, H-570, "— Aesop")

        # ── 14. Footer info box ───────────────────────────
        sf(cv, TEAL)
        cv.roundRect(50, 58, W-100, 54, 4, fill=1, stroke=0)
        ss(cv, GOLD_MID); cv.setLineWidth(0.9)
        cv.roundRect(50, 58, W-100, 54, 4, fill=0, stroke=1)
        ss(cv, GOLD_LIGHT); cv.setLineWidth(0.3)
        cv.roundRect(53, 61, W-106, 48, 3, fill=0, stroke=1)

        sf(cv, GOLD_LIGHT)
        cv.setFont("Times-Roman", 9.5)
        cv.drawCentredString(W/2, 100,
            f"Date: {str(date)[:10]}     •     Certificate No: TF-{did:06d}     •     trustflow.in")
        sf(cv, (0.72, 0.67, 0.52))
        cv.setFont("Times-Italic", 8.2)
        cv.drawCentredString(W/2, 80,
            "Digitally generated certificate. Valid without physical signature.")

        cv.save()
        buf.seek(0)
        return send_file(buf, as_attachment=True,
                         download_name=f"TrustFlow_Certificate_{did}.pdf",
                         mimetype='application/pdf')
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ============================================================
#  DONOR LEADERBOARD
# ============================================================

@app.route('/campaign_leaderboard/<int:campaign_id>', methods=['GET'])
def campaign_leaderboard(campaign_id):
    """Top 10 donors for a campaign — public route."""
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT u.name,
                   COALESCE(SUM(d.amount), 0) AS total,
                   COUNT(d.id)                AS num_donations,
                   u.profile_photo,
                   c.title AS campaign_title
            FROM donations d
            JOIN users u         ON d.donor_id    = u.id
            JOIN campaigns c     ON d.campaign_id = c.id
            WHERE d.campaign_id = %s AND d.status = 'success'
            GROUP BY d.donor_id, u.name, u.profile_photo, c.title
            ORDER BY total DESC
            LIMIT 10
        """, (campaign_id,))
        rows = cur.fetchall()
        cur.close()
        return jsonify([{
            "rank":           i + 1,
            "donor_name":     r[0] or "Anonymous",
            "total_donated":  float(r[1]),
            "num_donations":  int(r[2]),
            "profile_photo":  r[3],
            "campaign_title": r[4] or ""
        } for i, r in enumerate(rows)])
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ============================================================
#  CAMPAIGN MILESTONES
# ============================================================

@app.route('/add_milestone', methods=['POST'])
@role_required('ngo')
def add_milestone():
    """NGO adds a milestone to a campaign."""
    try:
        data = request.get_json()
        cur  = mysql.connection.cursor()
        cur.execute("""
            SELECT c.id FROM campaigns c JOIN ngos n ON c.ngo_id = n.id
            WHERE c.id=%s AND n.user_id=%s
        """, (data['campaign_id'], session['user_id']))
        if not cur.fetchone():
            cur.close()
            return jsonify({"status": "error", "message": "Unauthorized"})

        cur.execute("""
            INSERT INTO campaign_milestones (campaign_id, amount, title, description)
            VALUES (%s, %s, %s, %s)
        """, (data['campaign_id'], data['amount'],
              data['title'], data.get('description', '')))
        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": "Milestone added"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/get_milestones/<int:campaign_id>', methods=['GET'])
def get_milestones(campaign_id):
    """Public — get all milestones for a campaign."""
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT id, amount, title, description, reached, reached_at
            FROM campaign_milestones
            WHERE campaign_id=%s ORDER BY amount ASC
        """, (campaign_id,))
        rows = cur.fetchall()

        # Get current collected_amount to show progress
        cur.execute("SELECT collected_amount FROM campaigns WHERE id=%s", (campaign_id,))
        camp = cur.fetchone()
        collected = float(camp[0]) if camp and camp[0] else 0
        cur.close()

        return jsonify({
            "collected": collected,
            "milestones": [{
                "id":          r[0],
                "amount":      float(r[1]),
                "title":       r[2],
                "description": r[3] or "",
                "reached":     bool(r[4]),
                "reached_at":  str(r[5]) if r[5] else None
            } for r in rows]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ============================================================
#  DONOR Q&A
# ============================================================

@app.route('/ask_question', methods=['POST'])
@role_required('donor')
def ask_question():
    """Donor asks a question about a campaign."""
    try:
        data     = request.get_json()
        question = data.get('question', '').strip()
        if not question:
            return jsonify({"status": "error", "message": "Question cannot be empty"})

        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO campaign_questions (campaign_id, donor_id, question)
            VALUES (%s, %s, %s)
        """, (data['campaign_id'], session['user_id'], question))

        # Notify the NGO
        cur.execute("""
            SELECT n.user_id FROM campaigns c JOIN ngos n ON c.ngo_id = n.id
            WHERE c.id = %s
        """, (data['campaign_id'],))
        ngo_row = cur.fetchone()
        if ngo_row:
            cur.execute("""
                INSERT INTO notifications (user_id, message, type)
                VALUES (%s, %s, 'question')
            """, (ngo_row[0], f"New question on your campaign: {question[:80]}..."))

        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": "Question submitted"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/get_questions/<int:campaign_id>', methods=['GET'])
def get_questions(campaign_id):
    """Public — get all Q&A for a campaign."""
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT q.id, q.question, q.answer, q.created_at, q.answered_at,
                   u.name AS donor_name,
                   a.name AS answered_by_name
            FROM campaign_questions q
            JOIN users u ON q.donor_id = u.id
            LEFT JOIN users a ON q.answered_by = a.id
            WHERE q.campaign_id = %s AND q.is_public = 1
            ORDER BY q.created_at DESC
        """, (campaign_id,))
        rows = cur.fetchall()
        cur.close()
        return jsonify([{
            "id":              r[0],
            "question":        r[1],
            "answer":          r[2],
            "asked_at":        str(r[3]),
            "answered_at":     str(r[4]) if r[4] else None,
            "donor_name":      r[5] or "Anonymous",
            "answered_by":     r[6] or None
        } for r in rows])
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/answer_question/<int:question_id>', methods=['POST'])
@role_required('ngo')
def answer_question(question_id):
    """NGO answers a donor question."""
    try:
        data   = request.get_json()
        answer = data.get('answer', '').strip()
        if not answer:
            return jsonify({"status": "error", "message": "Answer cannot be empty"})

        cur = mysql.connection.cursor()
        # Verify the question belongs to this NGO's campaign
        cur.execute("""
            SELECT q.id, q.donor_id, q.campaign_id FROM campaign_questions q
            JOIN campaigns c ON q.campaign_id = c.id
            JOIN ngos n      ON c.ngo_id       = n.id
            WHERE q.id = %s AND n.user_id = %s
        """, (question_id, session['user_id']))
        q = cur.fetchone()
        if not q:
            cur.close()
            return jsonify({"status": "error", "message": "Unauthorized"})

        cur.execute("""
            UPDATE campaign_questions
            SET answer=%s, answered_by=%s, answered_at=NOW()
            WHERE id=%s
        """, (answer, session['user_id'], question_id))

        # Notify the donor
        cur.execute("""
            INSERT INTO notifications (user_id, message, type)
            VALUES (%s, %s, 'answer')
        """, (q[1], f"Your question on campaign #{q[2]} has been answered!"))

        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": "Answer posted"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ============================================================
#  DONATION CAP + CAMPAIGN DEADLINE MANAGEMENT
# ============================================================

@app.route('/set_donation_cap', methods=['POST'])
@role_required('ngo')
def set_donation_cap():
    """NGO sets a maximum funding cap on their campaign."""
    try:
        data = request.get_json()
        cap  = float(data.get('cap', 0))
        cur  = mysql.connection.cursor()
        cur.execute("""
            SELECT c.id FROM campaigns c JOIN ngos n ON c.ngo_id = n.id
            WHERE c.id=%s AND n.user_id=%s
        """, (data['campaign_id'], session['user_id']))
        if not cur.fetchone():
            cur.close()
            return jsonify({"status": "error", "message": "Unauthorized"})
        cur.execute("UPDATE campaigns SET donation_cap=%s WHERE id=%s",
                    (cap if cap > 0 else None, data['campaign_id']))
        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": "Donation cap set"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/admin/campaign/<int:campaign_id>/extend', methods=['POST'])
@role_required('admin')
def admin_extend_campaign(campaign_id):
    """Admin extends a campaign deadline."""
    try:
        data         = request.get_json()
        new_deadline = data.get('new_deadline')
        if not new_deadline:
            return jsonify({"status": "error", "message": "New deadline required"})

        cur = mysql.connection.cursor()
        cur.execute("SELECT deadline, title FROM campaigns WHERE id=%s", (campaign_id,))
        row = cur.fetchone()
        if not row:
            cur.close()
            return jsonify({"status": "error", "message": "Campaign not found"})

        cur.execute("""
            UPDATE campaigns SET deadline=%s, status='approved' WHERE id=%s
        """, (new_deadline, campaign_id))

        # Notify NGO
        cur.execute("""
            SELECT n.user_id FROM campaigns c JOIN ngos n ON c.ngo_id = n.id
            WHERE c.id = %s
        """, (campaign_id,))
        ngo_row = cur.fetchone()
        if ngo_row:
            cur.execute("""
                INSERT INTO notifications (user_id, message, type)
                VALUES (%s, %s, 'info')
            """, (ngo_row[0],
                  f"Your campaign '{row[1]}' deadline has been extended to {new_deadline}."))

        write_audit_log(cur, 'campaigns', campaign_id, 'update',
                        old_value=f"deadline={row[0]}",
                        new_value=f"deadline={new_deadline}")
        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": "Campaign deadline extended"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/admin/campaign/<int:campaign_id>/close', methods=['POST'])
@role_required('admin')
def admin_close_campaign(campaign_id):
    """Admin closes a campaign with a message (missed deadline / underfunded)."""
    try:
        data    = request.get_json()
        message = data.get('message', 'This campaign has been closed.')

        cur = mysql.connection.cursor()
        cur.execute("SELECT title FROM campaigns WHERE id=%s", (campaign_id,))
        row = cur.fetchone()
        if not row:
            cur.close()
            return jsonify({"status": "error", "message": "Campaign not found"})

        cur.execute("""
            UPDATE campaigns SET status='rejected', close_message=%s WHERE id=%s
        """, (message, campaign_id))

        # Notify all donors who gave to this campaign
        cur.execute("""
            SELECT DISTINCT donor_id FROM donations WHERE campaign_id=%s
        """, (campaign_id,))
        donor_ids = [r[0] for r in cur.fetchall()]
        for did in donor_ids:
            cur.execute("""
                INSERT INTO notifications (user_id, message, type)
                VALUES (%s, %s, 'warning')
            """, (did, f"Campaign '{row[0]}' has been closed. {message}"))

        # Notify NGO
        cur.execute("""
            SELECT n.user_id FROM campaigns c JOIN ngos n ON c.ngo_id = n.id
            WHERE c.id = %s
        """, (campaign_id,))
        ngo_row = cur.fetchone()
        if ngo_row:
            cur.execute("""
                INSERT INTO notifications (user_id, message, type)
                VALUES (%s, %s, 'warning')
            """, (ngo_row[0], f"Your campaign '{row[0]}' has been closed by admin."))

        write_audit_log(cur, 'campaigns', campaign_id, 'update',
                        new_value=f"status=rejected, close_message set")
        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": "Campaign closed"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ============================================================
#  FRAUD — BLOCK DONOR / MANUAL VERIFICATION
# ============================================================

@app.route('/admin/block_donor/<int:user_id>', methods=['POST'])
@role_required('admin')
def admin_block_donor(user_id):
    """Block a donor account after fraud review."""
    try:
        data   = request.get_json()
        reason = data.get('reason', 'Blocked due to suspicious activity')
        cur    = mysql.connection.cursor()
        cur.execute("""
            UPDATE users SET is_blocked=1, block_reason=%s WHERE id=%s
        """, (reason, user_id))
        write_audit_log(cur, 'users', user_id, 'update',
                        new_value=f"is_blocked=1, reason={reason}")
        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": "Donor blocked"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/admin/unblock_donor/<int:user_id>', methods=['POST'])
@role_required('admin')
def admin_unblock_donor(user_id):
    """Unblock a donor after manual verification."""
    try:
        cur = mysql.connection.cursor()
        cur.execute("UPDATE users SET is_blocked=0, block_reason=NULL WHERE id=%s", (user_id,))
        write_audit_log(cur, 'users', user_id, 'update', new_value="is_blocked=0 (unblocked)")
        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": "Donor unblocked"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ============================================================
#  EXPORT TO CSV
# ============================================================

@app.route('/admin/export_donations_csv', methods=['GET'])
@role_required('admin')
def admin_export_donations_csv():
    """Export full donation ledger as CSV."""
    import csv, io
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT d.id, u.name, u.email, c.title, d.amount,
                   d.payment_method, d.status, d.donation_date, d.transaction_id
            FROM donations d
            JOIN users u     ON d.donor_id    = u.id
            JOIN campaigns c ON d.campaign_id = c.id
            ORDER BY d.donation_date DESC
        """)
        rows = cur.fetchall()
        cur.close()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID','Donor Name','Email','Campaign','Amount (Rs.)',
                         'Method','Status','Date','Transaction ID'])
        for r in rows:
            writer.writerow([r[0], r[1], r[2], r[3],
                             f"{float(r[4]):,.2f}", r[5] or 'online', r[6] or 'success',
                             r[7].strftime('%d/%m/%Y') if r[7] else '', r[8] or ''])

        output.seek(0)
        return send_file(
            io.BytesIO(('\ufeff'+output.getvalue()).encode('utf-8')),
            as_attachment=True,
            download_name='trustflow_donations.csv',
            mimetype='text/csv'
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/admin/export_ngos_csv', methods=['GET'])
@role_required('admin')
def admin_export_ngos_csv():
    """Export NGO list as CSV."""
    import csv, io
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT u.id, u.name, u.email, n.trust_badge,
                   n.is_verified, n.document_score, u.created_at
            FROM users u JOIN ngos n ON n.user_id = u.id
            ORDER BY u.created_at DESC
        """)
        rows = cur.fetchall()
        cur.close()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID','Name','Email','Badge','Verified','Doc Score','Joined'])
        for r in rows:
            writer.writerow([r[0], r[1], r[2], r[3] or 'None',
                             'Yes' if r[4] else 'No', r[5] or 0, r[6].strftime('%d/%m/%Y') if r[6] else ''])
        output.seek(0)
        return send_file(
            io.BytesIO(('\ufeff'+output.getvalue()).encode('utf-8')),
            as_attachment=True,
            download_name='trustflow_ngos.csv',
            mimetype='text/csv'
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ============================================================
#  EXPORT CAMPAIGNS CSV
# ============================================================

@app.route('/admin/export_campaigns_csv', methods=['GET'])
@role_required('admin')
def admin_export_campaigns_csv():
    """Export campaigns list as CSV with full details."""
    import csv, io
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT c.id, c.title, u.name AS ngo_name, c.category,
                   c.target_amount, c.collected_amount, c.completion_rate,
                   c.total_donors, c.status, c.urgency_level,
                   c.deadline, c.created_at
            FROM campaigns c
            JOIN ngos n   ON c.ngo_id  = n.id
            JOIN users u  ON n.user_id = u.id
            ORDER BY c.created_at DESC
        """)
        rows = cur.fetchall()
        cur.close()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID','Title','NGO','Category','Target (Rs.)','Raised (Rs.)',
                         'Progress %','Donors','Status','Urgency','Deadline','Created'])
        for r in rows:
            writer.writerow([
                r[0], r[1], r[2], r[3],
                f"{float(r[4]):,.2f}" if r[4] else 0,
                f"{float(r[5]):,.2f}" if r[5] else 0,
                f"{float(r[6]):.1f}" if r[6] else "0.0",
                r[7] or 0, r[8] or 'pending', r[9] or 5,
                r[10].strftime('%d/%m/%Y') if r[10] else '',
                r[11].strftime('%d/%m/%Y') if r[11] else ''
            ])
        output.seek(0)
        return send_file(
            io.BytesIO(('\ufeff'+output.getvalue()).encode('utf-8')),
            as_attachment=True,
            download_name='trustflow_campaigns.csv',
            mimetype='text/csv'
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ============================================================
#  COMPLETED CAMPAIGN — appreciation summary
# ============================================================

@app.route('/campaign_completion_summary/<int:campaign_id>', methods=['GET'])
def campaign_completion_summary(campaign_id):
    """Public summary for a completed campaign — stats + top donors + message."""
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT c.title, c.description, c.target_amount, c.collected_amount,
                   c.total_donors, c.status, c.close_message,
                   u.name AS ngo_name
            FROM campaigns c
            JOIN ngos n   ON c.ngo_id  = n.id
            JOIN users u  ON n.user_id = u.id
            WHERE c.id = %s
        """, (campaign_id,))
        row = cur.fetchone()
        if not row:
            cur.close()
            return jsonify({"status": "error", "message": "Campaign not found"})

        # Top 5 donors
        cur.execute("""
            SELECT u.name, COALESCE(SUM(d.amount),0) AS total
            FROM donations d JOIN users u ON d.donor_id = u.id
            WHERE d.campaign_id=%s AND d.status='success'
            GROUP BY d.donor_id, u.name
            ORDER BY total DESC LIMIT 5
        """, (campaign_id,))
        top_donors = cur.fetchall()

        # Total expenses logged
        cur.execute("""
            SELECT COUNT(*), COALESCE(SUM(amount),0)
            FROM campaign_expenses WHERE campaign_id=%s
        """, (campaign_id,))
        exp_row = cur.fetchone()
        cur.close()

        pct = round((float(row[3])/float(row[2])*100), 1) if row[2] else 0
        return jsonify({
            "status":          "success",
            "title":           row[0],
            "description":     row[1],
            "target":          float(row[2]) if row[2] else 0,
            "collected":       float(row[3]) if row[3] else 0,
            "pct_funded":      pct,
            "total_donors":    row[4] or 0,
            "campaign_status": row[5],
            "close_message":   row[6],
            "ngo_name":        row[7],
            "was_successful":  row[5] == 'completed' or pct >= 80,
            "top_donors":      [{"name": d[0], "amount": float(d[1])} for d in top_donors],
            "expenses_logged": int(exp_row[0]) if exp_row else 0,
            "total_spent":     float(exp_row[1]) if exp_row else 0
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ============================================================
#  NOTIFICATIONS
# ============================================================

@app.route('/get_notification_count', methods=['GET'])
def get_notification_count():
    if 'user_id' not in session:
        return jsonify({"count": 0})
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM notifications WHERE user_id=%s AND is_read=FALSE",
            (session['user_id'],)
        )
        count = cur.fetchone()[0]
        cur.close()
        return jsonify({"count": count})
    except Exception:
        return jsonify({"count": 0})


@app.route('/get_notifications', methods=['GET'])
def get_notifications():
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "Not logged in"})
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT id, message, type, is_read, created_at
            FROM notifications
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 20
        """, (session['user_id'],))
        rows = cur.fetchall()
        cur.close()
        return jsonify([{
            "id":         r[0],
            "message":    r[1],
            "type":       r[2],
            "is_read":    bool(r[3]),
            "created_at": str(r[4])
        } for r in rows])
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/mark_notifications_read', methods=['POST'])
def mark_notifications_read():
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "Not logged in"})
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            "UPDATE notifications SET is_read=TRUE WHERE user_id=%s",
            (session['user_id'],)
        )
        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ============================================================
#  CAMPAIGNS  (public)
# ============================================================

@app.route('/campaign_public/<int:campaign_id>', methods=['GET'])
def campaign_public(campaign_id):
    """Public campaign detail — no auth required. Used by campaign detail page."""
    try:
        cur = mysql.connection.cursor()
        # Full campaign info + NGO name + trust badge
        cur.execute("""
            SELECT c.id, c.title, c.description, c.category,
                   c.target_amount, c.collected_amount, c.total_donors,
                   c.urgency_level, c.severity_score, c.deadline,
                   c.status, c.completion_rate, c.priority_score,
                   u.name AS ngo_name, n.trust_badge, n.is_verified
            FROM campaigns c
            JOIN ngos n ON c.ngo_id = n.id
            JOIN users u ON n.user_id = u.id
            WHERE c.id = %s
        """, (campaign_id,))
        row = cur.fetchone()
        if not row:
            cur.close()
            return jsonify({"status": "error", "message": "Campaign not found"}), 404

        # Campaign updates (latest 10)
        cur.execute("""
            SELECT update_text, proof_document, created_at
            FROM campaign_updates
            WHERE campaign_id = %s
            ORDER BY created_at DESC LIMIT 10
        """, (campaign_id,))
        updates = cur.fetchall()

        # Fund allocations
        cur.execute("""
            SELECT category, amount, description, allocation_date
            FROM fund_allocation
            WHERE campaign_id = %s
            ORDER BY allocation_date DESC
        """, (campaign_id,))
        allocations = cur.fetchall()

        # Beneficiaries
        cur.execute("""
            SELECT name, story, age, location
            FROM beneficiaries
            WHERE campaign_id = %s
            ORDER BY id DESC LIMIT 6
        """, (campaign_id,))
        beneficiaries = cur.fetchall()

        # Recent donors (anonymous amounts shown)
        cur.execute("""
            SELECT u.name, d.amount, d.donation_date
            FROM donations d JOIN users u ON d.donor_id = u.id
            WHERE d.campaign_id = %s AND d.status = 'success'
            ORDER BY d.donation_date DESC LIMIT 10
        """, (campaign_id,))
        donors = cur.fetchall()

        # Expenses — wrapped separately so a missing table never
        # crashes the whole endpoint during first-run before init_db
        try:
            cur.execute("""
                SELECT title, amount, expense_date, description
                FROM campaign_expenses
                WHERE campaign_id = %s
                ORDER BY expense_date DESC LIMIT 10
            """, (campaign_id,))
            expenses = cur.fetchall()
        except Exception:
            expenses = []

        cur.close()

        return jsonify({
            "status": "success",
            "campaign": {
                "id":               row[0],
                "title":            row[1] or "",
                "description":      row[2] or "",
                "category":         row[3] or "General",
                "target_amount":    float(row[4]) if row[4] else 0,
                "collected_amount": float(row[5]) if row[5] else 0,
                "total_donors":     row[6] or 0,
                "urgency_level":    row[7] or 5,
                "severity_score":   row[8] or 5,
                "deadline":         str(row[9]) if row[9] else "",
                "status":           row[10] or "pending",
                "completion_rate":  float(row[11]) if row[11] else 0,
                "priority_score":   float(row[12]) if row[12] else 0,
                "ngo_name":         row[13] or "Unknown NGO",
                "trust_badge":      row[14] or "bronze",
                "is_verified":      bool(row[15])
            },
            "updates": [{
                "text":  u[0] or "",
                "proof": u[1] or None,
                "date":  str(u[2])
            } for u in updates],
            "allocations": [{
                "category":    a[0] or "",
                "amount":      float(a[1]) if a[1] else 0,
                "description": a[2] or "",
                "date":        str(a[3])
            } for a in allocations],
            "beneficiaries": [{
                "name":     b[0] or "Anonymous",
                "story":    b[1] or "",
                "age":      b[2],
                "location": b[3] or ""
            } for b in beneficiaries],
            "recent_donors": [{
                "name":   d[0] or "Anonymous",
                "amount": float(d[1]) if d[1] else 0,
                "date":   str(d[2])
            } for d in donors],
            "expenses": [{
                "title":       e[0] or "",
                "amount":      float(e[1]) if e[1] else 0,
                "date":        str(e[2]),
                "description": e[3] or ""
            } for e in expenses]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/campaigns', methods=['GET'])
def get_campaigns():
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT id, title, category, target_amount,
                   collected_amount, status, ngo_id, description,
                   total_donors, deadline, completion_rate, urgency_level
            FROM campaigns
            ORDER BY priority_score DESC, id DESC
        """)
        rows = cur.fetchall()
        cur.close()
        return jsonify([{
            "campaign_id":      r[0],
            "title":            r[1] or "Untitled Campaign",
            "category":         r[2] or "General",
            "target_amount":    float(r[3]) if r[3] else 0,
            "collected_amount": float(r[4]) if r[4] else 0,
            "status":           r[5] or "pending",
            "ngo_id":           r[6],
            "description":      r[7] or "No description available",
            "total_donors":     r[8] or 0,
            "deadline":         str(r[9]) if r[9] else None,
            "completion_rate":  float(r[10]) if r[10] else 0,
            "urgency_level":    r[11] or 5
        } for r in rows])
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/recommend_campaigns', methods=['GET'])
def recommend_campaigns():
    """
    Returns campaigns ranked by the AI recommendation engine.
    For logged-in donors: uses category-affinity collaborative filtering.
    For anonymous/new donors: falls back to AI priority score ranking.
    """
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT id, title, category, target_amount,
                   collected_amount, completion_rate, priority_score,
                   urgency_level, deadline, status, description, total_donors
            FROM campaigns
            WHERE status IN ('approved', 'pending')
        """)
        rows = cur.fetchall()
        all_campaigns = [{
            "campaign_id":      r[0],
            "title":            r[1] or "Untitled",
            "category":         r[2] or "General",
            "target_amount":    float(r[3]) if r[3] else 0,
            "collected_amount": float(r[4]) if r[4] else 0,
            "completion_rate":  float(r[5]) if r[5] else 0,
            "priority_score":   float(r[6]) if r[6] else 0,
            "urgency_level":    r[7] or 5,
            "deadline":         str(r[8]) if r[8] else None,
            "status":           r[9] or "pending",
            "description":      r[10] or "",
            "total_donors":     r[11] or 0
        } for r in rows]

        donor_id = session.get('user_id')
        recommended = ai.recommend_campaigns(donor_id, all_campaigns, cur)
        cur.close()
        return jsonify(recommended)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/ranked_campaigns', methods=['GET'])
def ranked_campaigns():
    """
    FIX: was WHERE status='approved' — returned empty because all campaigns
    are 'pending'. Now includes both approved and pending.
    """
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT id, title, category, target_amount,
                   collected_amount, completion_rate, priority_score,
                   urgency_level, deadline, status
            FROM campaigns
            WHERE status IN ('approved', 'pending')
            ORDER BY priority_score DESC, urgency_level DESC
        """)
        rows = cur.fetchall()
        cur.close()
        return jsonify([{
            "campaign_id":      r[0],
            "title":            r[1] or "Untitled",
            "category":         r[2] or "General",
            "target_amount":    float(r[3]) if r[3] else 0,
            "collected_amount": float(r[4]) if r[4] else 0,
            "completion_rate":  float(r[5]) if r[5] else 0,
            "priority_score":   float(r[6]) if r[6] else 0,
            "urgency_level":    r[7] or 0,
            "deadline":         str(r[8]) if r[8] else None,
            "status":           r[9] or "pending"
        } for r in rows])
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/fund_alerts', methods=['GET'])
def fund_alerts():
    """
    Smart campaign health alerts — 4 distinct alert types with severity scores.
    Each campaign is scored independently; alerts are sorted by severity.

    Alert types:
      1. DEADLINE CRITICAL  — deadline within 7 days, funded < 50%
      2. STALLED CAMPAIGN   — no donations in last 14 days, funded < 60%
      3. LOW MOMENTUM       — funded < 20% with more than 7 days left
      4. NEARLY THERE       — funded >= 75% — positive nudge to close the gap
    """
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT c.id, c.title, c.urgency_level, c.completion_rate,
                   c.deadline, c.collected_amount, c.target_amount,
                   c.total_donors,
                   MAX(d.donation_date) AS last_donation
            FROM campaigns c
            LEFT JOIN donations d ON d.campaign_id = c.id AND d.status = 'success'
            WHERE c.status = 'approved'
            GROUP BY c.id
        """)
        rows = cur.fetchall()
        cur.close()

        alerts = []
        today  = datetime.today().date()

        for r in rows:
            cid        = r[0]
            title      = r[1]
            urgency    = r[2] or 5
            completion = float(r[3]) if r[3] else 0
            deadline   = r[4]
            collected  = float(r[5]) if r[5] else 0
            target     = float(r[6]) if r[6] else 0
            total_don  = r[7] or 0
            last_don   = r[8]

            days_left = (deadline - today).days if deadline else None
            gap_amount = max(0, target - collected)

            # --- Alert 1: DEADLINE CRITICAL ---
            if deadline and days_left is not None and days_left <= 7 and days_left >= 0 and completion < 50:
                severity = round(min(100, (1 - completion/100) * (urgency/10) * 100), 1)
                alerts.append({
                    "campaign_id":     cid,
                    "title":          title,
                    "completion_rate": round(completion, 1),
                    "days_left":      days_left,
                    "gap_amount":     round(gap_amount, 0),
                    "total_donors":   total_don,
                    "alert_type":     "DEADLINE CRITICAL",
                    "alert_label":    f"Deadline in {days_left} day{'s' if days_left!=1 else ''} — only {completion:.1f}% funded",
                    "severity":       severity,
                    "action":         "Extend deadline or feature campaign prominently"
                })

            # --- Alert 2: STALLED CAMPAIGN ---
            elif last_don and completion < 60:
                days_since = (today - last_don.date()).days if hasattr(last_don, 'date') else 99
                if days_since >= 14:
                    severity = round(min(100, (days_since / 30) * 60 + (1 - completion/100) * 40), 1)
                    alerts.append({
                        "campaign_id":     cid,
                        "title":          title,
                        "completion_rate": round(completion, 1),
                        "days_left":      days_left,
                        "gap_amount":     round(gap_amount, 0),
                        "total_donors":   total_don,
                        "alert_type":     "STALLED",
                        "alert_label":    f"No donations for {days_since} days — campaign losing momentum",
                        "severity":       severity,
                        "action":         "Send donor re-engagement notification"
                    })

            # --- Alert 3: LOW MOMENTUM ---
            elif completion < 20 and (days_left is None or days_left > 7) and total_don < 5:
                severity = round((1 - completion/100) * urgency * 8, 1)
                alerts.append({
                    "campaign_id":     cid,
                    "title":          title,
                    "completion_rate": round(completion, 1),
                    "days_left":      days_left,
                    "gap_amount":     round(gap_amount, 0),
                    "total_donors":   total_don,
                    "alert_type":     "LOW MOMENTUM",
                    "alert_label":    f"Only {total_don} donor{'s' if total_don!=1 else ''} — needs visibility boost",
                    "severity":       severity,
                    "action":         "Feature on landing page or notify similar donors"
                })

            # --- Alert 4: NEARLY THERE (positive) ---
            elif completion >= 75 and completion < 100 and (days_left is None or days_left <= 30):
                severity = round(completion, 1)  # higher = closer to goal
                alerts.append({
                    "campaign_id":     cid,
                    "title":          title,
                    "completion_rate": round(completion, 1),
                    "days_left":      days_left,
                    "gap_amount":     round(gap_amount, 0),
                    "total_donors":   total_don,
                    "alert_type":     "NEARLY THERE",
                    "alert_label":    f"{completion:.1f}% funded — Rs.{gap_amount:,.0f} needed to complete",
                    "severity":       severity,
                    "action":         "Final push — notify all donors of similar campaigns"
                })

        # Sort: DEADLINE CRITICAL first, then by severity desc
        priority_order = {"DEADLINE CRITICAL": 0, "STALLED": 1, "LOW MOMENTUM": 2, "NEARLY THERE": 3}
        alerts.sort(key=lambda x: (priority_order.get(x["alert_type"], 9), -x["severity"]))
        return jsonify(alerts)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ============================================================
#  DONATE  (with fraud detection)
# ============================================================

@app.route('/donate', methods=['POST'])
@role_required("donor")
def donate():
    try:
        data        = request.get_json()
        donor_id    = session['user_id']
        campaign_id = data['campaign_id']
        amount      = float(data['amount'])
        pay_method  = data.get('payment_method', 'online')
        txn_id      = data.get('transaction_id', None)

        cur = mysql.connection.cursor()

        # ── Fetch campaign ─────────────────────────────────
        cur.execute("""
            SELECT target_amount, urgency_level, severity_score, collected_amount, donation_cap
            FROM campaigns WHERE id = %s
        """, (campaign_id,))
        campaign = cur.fetchone()
        if not campaign:
            cur.close()
            return jsonify({"status": "error", "message": "Campaign not found"})

        target_amount     = float(campaign[0])
        current_collected = float(campaign[3])
        donation_cap      = float(campaign[4]) if campaign[4] else None

        # ── Check if donor is blocked ──────────────────────
        cur.execute("SELECT is_blocked, block_reason FROM users WHERE id=%s", (donor_id,))
        donor_row = cur.fetchone()
        if donor_row and donor_row[0]:
            cur.close()
            return jsonify({
                "status":  "blocked",
                "message": "Your account has been restricted. Please contact support.",
                "reason":  donor_row[1] or "Account blocked"
            }), 403

        # ── Donation cap check ─────────────────────────────
        if donation_cap and (current_collected + amount) > donation_cap:
            remaining = max(0, donation_cap - current_collected)
            cur.close()
            return jsonify({
                "status":  "error",
                "message": f"This campaign has a funding cap of Rs.{donation_cap:,.0f}. "
                           f"Only Rs.{remaining:,.0f} more can be accepted."
            }), 400

        # ======================================================
        #  AI FRAUD DETECTION - Combined Scoring System
        #  Score 0-100: critical(>=70)/high(>=50)=block
        #  medium(>=30)=log only, low(<30)=allow silently
        # ======================================================
        fraud_score   = 0
        fraud_reasons = []
        ai_meta       = {}

        feature_vec, feat_info = ai.build_features(amount, donor_id, cur)

        # Hard override: exact duplicate 90s (score=100 immediately)
        cur.execute("""
            SELECT COUNT(*) FROM donations
            WHERE donor_id=%s AND campaign_id=%s AND amount=%s
              AND donation_date >= NOW() - INTERVAL 90 SECOND
              AND status != 'fraud'
        """, (donor_id, campaign_id, amount))
        if cur.fetchone()[0] > 0:
            fraud_score = 100
            fraud_reasons.append("Exact duplicate: identical transaction within 90 seconds")

        if fraud_score < 100:
            # Rule: new account +20
            if feat_info['is_new_account']:
                fraud_score += 20
                fraud_reasons.append(
                    f"New account: {feat_info['account_age_days']}d old (+20)"
                )

            # Rule: high velocity in rolling 24h window +25
            if feat_info['donations_today'] >= 5:
                fraud_score += 25
                fraud_reasons.append(
                    f"High velocity: {feat_info['donations_today']} donations in last 24h (+25)"
                )

            # Rule: unusual amount (only if >= 3 prior donations) +20
            if feat_info['total_donations'] >= 3 and feat_info['amount_ratio'] >= 10:
                fraud_score += 20
                fraud_reasons.append(
                    f"Unusual amount: Rs.{amount:,.0f} is {feat_info['amount_ratio']:.1f}x "
                    f"donor avg (need >=3 donations; has {feat_info['total_donations']}) (+20)"
                )

            # Rule: large 24h total +10
            if feat_info.get('amount_24h_total', 0) > 50000:
                fraud_score += 10
                fraud_reasons.append(
                    f"Large 24h total: Rs.{feat_info.get('amount_24h_total',0):,.0f} (+10)"
                )

            # ML: IsolationForest +15
            anomaly_risk = ai.anomaly_score(feature_vec)
            ai_meta['anomaly_risk'] = anomaly_risk
            if anomaly_risk >= 70:
                fraud_score += 15
                fraud_reasons.append(f"AI Anomaly Detector: {anomaly_risk:.1f}/100 (+15)")

            # ML: RandomForest +20
            fraud_prob = ai.fraud_probability(feature_vec)
            ai_meta['fraud_probability'] = fraud_prob
            if fraud_prob >= 0.75:
                fraud_score += 20
                fraud_reasons.append(
                    f"AI Classifier: {fraud_prob*100:.1f}% fraud probability (+20)"
                )

        # Risk level
        if   fraud_score >= 70: risk_level = 'critical'
        elif fraud_score >= 50: risk_level = 'high'
        elif fraud_score >= 30: risk_level = 'medium'
        else:                   risk_level = 'low'
        ai_meta['fraud_score'] = fraud_score
        ai_meta['risk_level']  = risk_level

        # Block on critical or high
        if risk_level in ('critical', 'high') and fraud_reasons:
            full_reason = (
                f"[Score:{fraud_score}/100 | {risk_level.upper()}] "
                + " | ".join(fraud_reasons)
            )
            cur.execute("""
                INSERT INTO fraud_logs (campaign_id, donor_id, reason, detected_at)
                VALUES (%s, %s, %s, NOW())
            """, (campaign_id, donor_id, full_reason))
            mysql.connection.commit()
            def _retrain():
                try:
                    with app.app_context():
                        _cur = mysql.connection.cursor()
                        ai.retrain(_cur)
                        _cur.close()
                except Exception:
                    pass
            threading.Thread(target=_retrain, daemon=True).start()
            cur.close()
            return jsonify({
                "status":    "fraud",
                "message":   "Transaction flagged by AI fraud detection. Please contact support.",
                "flags":     fraud_reasons,
                "ai_scores": ai_meta
            }), 403

        elif risk_level == 'medium' and fraud_reasons:
            try:
                cur.execute("""
                    INSERT INTO fraud_logs (campaign_id, donor_id, reason, detected_at)
                    VALUES (%s, %s, %s, NOW())
                """, (campaign_id, donor_id,
                       f"[Score:{fraud_score}/100 | MEDIUM-allowed] "
                       + " | ".join(fraud_reasons)))
                mysql.connection.commit()
            except Exception:
                pass

        # ======================================================
        #  END AI FRAUD DETECTION
        # ══════════════════════════════════════════════════════

        # ── Insert donation ────────────────────────────────
        # Simulated payment processing (15% random failure rate)
        # Replace with Razorpay/Stripe in production
        import random as _rnd
        if _rnd.random() < 0.15:
            cur.execute("""
                INSERT INTO donations
                    (donor_id, campaign_id, amount, payment_method, transaction_id, status)
                VALUES (%s, %s, %s, %s, %s, 'failed')
            """, (donor_id, campaign_id, amount, pay_method, txn_id))
            mysql.connection.commit()
            cur.close()
            return jsonify({
                "status":     "payment_failed",
                "message":    "Payment could not be processed. Please check your payment details and try again.",
                "suggestion": "Try a different payment method or retry after a few minutes."
            }), 402

        cur.execute("""
            INSERT INTO donations (donor_id, campaign_id, amount, payment_method, transaction_id)
            VALUES (%s, %s, %s, %s, %s)
        """, (donor_id, campaign_id, amount, pay_method, txn_id))

        donation_id   = cur.lastrowid
        new_collected = current_collected + amount
        completion_rate = (new_collected / target_amount * 100) if target_amount > 0 else 0

        # ── Update campaign totals + recalculate AI priority ──
        cur.execute(
            "SELECT urgency_level, severity_score, deadline FROM campaigns WHERE id=%s",
            (campaign_id,)
        )
        camp_meta    = cur.fetchone()
        new_priority = TrustFlowAI.compute_priority_score(
            urgency         = int(camp_meta[0]) if camp_meta else 5,
            severity        = int(camp_meta[1]) if camp_meta else 5,
            completion_rate = completion_rate,
            deadline        = camp_meta[2] if camp_meta else None
        )

        cur.execute("""
            UPDATE campaigns
            SET collected_amount = %s,
                total_donors     = total_donors + 1,
                completion_rate  = %s,
                priority_score   = %s
            WHERE id = %s
        """, (new_collected, completion_rate, new_priority, campaign_id))

        # ── Check and celebrate milestones ────────────────
        try:
            cur.execute("""
                SELECT id, amount, title FROM campaign_milestones
                WHERE campaign_id=%s AND reached=0 AND amount <= %s
            """, (campaign_id, new_collected))
            newly_reached = cur.fetchall()
            for ms in newly_reached:
                cur.execute("""
                    UPDATE campaign_milestones SET reached=1, reached_at=NOW() WHERE id=%s
                """, (ms[0],))
                # Celebrate with notification to all donors of this campaign
                cur.execute("""
                    SELECT DISTINCT donor_id FROM donations
                    WHERE campaign_id=%s AND status='success'
                """, (campaign_id,))
                all_donors = cur.fetchall()
                for d_row in all_donors:
                    cur.execute("""
                        INSERT INTO notifications (user_id, message, type)
                        VALUES (%s, %s, 'milestone')
                    """, (d_row[0],
                          f"🎉 Milestone reached on a campaign you support: '{ms[2]}' "
                          f"(Rs.{float(ms[1]):,.0f} funded)!"))
        except Exception:
            pass

        # ── Auto-close if cap reached ──────────────────────
        try:
            if donation_cap and new_collected >= donation_cap:
                cur.execute("UPDATE campaigns SET status='completed' WHERE id=%s", (campaign_id,))
                cur.execute("""
                    SELECT n.user_id FROM campaigns c JOIN ngos n ON c.ngo_id=n.id
                    WHERE c.id=%s
                """, (campaign_id,))
                ngo_u = cur.fetchone()
                if ngo_u:
                    cur.execute("""
                        INSERT INTO notifications (user_id, message, type)
                        VALUES (%s, %s, 'success')
                    """, (ngo_u[0],
                          f"Your campaign has reached its funding cap and has been marked complete!"))
        except Exception:
            pass

        # ── Notification ───────────────────────────────────
        try:
            cur.execute("""
                INSERT INTO notifications (user_id, message, type)
                VALUES (%s, %s, 'success')
            """, (donor_id, f"Your donation of ₹{amount:,.0f} was successful!"))
        except Exception:
            pass

        # ── Audit log ──────────────────────────────────────
        write_audit_log(cur, 'donations', donation_id, 'insert',
                        new_value=f"amount={amount}, campaign={campaign_id}")

        mysql.connection.commit()
        cur.close()

        return jsonify({
            "status":                "success",
            "message":               "Donation successful!",
            "donation_id":           donation_id,
            "updated_completion_rate": round(completion_rate, 1)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ============================================================
#  RECEIPT
# ============================================================

@app.route('/generate_receipt/<int:donation_id>', methods=['GET'])
def generate_receipt(donation_id):
    """
    Branded donation receipt PDF.
    Simple but professional — teal header, clean layout, gold accents.
    """
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT d.id, d.amount, d.donation_date, d.payment_method,
                   d.transaction_id,
                   u.name  AS donor_name,
                   u.email AS donor_email,
                   c.title AS campaign_title,
                   n_u.name AS ngo_name
            FROM donations d
            JOIN users    u   ON d.donor_id    = u.id
            JOIN campaigns c  ON d.campaign_id = c.id
            JOIN ngos     n   ON c.ngo_id       = n.id
            JOIN users    n_u ON n.user_id       = n_u.id
            WHERE d.id = %s
        """, (donation_id,))
        row = cur.fetchone()
        cur.close()

        if not row:
            return jsonify({"status": "error", "message": "Donation not found"})

        did, amount, date, pay_method, txn_id, donor_name, donor_email, campaign_title, ngo_name = row

        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas as pdfcanvas
        import io

        TEAL       = (0.05, 0.23, 0.28)
        TEAL_LIGHT = (0.88, 0.93, 0.95)
        GOLD       = (0.80, 0.62, 0.10)
        GOLD_LIGHT = (0.93, 0.82, 0.40)
        DARK       = (0.12, 0.14, 0.16)
        GREY       = (0.45, 0.48, 0.52)
        LIGHT_GREY = (0.92, 0.93, 0.94)
        WHITE      = (1, 1, 1)

        def sf(cv, rgb): cv.setFillColorRGB(*rgb)
        def ss(cv, rgb): cv.setStrokeColorRGB(*rgb)

        buf = io.BytesIO()
        W, H = A4
        cv = pdfcanvas.Canvas(buf, pagesize=A4)

        # ── White background ──────────────────────────────
        sf(cv, WHITE); cv.rect(0,0,W,H,fill=1,stroke=0)

        # ── Teal header band ──────────────────────────────
        sf(cv, TEAL); cv.rect(0, H-110, W, 110, fill=1, stroke=0)

        # Gold accent line at bottom of header
        ss(cv, GOLD); cv.setLineWidth(2.5)
        cv.line(0, H-110, W, H-110)

        # TrustFlow wordmark in header
        sf(cv, WHITE)
        cv.setFont("Times-Bold", 26)
        cv.drawString(48, H-62, "TrustFlow")
        sf(cv, GOLD_LIGHT)
        cv.setFont("Times-Italic", 9.5)
        cv.drawString(48, H-76, "Transparent Donation Platform")

        # RECEIPT label right-aligned in header
        sf(cv, GOLD_LIGHT)
        cv.setFont("Times-Bold", 22)
        cv.drawRightString(W-48, H-60, "RECEIPT")
        sf(cv, (0.72, 0.76, 0.70))
        cv.setFont("Times-Roman", 9)
        cv.drawRightString(W-48, H-75, f"No. TF-{did:06d}")

        # ── Donor info block ──────────────────────────────
        sf(cv, DARK); cv.setFont("Helvetica-Bold", 10)
        cv.drawString(48, H-145, "ISSUED TO")
        sf(cv, DARK); cv.setFont("Times-Bold", 15)
        dn = donor_name if len(donor_name)<=45 else donor_name[:42]+"..."
        cv.drawString(48, H-165, dn)
        sf(cv, GREY); cv.setFont("Times-Roman", 10)
        cv.drawString(48, H-180, donor_email or "")

        # Date top right
        sf(cv, GREY); cv.setFont("Helvetica-Bold", 9)
        cv.drawRightString(W-48, H-145, "DATE")
        sf(cv, DARK); cv.setFont("Times-Roman", 12)
        cv.drawRightString(W-48, H-162, str(date)[:10])

        # Thin divider
        ss(cv, LIGHT_GREY); cv.setLineWidth(0.8)
        cv.line(48, H-196, W-48, H-196)

        # ── Amount highlight box ──────────────────────────
        sf(cv, TEAL_LIGHT)
        cv.roundRect(48, H-268, W-96, 58, 6, fill=1, stroke=0)
        ss(cv, TEAL); cv.setLineWidth(0.8)
        cv.roundRect(48, H-268, W-96, 58, 6, fill=0, stroke=1)

        sf(cv, GREY); cv.setFont("Helvetica-Bold", 9)
        cv.drawString(62, H-222, "AMOUNT DONATED")
        sf(cv, TEAL); cv.setFont("Times-Bold", 30)
        cv.drawString(62, H-253, f"Rs. {float(amount):,.2f}")

        # ── Details table ─────────────────────────────────
        rows_data = [
            ("Campaign",       campaign_title if len(campaign_title)<=55 else campaign_title[:52]+"..."),
            ("Supported NGO",  ngo_name if len(ngo_name)<=45 else ngo_name[:42]+"..."),
            ("Payment Method", (pay_method or "online").title()),
            ("Transaction ID", txn_id or "—"),
            ("Status",         "Successful"),
        ]

        y = H-300
        for label, value in rows_data:
            # Light alternating row background
            if rows_data.index((label,value)) % 2 == 0:
                sf(cv, (0.97, 0.97, 0.97))
                cv.rect(48, y-6, W-96, 22, fill=1, stroke=0)

            sf(cv, GREY); cv.setFont("Helvetica-Bold", 8.5)
            cv.drawString(58, y+5, label.upper())
            sf(cv, DARK); cv.setFont("Times-Roman", 10.5)
            cv.drawString(185, y+5, value)

            # Row divider
            ss(cv, (0.90, 0.90, 0.90)); cv.setLineWidth(0.4)
            cv.line(48, y-6, W-48, y-6)
            y -= 26

        # Status row green highlight
        sf(cv, (0.86, 0.97, 0.90))
        cv.rect(48, y+18, W-96, 22, fill=1, stroke=0)
        sf(cv, (0.06, 0.52, 0.25)); cv.setFont("Helvetica-Bold", 8.5)
        cv.drawString(58, y+23, "STATUS")
        cv.setFont("Times-Bold", 10.5)
        cv.drawString(185, y+23, "✓  Payment Successful")

        # ── Thank you message ─────────────────────────────
        ty = y - 40
        sf(cv, TEAL)
        cv.setFont("Times-BoldItalic", 13)
        cv.drawCentredString(W/2, ty,
            "Thank you for your generosity. Every contribution makes a difference.")

        ss(cv, GOLD); cv.setLineWidth(0.8)
        cv.line(100, ty-10, W-100, ty-10)

        # ── Footer ────────────────────────────────────────
        sf(cv, TEAL); cv.rect(0, 0, W, 46, fill=1, stroke=0)
        ss(cv, GOLD); cv.setLineWidth(1.0)
        cv.line(0, 46, W, 46)

        sf(cv, (0.72, 0.76, 0.70))
        cv.setFont("Times-Italic", 8.5)
        cv.drawCentredString(W/2, 28,
            "TrustFlow Platform  ·  trustflow.in  ·  Computer-generated receipt, valid without physical signature")
        sf(cv, GOLD_LIGHT)
        cv.setFont("Times-Roman", 8)
        cv.drawCentredString(W/2, 14, f"Receipt ID: TF-{did:06d}  ·  {str(date)[:10]}")

        cv.save()
        buf.seek(0)
        return send_file(buf, as_attachment=True,
                         download_name=f"TrustFlow_Receipt_{did}.pdf",
                         mimetype='application/pdf')
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ============================================================
#  DONOR ENDPOINTS
# ============================================================

@app.route('/donation_history/<int:donor_id>', methods=['GET'])
def donation_history(donor_id):
    if 'user_id' not in session or session['user_id'] != donor_id:
        return jsonify({"status": "error", "message": "Unauthorized"})
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT d.id, d.amount, d.donation_date, d.status,
                   d.payment_method, c.title, c.id
            FROM donations d
            JOIN campaigns c ON d.campaign_id = c.id
            WHERE d.donor_id = %s
            ORDER BY d.donation_date DESC
        """, (donor_id,))
        rows = cur.fetchall()
        cur.close()
        return jsonify([{
            "id":             r[0],
            "amount":         float(r[1]) if r[1] else 0,
            "date":           str(r[2]),
            "status":         r[3] or "success",
            "payment_method": r[4] or "online",
            "campaign":       r[5] or "Unknown",
            "campaign_id":    r[6]
        } for r in rows])
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/donor_stats/<int:donor_id>', methods=['GET'])
def donor_stats(donor_id):
    if 'user_id' not in session or session['user_id'] != donor_id:
        return jsonify({"status": "error", "message": "Unauthorized"})
    try:
        cur = mysql.connection.cursor()

        cur.execute("SELECT COALESCE(SUM(amount),0) FROM donations WHERE donor_id=%s", (donor_id,))
        total = float(cur.fetchone()[0])

        cur.execute("SELECT COUNT(DISTINCT campaign_id) FROM donations WHERE donor_id=%s", (donor_id,))
        active = cur.fetchone()[0]

        cur.execute("""
            SELECT donation_date, amount FROM donations
            WHERE donor_id=%s ORDER BY donation_date DESC LIMIT 1
        """, (donor_id,))
        last = cur.fetchone()
        cur.close()

        return jsonify({
            "total":                total,
            "active_causes":        active,
            "last_donation_date":   str(last[0]) if last else None,
            "last_donation_amount": float(last[1]) if last else 0
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ── Impact tracking ────────────────────────────────────────

@app.route('/donor_impact/<int:donor_id>/<int:campaign_id>', methods=['GET'])
def get_donor_impact(donor_id, campaign_id):
    if 'user_id' not in session or session['user_id'] != donor_id:
        return jsonify({"status": "error", "message": "Unauthorized"})
    try:
        cur = mysql.connection.cursor()

        cur.execute("""
            SELECT COALESCE(SUM(amount), 0)
            FROM donations WHERE donor_id=%s AND campaign_id=%s
        """, (donor_id, campaign_id))
        donor_amount = float(cur.fetchone()[0])

        if donor_amount == 0:
            cur.close()
            return jsonify({"status": "error", "message": "No donation found"})

        cur.execute(
            "SELECT title, target_amount, collected_amount FROM campaigns WHERE id=%s",
            (campaign_id,)
        )
        campaign = cur.fetchone()
        campaign_title = campaign[0] or "Campaign"
        target    = float(campaign[1]) if campaign[1] else 0
        collected = float(campaign[2]) if campaign[2] else 0

        try:
            cur.execute("""
                SELECT title, amount, description, expense_date
                FROM campaign_expenses WHERE campaign_id=%s
                ORDER BY expense_date DESC
            """, (campaign_id,))
            expenses = cur.fetchall()
        except Exception:
            expenses = []

        contribution_pct = (donor_amount / collected * 100) if collected > 0 else 0
        total_expenses   = 0
        impact_breakdown = []

        for exp in expenses:
            exp_amount = float(exp[1]) if exp[1] else 0
            total_expenses += exp_amount
            donor_share = exp_amount * contribution_pct / 100
            impact_breakdown.append({
                "title":                exp[0] or "Expense",
                "total_amount":         exp_amount,
                "donor_share":          round(donor_share, 2),
                "percentage_of_donation": round(donor_share / donor_amount * 100, 1) if donor_amount else 0,
                "date":                 str(exp[3])
            })

        used_amount   = total_expenses * contribution_pct / 100
        unused_amount = donor_amount - used_amount
        completion_rate = (collected / target * 100) if target > 0 else 0
        cur.close()

        return jsonify({
            "status":               "success",
            "donor_amount":         donor_amount,
            "campaign_title":       campaign_title,
            "campaign_progress":    round(completion_rate, 1),
            "total_collected":      collected,
            "total_expenses":       total_expenses,
            "contribution_percentage": round(contribution_pct, 1),
            "impact_breakdown":     impact_breakdown,
            "used_amount":          round(used_amount, 2),
            "unused_amount":        round(unused_amount, 2),
            "message": f"Your ₹{donor_amount:,.0f} helped fund {len(impact_breakdown)} expense items"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/donor_overall_impact/<int:donor_id>', methods=['GET'])
def get_donor_overall_impact(donor_id):
    if 'user_id' not in session or session['user_id'] != donor_id:
        return jsonify({"status": "error", "message": "Unauthorized"})
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT DISTINCT c.id, c.title
            FROM donations d JOIN campaigns c ON d.campaign_id=c.id
            WHERE d.donor_id=%s
        """, (donor_id,))
        campaigns_list = cur.fetchall()

        cur.execute(
            "SELECT COALESCE(SUM(amount),0) FROM donations WHERE donor_id=%s",
            (donor_id,)
        )
        total_donated = float(cur.fetchone()[0])
        cur.close()

        return jsonify({
            "status":             "success",
            "campaigns_supported": len(campaigns_list),
            "beneficiaries_helped": len(campaigns_list) * 2,
            "updates_received":   len(campaigns_list),
            "total_donated":      total_donated
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ============================================================
#  NGO ENDPOINTS
# ============================================================

@app.route('/create_campaign', methods=['POST'])
@role_required("ngo")
def create_campaign():
    try:
        data = request.get_json()
        cur  = mysql.connection.cursor()

        cur.execute("SELECT id FROM ngos WHERE user_id=%s", (session['user_id'],))
        ngo = cur.fetchone()
        if not ngo:
            cur.close()
            return jsonify({"status": "error", "message": "NGO not found"})

        ngo_id = ngo[0]
        # Compute AI priority score at creation time
        initial_priority = TrustFlowAI.compute_priority_score(
            urgency      = int(data.get('urgency_level', 5)),
            severity     = int(data.get('severity_score', 5)),
            completion_rate = 0,
            deadline     = data.get('deadline')
        )

        cur.execute("""
            INSERT INTO campaigns
                (ngo_id, title, description, category, target_amount,
                 urgency_level, severity_score, deadline, status, priority_score)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'pending',%s)
        """, (
            ngo_id,
            data['title'],
            data['description'],
            data['category'],
            data['target_amount'],
            data['urgency_level'],
            data['severity_score'],
            data['deadline'],
            initial_priority
        ))
        mysql.connection.commit()

        campaign_id = cur.lastrowid
        write_audit_log(cur, 'campaigns', campaign_id, 'insert',
                        new_value=f"title={data['title']}, ngo_id={ngo_id}")
        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": "Campaign created successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/get_ngo_stats', methods=['GET'])
@role_required("ngo")
def get_ngo_stats():
    try:
        cur = mysql.connection.cursor()
        # FIX: trust_badge column added safely with COALESCE fallback
        cur.execute("""
            SELECT id, document_score, update_score, is_verified,
                   COALESCE(trust_badge, 'bronze') AS trust_badge
            FROM ngos WHERE user_id=%s
        """, (session['user_id'],))
        ngo = cur.fetchone()
        if not ngo:
            cur.close()
            return jsonify({"status": "error", "message": "NGO not found"})

        ngo_id          = ngo[0]
        document_score  = float(ngo[1]) if ngo[1] else 0
        update_score    = float(ngo[2]) if ngo[2] else 0
        is_verified     = bool(ngo[3])
        trust_badge     = ngo[4]

        cur.execute("""
            SELECT COUNT(*), COALESCE(SUM(collected_amount),0), COALESCE(AVG(completion_rate),0)
            FROM campaigns WHERE ngo_id=%s
        """, (ngo_id,))
        stats = cur.fetchone()
        total_campaigns = stats[0] or 0
        total_raised    = float(stats[1]) or 0
        avg_completion  = float(stats[2]) or 0

        verification_bonus = 20 if is_verified else 0
        transparency_score = (
            document_score * 0.3 +
            update_score   * 0.3 +
            avg_completion * 0.2 +
            verification_bonus
        )
        cur.close()

        return jsonify({
            "status":            "success",
            "total_campaigns":   total_campaigns,
            "total_raised":      total_raised,
            "avg_completion":    round(avg_completion, 1),
            "transparency_score": round(transparency_score, 1),
            "is_verified":       is_verified,
            "document_score":    document_score,
            "update_score":      update_score,
            "trust_badge":       trust_badge
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/get_ngo_campaigns', methods=['GET'])
@role_required("ngo")
def get_ngo_campaigns():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT id FROM ngos WHERE user_id=%s", (session['user_id'],))
        ngo = cur.fetchone()
        if not ngo:
            cur.close()
            return jsonify({"status": "error", "message": "NGO not found"})

        cur.execute("""
            SELECT id, title, description, category, target_amount,
                   collected_amount, completion_rate, priority_score, status, deadline
            FROM campaigns WHERE ngo_id=%s ORDER BY id DESC
        """, (ngo[0],))
        rows = cur.fetchall()
        cur.close()
        return jsonify([{
            "id":               r[0],
            "title":            r[1] or "Untitled",
            "description":      r[2] or "",
            "category":         r[3] or "General",
            "target_amount":    float(r[4]) if r[4] else 0,
            "collected_amount": float(r[5]) if r[5] else 0,
            "completion_rate":  float(r[6]) if r[6] else 0,
            "priority_score":   float(r[7]) if r[7] else 0,
            "status":           r[8] or "pending",
            "deadline":         str(r[9]) if r[9] else None
        } for r in rows])
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/add_expense', methods=['POST'])
@role_required("ngo")
def add_expense():
    try:
        data = request.get_json()
        cur  = mysql.connection.cursor()

        cur.execute("""
            SELECT c.id FROM campaigns c JOIN ngos n ON c.ngo_id=n.id
            WHERE c.id=%s AND n.user_id=%s
        """, (data['campaign_id'], session['user_id']))
        if not cur.fetchone():
            cur.close()
            return jsonify({"status": "error", "message": "Unauthorized"})

        cur.execute("""
            INSERT INTO campaign_expenses
                (campaign_id, title, amount, description, expense_date, bill_image)
            VALUES (%s,%s,%s,%s,%s,%s)
        """, (
            data['campaign_id'],
            data['title'],
            data['amount'],
            data.get('description', ''),
            data['expense_date'],
            data.get('bill_image')
        ))
        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": "Expense added successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/add_campaign_update', methods=['POST'])
@role_required("ngo")
def add_campaign_update():
    try:
        data = request.get_json()
        cur  = mysql.connection.cursor()

        cur.execute("""
            SELECT c.id FROM campaigns c JOIN ngos n ON c.ngo_id=n.id
            WHERE c.id=%s AND n.user_id=%s
        """, (data['campaign_id'], session['user_id']))
        if not cur.fetchone():
            cur.close()
            return jsonify({"status": "error", "message": "Unauthorized"})

        cur.execute("""
            INSERT INTO campaign_updates
                (campaign_id, update_text, proof_document)
            VALUES (%s, %s, %s)
        """, (
            data['campaign_id'],
            data.get('update_text', data.get('description', '')),
            data.get('proof_document', data.get('media_url'))
        ))

        # Boost NGO update score
        cur.execute(
            "UPDATE ngos SET update_score = update_score + 5 WHERE user_id=%s",
            (session['user_id'],)
        )
        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": "Update posted successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/campaign_donors/<int:campaign_id>', methods=['GET'])
@role_required("ngo")
def campaign_donors(campaign_id):
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT u.name, d.amount, d.donation_date, d.payment_method
            FROM donations d JOIN users u ON d.donor_id=u.id
            WHERE d.campaign_id=%s
            ORDER BY d.donation_date DESC
        """, (campaign_id,))
        rows = cur.fetchall()
        cur.close()
        return jsonify([{
            "donor_name":     r[0] or "Anonymous",
            "amount":         float(r[1]) if r[1] else 0,
            "date":           str(r[2]),
            "payment_method": r[3] or "online"
        } for r in rows])
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ============================================================
#  NGO — MISSING ROUTES  (added)
# ============================================================

# ── Get single campaign (for edit modal) ──────────────────
@app.route('/get_campaign/<int:campaign_id>', methods=['GET'])
@role_required("ngo")
def get_campaign(campaign_id):
    """Return one campaign owned by this NGO."""
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT c.id, c.title, c.description, c.category,
                   c.target_amount, c.urgency_level, c.severity_score,
                   c.deadline, c.status
            FROM campaigns c
            JOIN ngos n ON c.ngo_id = n.id
            WHERE c.id = %s AND n.user_id = %s
        """, (campaign_id, session['user_id']))
        row = cur.fetchone()
        cur.close()
        if not row:
            return jsonify({"status": "error", "message": "Campaign not found or unauthorized"})
        return jsonify({
            "id":            row[0],
            "title":         row[1] or "",
            "description":   row[2] or "",
            "category":      row[3] or "General",
            "target_amount": float(row[4]) if row[4] else 0,
            "urgency_level": row[5] or 5,
            "severity_score":row[6] or 5,
            "deadline":      str(row[7]) if row[7] else "",
            "status":        row[8] or "pending"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ── Update campaign ───────────────────────────────────────
@app.route('/update_campaign/<int:campaign_id>', methods=['PUT'])
@role_required("ngo")
def update_campaign(campaign_id):
    """Edit title, description, and target of an owned campaign."""
    try:
        data = request.get_json()
        cur  = mysql.connection.cursor()

        # Verify ownership
        cur.execute("""
            SELECT c.id, c.title FROM campaigns c
            JOIN ngos n ON c.ngo_id = n.id
            WHERE c.id = %s AND n.user_id = %s
        """, (campaign_id, session['user_id']))
        existing = cur.fetchone()
        if not existing:
            cur.close()
            return jsonify({"status": "error", "message": "Campaign not found or unauthorized"})

        title         = data.get('title', existing[1])
        description   = data.get('description', '')
        target_amount = float(data.get('target_amount', 0)) if data.get('target_amount') else None

        if target_amount:
            cur.execute("""
                UPDATE campaigns
                SET title=%s, description=%s, target_amount=%s
                WHERE id=%s
            """, (title, description, target_amount, campaign_id))
        else:
            cur.execute("""
                UPDATE campaigns
                SET title=%s, description=%s
                WHERE id=%s
            """, (title, description, campaign_id))

        write_audit_log(cur, 'campaigns', campaign_id, 'update',
                        old_value=f"title={existing[1]}",
                        new_value=f"title={title}")
        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": "Campaign updated successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ── Delete campaign ───────────────────────────────────────
@app.route('/delete_campaign/<int:campaign_id>', methods=['DELETE'])
@role_required("ngo")
def delete_campaign(campaign_id):
    """Soft-delete: set status to 'rejected'. Hard delete only if no donations."""
    try:
        cur = mysql.connection.cursor()

        # Verify ownership
        cur.execute("""
            SELECT c.id FROM campaigns c
            JOIN ngos n ON c.ngo_id = n.id
            WHERE c.id = %s AND n.user_id = %s
        """, (campaign_id, session['user_id']))
        if not cur.fetchone():
            cur.close()
            return jsonify({"status": "error", "message": "Campaign not found or unauthorized"})

        # Check if any donations exist — if so, soft-delete only
        cur.execute("SELECT COUNT(*) FROM donations WHERE campaign_id=%s", (campaign_id,))
        donation_count = cur.fetchone()[0]

        if donation_count > 0:
            # Soft delete — mark as rejected so donors can still see history
            cur.execute("UPDATE campaigns SET status='rejected' WHERE id=%s", (campaign_id,))
            write_audit_log(cur, 'campaigns', campaign_id, 'update',
                            new_value="status=rejected (soft-delete, has donations)")
            mysql.connection.commit()
            cur.close()
            return jsonify({
                "status":  "success",
                "message": "Campaign deactivated (soft deleted — it had donations so records are preserved)"
            })
        else:
            # Hard delete — no donations, safe to remove
            cur.execute("DELETE FROM campaigns WHERE id=%s", (campaign_id,))
            write_audit_log(cur, 'campaigns', campaign_id, 'delete')
            mysql.connection.commit()
            cur.close()
            return jsonify({"status": "success", "message": "Campaign deleted successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ── Add beneficiary ───────────────────────────────────────
@app.route('/add_beneficiary', methods=['POST'])
@role_required("ngo")
def add_beneficiary():
    """Add a beneficiary story to a campaign."""
    try:
        data = request.get_json()
        cur  = mysql.connection.cursor()

        # Verify NGO owns this campaign
        cur.execute("""
            SELECT c.id FROM campaigns c
            JOIN ngos n ON c.ngo_id = n.id
            WHERE c.id = %s AND n.user_id = %s
        """, (data['campaign_id'], session['user_id']))
        if not cur.fetchone():
            cur.close()
            return jsonify({"status": "error", "message": "Unauthorized"})

        cur.execute("""
            INSERT INTO beneficiaries
                (campaign_id, name, story, image, helped_date)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            data['campaign_id'],
            data['name'],
            data['story'],
            data.get('image'),
            data.get('helped_date')
        ))
        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": "Beneficiary story added successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ── Get beneficiaries ─────────────────────────────────────
@app.route('/get_beneficiaries/<int:campaign_id>', methods=['GET'])
def get_beneficiaries(campaign_id):
    """Get all beneficiary stories for a campaign (public)."""
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT id, name, story, image, helped_date, created_at
            FROM beneficiaries
            WHERE campaign_id = %s
            ORDER BY helped_date DESC
        """, (campaign_id,))
        rows = cur.fetchall()
        cur.close()
        return jsonify([{
            "id":         r[0],
            "name":       r[1] or "",
            "story":      r[2] or "",
            "image":      r[3],
            "helped_date":str(r[4]) if r[4] else None,
            "created_at": str(r[5])
        } for r in rows])
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ── Add fund allocation ───────────────────────────────────
@app.route('/add_fund_allocation', methods=['POST'])
@role_required("ngo")
def add_fund_allocation():
    """Record how raised funds are being allocated per category."""
    try:
        data = request.get_json()
        cur  = mysql.connection.cursor()

        # Verify ownership
        cur.execute("""
            SELECT c.id, c.collected_amount FROM campaigns c
            JOIN ngos n ON c.ngo_id = n.id
            WHERE c.id = %s AND n.user_id = %s
        """, (data['campaign_id'], session['user_id']))
        campaign = cur.fetchone()
        if not campaign:
            cur.close()
            return jsonify({"status": "error", "message": "Unauthorized"})

        # Validate amount does not exceed collected
        alloc_amount = float(data['amount'])
        cur.execute("""
            SELECT COALESCE(SUM(amount), 0)
            FROM fund_allocation WHERE campaign_id = %s
        """, (data['campaign_id'],))
        already_allocated = float(cur.fetchone()[0])
        collected         = float(campaign[1]) if campaign[1] else 0

        if already_allocated + alloc_amount > collected:
            cur.close()
            return jsonify({
                "status":  "error",
                "message": f"Allocation exceeds collected amount. Available: ₹{collected - already_allocated:,.0f}"
            })

        cur.execute("""
            INSERT INTO fund_allocation
                (campaign_id, category, amount, description)
            VALUES (%s, %s, %s, %s)
        """, (
            data['campaign_id'],
            data['category'],
            alloc_amount,
            data.get('description', '')
        ))
        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": "Fund allocation recorded successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ── Get fund allocations ──────────────────────────────────
@app.route('/get_fund_allocations/<int:campaign_id>', methods=['GET'])
def get_fund_allocations(campaign_id):
    """Get all fund allocations for a campaign (public)."""
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT id, category, amount, description, allocation_date
            FROM fund_allocation
            WHERE campaign_id = %s
            ORDER BY allocation_date DESC
        """, (campaign_id,))
        rows = cur.fetchall()
        cur.close()
        return jsonify([{
            "id":              r[0],
            "category":        r[1] or "",
            "amount":          float(r[2]) if r[2] else 0,
            "description":     r[3] or "",
            "allocation_date": str(r[4])
        } for r in rows])
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ── Donor year-end giving statement (PDF) ────────────────
@app.route('/donor_statement/<int:donor_id>/<int:year>', methods=['GET'])
def donor_statement(donor_id, year):
    """Generate a year-end PDF giving statement for a donor."""
    if 'user_id' not in session or session['user_id'] != donor_id:
        return jsonify({"status": "error", "message": "Unauthorized"})
    try:
        cur = mysql.connection.cursor()

        # Donor info
        cur.execute("SELECT name, email FROM users WHERE id=%s", (donor_id,))
        user = cur.fetchone()
        if not user:
            cur.close()
            return jsonify({"status": "error", "message": "User not found"})

        # All donations in that year
        cur.execute("""
            SELECT d.id, d.amount, d.donation_date, d.payment_method, c.title
            FROM donations d
            JOIN campaigns c ON d.campaign_id = c.id
            WHERE d.donor_id = %s
              AND YEAR(d.donation_date) = %s
              AND d.status = 'success'
            ORDER BY d.donation_date ASC
        """, (donor_id, year))
        donations = cur.fetchall()
        cur.close()

        # Block download if no donations exist for that year
        if not donations:
            return jsonify({
                "status":  "error",
                "message": f"No donations found for {year}. Nothing to download."
            }), 404

        total = sum(float(d[1]) for d in donations)

        # Build PDF
        buffer = io.BytesIO()
        doc    = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        elems  = []

        elems.append(Paragraph(f"<b>{APP_NAME} — Annual Giving Statement</b>", styles['Title']))
        elems.append(Spacer(1, 0.3 * inch))
        elems.append(Paragraph(f"Donor: {user[0]}", styles['Normal']))
        elems.append(Paragraph(f"Email: {user[1]}", styles['Normal']))
        elems.append(Paragraph(f"Year: {year}", styles['Normal']))
        elems.append(Spacer(1, 0.3 * inch))
        elems.append(Paragraph(f"<b>Total Donated in {year}: ₹{total:,.2f}</b>", styles['Normal']))
        elems.append(Spacer(1, 0.2 * inch))
        elems.append(Paragraph("<b>Donation Details:</b>", styles['Normal']))
        elems.append(Spacer(1, 0.1 * inch))

        for d in donations:
            elems.append(Paragraph(
                f"• {d[2].strftime('%d %b %Y')}  |  {d[4]}  |  ₹{float(d[1]):,.2f}  |  {d[3] or 'online'}",
                styles['Normal']
            ))

        elems.append(Spacer(1, 0.4 * inch))
        elems.append(Paragraph(
            "This statement is for your records. Thank you for supporting TrustFlow's mission.",
            styles['Normal']
        ))

        doc.build(elems)
        buffer.seek(0)
        return send_file(buffer, as_attachment=True,
                         download_name=f"giving_statement_{year}_{donor_id}.pdf",
                         mimetype='application/pdf')
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ============================================================
#  ADMIN ENDPOINTS  (all new)
# ============================================================

@app.route('/admin/stats', methods=['GET'])
@role_required("admin")
def admin_stats():
    """Overview numbers for the admin dashboard."""
    try:
        cur = mysql.connection.cursor()

        cur.execute("SELECT COUNT(*) FROM users WHERE role='donor'")
        total_donors = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM users WHERE role='ngo'")
        total_ngos = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*), COALESCE(SUM(amount),0) FROM donations WHERE status='success'")
        row = cur.fetchone()
        total_donations  = row[0]
        total_amount     = float(row[1])

        cur.execute("SELECT COUNT(*) FROM campaigns WHERE status='pending'")
        pending_campaigns = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM fraud_logs")
        total_fraud = cur.fetchone()[0]

        cur.close()
        return jsonify({
            "status":           "success",
            "total_donors":     total_donors,
            "total_ngos":       total_ngos,
            "total_donations":  total_donations,
            "total_amount":     round(total_amount, 2),
            "pending_campaigns": pending_campaigns,
            "total_fraud_flags": total_fraud
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/admin/users', methods=['GET'])
@role_required("admin")
def admin_get_users():
    """List all users. For NGO users also returns ngo_id, is_verified, trust_badge."""
    try:
        role_filter = request.args.get('role')
        cur = mysql.connection.cursor()
        # LEFT JOIN ngos so we get verification info for NGO users in one query
        base_query = """
            SELECT u.id, u.name, u.email, u.role, u.phone, u.created_at,
                   n.id        AS ngo_id,
                   n.is_verified,
                   COALESCE(n.trust_badge, 'bronze') AS trust_badge,
                   u.profile_photo
            FROM users u
            LEFT JOIN ngos n ON n.user_id = u.id
        """
        if role_filter:
            cur.execute(base_query + " WHERE u.role=%s ORDER BY u.created_at DESC", (role_filter,))
        else:
            cur.execute(base_query + " ORDER BY u.created_at DESC")
        rows = cur.fetchall()
        cur.close()
        return jsonify([{
            "id":          r[0],
            "name":        r[1] or "",
            "email":       r[2] or "",
            "role":        r[3],
            "phone":       r[4] or "",
            "created_at":  str(r[5]),
            "ngo_id":      r[6],           # None for non-NGO users
            "is_verified": bool(r[7]) if r[7] is not None else False,
            "trust_badge":    r[8] or "bronze",
            "profile_photo":  r[9]
        } for r in rows])
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/admin/campaigns', methods=['GET'])
@role_required("admin")
def admin_get_campaigns():
    """All campaigns with NGO name."""
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT c.id, c.title, c.category, c.target_amount,
                   c.collected_amount, c.completion_rate, c.status,
                   c.urgency_level, c.deadline, c.created_at,
                   u.name AS ngo_name
            FROM campaigns c
            LEFT JOIN ngos    n ON c.ngo_id = n.id
            LEFT JOIN users   u ON n.user_id = u.id
            ORDER BY c.created_at DESC
        """)
        rows = cur.fetchall()
        cur.close()
        return jsonify([{
            "id":               r[0],
            "title":            r[1] or "Untitled",
            "category":         r[2] or "General",
            "target_amount":    float(r[3]) if r[3] else 0,
            "collected_amount": float(r[4]) if r[4] else 0,
            "completion_rate":  float(r[5]) if r[5] else 0,
            "status":           r[6] or "pending",
            "urgency_level":    r[7] or 0,
            "deadline":         str(r[8]) if r[8] else None,
            "created_at":       str(r[9]),
            "ngo_name":         r[10] or "Unknown NGO"
        } for r in rows])
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/admin/campaign/<int:campaign_id>/status', methods=['POST'])
@role_required("admin")
def admin_update_campaign_status(campaign_id):
    """Approve or reject a campaign."""
    try:
        data   = request.get_json()
        status = data.get('status')
        if status not in ('approved', 'rejected', 'completed'):
            return jsonify({"status": "error", "message": "Invalid status value"})

        cur = mysql.connection.cursor()

        # Fetch old status for audit log
        cur.execute("SELECT status FROM campaigns WHERE id=%s", (campaign_id,))
        old = cur.fetchone()
        if not old:
            cur.close()
            return jsonify({"status": "error", "message": "Campaign not found"})

        cur.execute(
            "UPDATE campaigns SET status=%s WHERE id=%s",
            (status, campaign_id)
        )
        write_audit_log(cur, 'campaigns', campaign_id, 'update',
                        old_value=f"status={old[0]}", new_value=f"status={status}")

        # Notify the NGO
        cur.execute("SELECT ngo_id FROM campaigns WHERE id=%s", (campaign_id,))
        ngo_row = cur.fetchone()
        if ngo_row:
            cur.execute("SELECT user_id FROM ngos WHERE id=%s", (ngo_row[0],))
            ngo_user = cur.fetchone()
            if ngo_user:
                msg = f"Your campaign (ID {campaign_id}) has been {status} by admin."
                cur.execute(
                    "INSERT INTO notifications (user_id, message, type) VALUES (%s,%s,%s)",
                    (ngo_user[0], msg, 'info')
                )

        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": f"Campaign {status} successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/admin/ai_stats', methods=['GET'])
@role_required('admin')
def admin_ai_stats():
    """Returns current AI model status and key metrics for admin dashboard."""
    try:
        cur = mysql.connection.cursor()

        # Total fraud flags and reviewed count
        cur.execute("SELECT COUNT(*), SUM(reviewed) FROM fraud_logs")
        row = cur.fetchone()
        total_flags    = int(row[0] or 0)
        reviewed_flags = int(row[1] or 0)

        # Flags in last 7 days
        cur.execute("""
            SELECT COUNT(*) FROM fraud_logs
            WHERE detected_at >= NOW() - INTERVAL 7 DAY
        """)
        recent_flags = int(cur.fetchone()[0] or 0)

        # Average anomaly risk from recent flags (stored in reason string)
        # Count by type
        cur.execute("""
            SELECT
              SUM(reason LIKE '%Anomaly Detection%') as ai_anomaly,
              SUM(reason LIKE '%Classifier%')        as ai_classifier,
              SUM(reason LIKE '%duplicate%')         as hard_rule
            FROM fraud_logs
        """)
        type_row = cur.fetchone()

        # Campaign priority score distribution
        cur.execute("""
            SELECT
              AVG(priority_score),
              MAX(priority_score),
              SUM(priority_score = 0) as unscored
            FROM campaigns WHERE status IN ('approved','pending')
        """)
        p_row = cur.fetchone()

        cur.close()
        return jsonify({
            'status':           'success',
            'model_status': {
                'isolation_forest_trained': ai.is_trained,
                'random_forest_trained':    ai.rf_trained,
            },
            'fraud_detection': {
                'total_flags':     total_flags,
                'reviewed_flags':  reviewed_flags,
                'recent_7d':       recent_flags,
                'by_type': {
                    'ai_anomaly_detection': int(type_row[0] or 0),
                    'ai_classifier':        int(type_row[1] or 0),
                    'hard_rule_duplicate':  int(type_row[2] or 0)
                }
            },
            'priority_scoring': {
                'avg_score':   round(float(p_row[0] or 0), 1),
                'max_score':   round(float(p_row[1] or 0), 1),
                'unscored':    int(p_row[2] or 0)
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/admin/fraud_logs', methods=['GET'])
@role_required("admin")
def admin_get_fraud_logs():
    """All fraud log entries with donor and campaign details."""
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT fl.id, fl.reason, fl.detected_at,
                   COALESCE(u.name, 'Unknown')  AS donor_name,
                   COALESCE(c.title, 'Unknown') AS campaign_title,
                   fl.donor_id, fl.campaign_id,
                   u.profile_photo
            FROM fraud_logs fl
            LEFT JOIN users     u ON fl.donor_id    = u.id
            LEFT JOIN campaigns c ON fl.campaign_id = c.id
            ORDER BY fl.detected_at DESC
        """)
        rows = cur.fetchall()
        cur.close()
        return jsonify([{
            "id":             r[0],
            "reason":         r[1],
            "detected_at":    str(r[2]),
            "donor_name":     r[3],
            "campaign_title": r[4],
            "donor_id":       r[5],
            "campaign_id":    r[6],
            "profile_photo":  r[7]
        } for r in rows])
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/admin/fraud_logs/<int:log_id>/review', methods=['POST'])
@role_required("admin")
def admin_review_fraud_log(log_id):
    """
    Mark a fraud log entry as reviewed with an admin note.
    Requires fraud_logs to have a 'reviewed' tinyint and 'admin_note' text column.
    If those columns don't exist yet, run the SQL migration below.
    """
    try:
        data  = request.get_json()
        note  = data.get('note', '')
        cur   = mysql.connection.cursor()

        cur.execute("""
            UPDATE fraud_logs
            SET reviewed=1, admin_note=%s, reviewed_at=NOW(), reviewed_by=%s
            WHERE id=%s
        """, (note, session['user_id'], log_id))

        write_audit_log(cur, 'fraud_logs', log_id, 'update',
                        new_value=f"reviewed=1, note={note[:80]}")
        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": "Fraud log marked as reviewed"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/admin/verify_ngo/<int:ngo_id>', methods=['POST'])
@role_required("admin")
def admin_verify_ngo(ngo_id):
    """Toggle NGO verified status and update trust badge."""
    try:
        data       = request.get_json()
        is_verified = int(bool(data.get('is_verified', True)))
        badge      = data.get('trust_badge', 'silver')

        cur = mysql.connection.cursor()
        cur.execute("""
            UPDATE ngos SET is_verified=%s, trust_badge=%s WHERE id=%s
        """, (is_verified, badge, ngo_id))

        write_audit_log(cur, 'ngos', ngo_id, 'update',
                        new_value=f"is_verified={is_verified}, badge={badge}")
        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": "NGO verification updated"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/admin/all_donations', methods=['GET'])
@role_required("admin")
def admin_all_donations():
    """Full donation ledger with optional filters."""
    try:
        campaign_id = request.args.get('campaign_id')
        donor_id    = request.args.get('donor_id')
        status_f    = request.args.get('status')

        query  = """
            SELECT d.id, d.amount, d.donation_date, d.status, d.payment_method,
                   u.name AS donor_name, c.title AS campaign_title
            FROM donations d
            JOIN users     u ON d.donor_id    = u.id
            JOIN campaigns c ON d.campaign_id = c.id
            WHERE 1=1
        """
        params = []
        if campaign_id:
            query += " AND d.campaign_id=%s"; params.append(campaign_id)
        if donor_id:
            query += " AND d.donor_id=%s";    params.append(donor_id)
        if status_f:
            query += " AND d.status=%s";      params.append(status_f)
        query += " ORDER BY d.donation_date DESC"

        cur = mysql.connection.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()

        return jsonify([{
            "id":             r[0],
            "amount":         float(r[1]) if r[1] else 0,
            "date":           str(r[2]),
            "status":         r[3] or "success",
            "payment_method": r[4] or "online",
            "donor_name":     r[5] or "Unknown",
            "campaign_title": r[6] or "Unknown"
        } for r in rows])
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ============================================================
#  RUN
# ============================================================

if __name__ == '__main__':
    app.run(debug=True)