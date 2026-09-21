from flask import Blueprint, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import pandas as pd

from prediction import analyze_email
from database import insert_analysis, fetch_stats, fetch_recent_activity, fetch_reports

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "..", "reports")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"txt", "csv"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


api_bp = Blueprint("api", __name__)
ui_bp = Blueprint("ui", __name__)


# ---------------- UI ROUTES ----------------


@ui_bp.route("/")
def dashboard_page():
    return render_template("dashboard.html")


@ui_bp.route("/email-analysis")
def email_analysis_page():
    return render_template("email_analysis.html")


@ui_bp.route("/reports")
def reports_page():
    return render_template("reports.html")


@ui_bp.route("/about")
def about_page():
    return render_template("about.html")


# ---------------- API ROUTES ----------------


@api_bp.route("/stats", methods=["GET"])
def api_stats():
    stats = fetch_stats()
    recent = fetch_recent_activity(limit=10)
    return jsonify({"stats": stats, "recent": recent})


@api_bp.route("/reports", methods=["GET"])
def api_reports():
    search = request.args.get("search", "").strip()
    risk_filter = request.args.get("risk_level", "").strip()
    prediction_filter = request.args.get("prediction", "").strip()

    reports = fetch_reports(
        search_term=search,
        risk_level=risk_filter,
        prediction=prediction_filter,
    )
    return jsonify({"reports": reports})


@api_bp.route("/analyze-email", methods=["POST"])
def api_analyze_email():
    subject = request.form.get("subject", "").strip()
    email_text = request.form.get("email_text", "").strip()
    file = request.files.get("file")

    if not email_text and file is None:
        return jsonify({"error": "Please provide email text or upload a file."}), 400

    # ---------- FILE UPLOAD ---------- #
    if file and file.filename:
        if not allowed_file(file.filename):
            return jsonify({"error": "Only .txt and .csv files are supported."}), 400

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_DIR, filename)
        file.save(save_path)

        ext = filename.rsplit(".", 1)[1].lower()
        try:
            if ext == "txt":
                with open(save_path, "r", encoding="utf-8", errors="ignore") as f:
                    email_text = f.read()
            elif ext == "csv":
                df = pd.read_csv(save_path)
                subj_col = next((c for c in df.columns if "subject" in c.lower()), None)
                text_col = next(
                    (c for c in df.columns if "email" in c.lower() and "text" in c.lower()),
                    None,
                )
                if text_col is None:
                    # PPT: format validation
                    return jsonify({"error": "CSV must contain an email_text column."}), 400

                row = df.iloc[0]
                email_text = str(row[text_col])
                if not subject and subj_col is not None:
                    subject = str(row[subj_col])
        except Exception as exc:  # pragma: no cover
            return jsonify({"error": f"Failed to process uploaded file: {exc}"}), 400

    email_text = email_text.strip()
    if not email_text:
        return jsonify({"error": "Email content cannot be empty."}), 400

    # ---------- RUN MODEL ---------- #
    analysis = analyze_email(email_text=email_text, subject=subject or "N/A")

    # ---------- STORE RESULT (no email body stored) ---------- #
    insert_analysis(
        subject=analysis.get("subject", subject or "N/A"),
        prediction=analysis["prediction"],
        risk_level=analysis["risk_level"],
    )

    return jsonify(analysis)

