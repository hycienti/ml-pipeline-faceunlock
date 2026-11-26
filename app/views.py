from flask import Blueprint, redirect, render_template, session, url_for

from app.auth import login_required

bp = Blueprint("views", __name__)


@bp.get("/")
def index():
    return redirect(url_for("views.dashboard" if "user" in session else "views.login_page"))


@bp.get("/register")
def register_page():
    return render_template("register.html")


@bp.get("/login")
def login_page():
    return render_template("login.html")


@bp.get("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", username=session["user"])
