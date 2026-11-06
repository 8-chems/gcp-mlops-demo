import os

import streamlit as st


def _emails(var: str) -> set[str]:
    return {e.strip().lower() for e in os.getenv(var, "").split(",") if e.strip()}


def require_login() -> str:
    if not st.user.is_logged_in:
        st.title("MLOps demo")
        st.write("Sign in with your Google account to continue.")
        st.button("Sign in with Google", on_click=st.login)
        st.stop()
    email = (getattr(st.user, "email", "") or "").lower()
    if not email or not getattr(st.user, "email_verified", False):
        st.error("Your Google e-mail is missing or not verified.")
        st.button("Sign out", on_click=st.logout)
        st.stop()
    return email


def role_of(email: str) -> str | None:
    if email in _emails("ADMIN_EMAILS"):
        return "admin"
    allowed = _emails("ALLOWED_EMAILS")        # empty = any verified Google account
    return "user" if not allowed or email in allowed else None


def require_admin() -> str:
    """Call this INSIDE every admin function, not only when drawing the UI."""
    email = (getattr(st.user, "email", "") or "").lower()
    if not st.user.is_logged_in or role_of(email) != "admin":
        raise PermissionError("Admin only")
    return email