"""DocuMind entry point. Run with: streamlit run app.py

Navigation and pages are added in Phase 6; this file stays a thin shell.
"""

import streamlit as st

from config import APP_NAME, APP_TAGLINE, ensure_directories, settings


def main() -> None:
    st.set_page_config(page_title=APP_NAME, layout="wide")
    ensure_directories()
    st.title(APP_NAME)
    st.caption(APP_TAGLINE)
    st.info("Project scaffold only. No document features are implemented yet.")
    st.text(f"Data directory: {settings.paths.data_dir}")


if __name__ == "__main__":
    main()
