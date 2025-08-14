import os

# Hint to app that we're running in a cloud context (optional for conditional behavior)
os.environ.setdefault("STOCK4U_CLOUD", "1")

# Reuse the exact same Streamlit dashboard as the developer environment
from dashboard.app import main

if __name__ == "__main__":
	main()
