# Step 1-2: Create project and init
mkdir resume-ai-assistant && cd resume-ai-assistant && uv init

# Step 3: Create venv
uv venv

# Step 5: Install all packages
uv add streamlit langchain langchain-community langchain-google-genai langchain-text-splitters pypdf faiss-cpu python-dotenv

# Step 7: Create pdf folder
mkdir pdf

# Step 9: Run app
uv run streamlit run app.py