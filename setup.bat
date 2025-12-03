@echo off
echo 🔹 Creating virtual environment 'venv'...
python -m venv venv

echo 🔹 Activating virtual environment...
call venv\Scripts\activate

echo 🔹 Upgrading pip...
python -m pip install --upgrade pip

echo 🔹 Installing required Python packages...
pip install google-adk chromadb sentence-transformers

echo 🔹 Installing optional LLM packages...
pip install ollama groq

echo ✅ Setup complete!
echo To activate the environment later, run: venv\Scripts\activate
echo To deactivate, run: deactivate
pause
