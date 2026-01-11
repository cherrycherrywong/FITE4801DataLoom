# FITE4801 - Alternative Credit Scoring (DataLoom)

Quick guide to run locally, push to GitHub, and configure real-time Base44 integration.

Prerequisites
- Python 3.9+
- git (for pushing to GitHub)

Install
```bash
pip install -r requirement.txt
```

Local run
1. (Optional) Create a `.env` file in the project root with:
```
BASE44_API_KEY=your_real_base44_api_key_here
BASE44_ENDPOINT_URL=https://app.base44.com/api/apps/<APP_ID>/entities/TravelDataProfile
```
2. Train the model (creates `credit_brain.pkl`):
```bash
python train_model.py
```
3. Start the server:
```bash
uvicorn main:app --reload
```
4. Open frontend: http://127.0.0.1:8000/search

Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: alternative credit scoring app"
git remote add origin https://github.com/cherrycherrywong/FITE4801DataLoom.git
git branch -M main
git push -u origin main
```

Notes on real-time Base44 data
- `main.py` and `extract_data.py` prefer `BASE44_API_KEY` and `BASE44_ENDPOINT_URL` from environment or `.env`.
- If those are not set, the app will fall back to the provided internal CSV backup.

Deploying publicly
- This project is a FastAPI app (server + templates). To expose the live app publicly, deploy to a service like Render, Fly, or Heroku and set environment variables (BASE44_API_KEY, BASE44_ENDPOINT_URL) in the service dashboard.
- Alternatively, keep the repo on GitHub and run locally to view the HTML frontend via `uvicorn`.

Security
- Never put your real API key into a public repo. Use environment variables or GitHub secrets when deploying.
