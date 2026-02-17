# Theriax

Full-stack healthcare recommendation prototype:
- `Backend11.py`: main FastAPI backend (auth, dashboard, logs, retraining)
- `evolve/evolve_api.py`: AI prediction microservice
- `theriax-frontend/`: React + Vite frontend

## 1. Backend setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Create `.env` in the project root with:

```env
DATABASE_URL=...
SUPABASE_URL=...
SUPABASE_SERVICE_KEY=...
EVOLVE_URL=http://localhost:9000/predict
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
GOOGLE_REDIRECT_URL=http://localhost:5173/auth/callback
ENABLE_RETRAIN_SCHEDULER=false
```

Run backend:

```bash
uvicorn Backend11:app --reload --port 8000
```

## 2. Evolve service setup

Run in another terminal from project root:

```bash
uvicorn evolve.evolve_api:app --reload --port 9000
```

Train/upload initial model artifacts:

```bash
python evolve/train_model.py
```

## 3. Frontend setup

```bash
cd theriax-frontend
npm install
```

Create `theriax-frontend/.env`:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_SUPABASE_URL=...
VITE_SUPABASE_ANON_KEY=...
```

Run frontend:

```bash
npm run dev
```
