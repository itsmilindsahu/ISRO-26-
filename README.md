# ISRO-26-
Bharatiya Antariksh Hackathon 2026

## Free deployment plan

This app is ready for a free split deployment:
- Frontend: Netlify (static site)
- Backend: Render (free web service)

### 1) Deploy the backend on Render
1. Push this repository to GitHub.
2. In Render, create a new Web Service and connect this repo.
3. Use these settings:
   - Root Directory: keep empty
   - Build Command: `pip install -r backend/requirements.txt`
   - Start Command: `cd backend && gunicorn app:app --bind 0.0.0.0:$PORT`
4. Render will give you a public URL such as `https://sat-frame-interp-backend.onrender.com`.

### 2) Deploy the frontend on Netlify
1. In Netlify, create a new site from Git and connect this repo.
2. Set the publish directory to `frontend`.
3. Add a redirect so `/api/*` reaches your Render backend.
   Use this redirect rule in Netlify:
   - `/api/* https://YOUR_RENDER_BACKEND_URL/:splat 200`
4. Deploy the site.

### 3) Final app URL
Once the backend URL is plugged in, the frontend will work from Netlify and call the backend through `/api`.

### Local run
```bash
python backend/app.py
```
Then open `http://localhost:5000`.
