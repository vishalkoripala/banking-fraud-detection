# Deployment Guide

This gets you a real, public HTTPS URL for the fraud detection API and demo
page. Verified against current (Aug 2026) platform documentation before
being written down here — not copied from a stale tutorial.

## Recommended: Render.com

Render's free Web Service tier requires no credit card, builds directly
from a Dockerfile, and needs zero changes to this project.

1. **Train locally first.** Render builds your code; it does not have the
   150MB dataset and won't run `train_final_model.py` for you. Before you
   push anything, run it locally so `model/fraud_model.joblib`,
   `model/model_metadata.json`, and `model/background_sample.csv` exist and
   get committed.
2. **Push this project to a GitHub repo.** `data/` is already excluded via
   `.gitignore` — you do not need (and should not try) to push the raw
   150MB dataset. `git init`, `git add .`, `git commit -m "fraud pipeline"`,
   create a new repo on GitHub, then `git push`.
3. **Sign up at [render.com](https://render.com)** — email or GitHub OAuth,
   no payment details required for the free tier.
4. **New + → Web Service → connect your repo.** Render detects the
   Dockerfile automatically. Choose the **Free** instance type.
5. **Deploy.** First build takes a few minutes (installing pandas,
   scikit-learn, shap, etc.). Render gives you a URL like
   `https://your-app-name.onrender.com` once it's live.
6. **Before a live demo, warm it up.** Free services spin down after 15
   minutes of inactivity; the next request takes 30–60 seconds to wake back
   up. Hit `https://your-app-name.onrender.com/health` a minute or two
   before you present so the audience doesn't wait through a cold start.

Your live demo page is then at the root URL itself (no `/demo` path needed)
and the API docs are at `/docs`.

## Alternative: Hugging Face Spaces (Docker SDK)

Also free, no credit card, and a natural fit given this is an ML project —
but needs two small changes first, because Spaces has different platform
conventions than a generic container host:

- Spaces expects the app to listen on **port 7860**, not 8000. Either set
  `ENV PORT=7860` in the Dockerfile, or note that this project's Dockerfile
  already reads `$PORT` at runtime, so Spaces' own injected port is
  respected automatically — just confirm in the Space's logs that it bound
  to 7860.
- Spaces containers can typically only write outside the image at `/tmp`.
  This project's audit logger writes to `logs/predictions.log` relative to
  the working directory, which will fail on Spaces' filesystem restrictions.
  Change the path in `api.py`'s `logging.FileHandler(...)` call to
  `/tmp/predictions.log` before deploying there.

If you don't want to make those two edits, use Render instead — it needs
none.

## Fastest option if you're out of time: run it locally and demo on your
   own screen

A public URL is not the only way to demonstrate a "live" application. If
neither of the above fits before your deadline, `uvicorn api:app` on your
own laptop plus a screen-share is a completely legitimate live demo of a
real, working system — every request in that demo hits the actual trained
model, the same as it would on a public URL.
