# Hosting the demo (free)

The hosted demo lets people play sample calls or record their own voice and see what each
model records with and without voxguard. It runs on Streamlit Community Cloud.

## 1. Prepare the content on your machine

```bash
voxguard samples          # picks benchmark calls (hardest ones first) into demo/samples/
voxguard publish          # copies the latest run per model into published/
git add demo/samples published requirements.txt
git commit -m "Add demo samples and published results"
git push
```

Listen to a few files in `demo/samples/` before publishing. Sample voices are synthetic
(OpenAI TTS); the Live tab says so next to each sample.

## 2. Deploy

1. Go to share.streamlit.io and sign in with GitHub.
2. Create app: repository `sriharshamudumba/voxguard`, branch `main`, main file `ui/app.py`.
3. Advanced settings, Secrets: paste the contents of `.streamlit/secrets.toml.example` with
   your values filled in. Choose a `DEMO_PASSWORD` to share only with people you invite.
4. Deploy.

## How keys are protected

| Rule | Why |
|---|---|
| Hosted keys are unlocked only with the passcode | Strangers cannot spend your credits |
| `DEMO_SESSION_LIMIT` and `DEMO_DAILY_LIMIT` cap usage | A shared passcode cannot run up a large bill |
| Keys are passed to providers per request, never put in environment variables | Streamlit serves many visitors from one process; environment variables would be shared between them |
| Visitors can paste their own keys instead | Those are used for their request only |

Rotate any key you put in Secrets if the passcode leaks, and set spending limits in each
provider's dashboard as a second line of defense.
