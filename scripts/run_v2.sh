#!/usr/bin/env bash
# v2 benchmark, start to finish. Resumable: rerunning skips audio that already exists.
#   nohup scripts/run_v2.sh > logs/run_v2.log 2>&1 &
#   tail -f logs/run_v2.log
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
set -a; source .env; set +a
mkdir -p logs data/v2

ACCENTS="${ACCENTS:-us,indian,nigerian,mexican}"
PER_KIND="${PER_KIND:-40}"

# keep earlier runs out of this report, without deleting them
if [ -d runs ] && [ ! -d runs_v1 ] && ls runs | grep -q .; then
  mv runs runs_v1 && echo "moved earlier runs to runs_v1/"
fi

if [ ! -f data/v2/cases.jsonl ]; then
  voxguard gen --n "$PER_KIND" --seed 2026 --out data/v2/cases.jsonl --accents "$ACCENTS"
fi
echo "== synthesizing caller audio ($(wc -l < data/v2/cases.jsonl) turns)"
voxguard synth --cases data/v2/cases.jsonl --out data/v2/audio --accents "$ACCENTS"
voxguard audit --cases data/v2/cases.jsonl

echo "== benchmarks (parallel across providers)"
for c in openai gemini assemblyai deepgram; do
  voxguard bench --config "configs/v2/$c.toml" > "logs/v2_$c.log" 2>&1 &
done
wait
for c in openai gemini assemblyai deepgram; do echo "-- $c"; tail -n 12 "logs/v2_$c.log"; done

echo "== layer 3: recovery"
voxguard recover --config configs/v2/deepgram.toml
voxguard report && voxguard samples --cases data/v2/cases.jsonl && voxguard publish
echo "== Three layers"; sed -n '/## Three layers/,$p' report/REPORT.md
echo "== ALL DONE"
