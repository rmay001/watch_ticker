# Watch Ticker

A mechanical watch timegrapher. It listens to a microphone, detects ticks, and shows live graphs and analysis in a tkinter window.

## Features
- Live scrolling level graph with an adjustable tick threshold
- Beat rate, nearest standard bph, rate error (s/day), beat error, jitter and intervals
- Rate error measured by folding the high-frequency waveform over the whole run, so it does not depend on per-tick threshold detection
- Optional sound-card clock calibration against the system clock
- Input device selector, optional speaker monitoring, CSV export

## Setup (macOS)
```bash
brew install portaudio python-tk@3.14
python3 -m venv .venv
CFLAGS="-I$(brew --prefix portaudio)/include" LDFLAGS="-L$(brew --prefix portaudio)/lib" \
  .venv/bin/pip install -r requirements.txt
.venv/bin/python watch_timer.py
```

## Tips
- Put the watch directly on the microphone, or use a piezo contact mic.
- Run at least 2 minutes (5 or more is better) for a stable rate error.
- Use headphones if you enable Listen, to avoid feedback.
