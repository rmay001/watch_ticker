# Watch Ticker

A mechanical watch timegrapher. It listens to a microphone, detects ticks, and shows live graphs and analysis in a tkinter window.

## Features
- Live scrolling level graph with an adjustable tick threshold
- Tick threshold crossings are timestamped from a 5 ms rolling RMS within each audio block, rather than at block boundaries
- Beat rate, nearest standard bph, rate error (s/day), beat error, jitter and intervals
- Selectable movement lift angle and an acoustic-event amplitude estimate
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
- The amplitude estimate applies the standard lift-angle formula to the spacing of two detected acoustic events. That spacing is only a proxy for escapement lift time and may not identify the true lift endpoints; microphone-only results are uncalibrated and should be checked against a timegrapher.
- App timing status reflects generic app targets for rate, beat error, and interval jitter; it is not a watch-service diagnosis.
- Run at least 2 minutes (5 or more is better) for a stable rate error.
- Use headphones if you enable Listen, to avoid feedback.
- "Synthetic ticks from readings" plays a soft artificial tick/tock (alternating pitch) for each detected tick, with loudness following the measured tick level. It is mutually exclusive with Listen, which passes the filtered microphone audio through.
