# echoSentinel — Stitch UI generation prompts

Paste the MASTER PROMPT first to set the design system, then generate screens
one at a time with the per-screen prompts (Stitch works best one screen per
generation, carrying the theme forward).

---

## MASTER PROMPT (paste first, keep in every generation)

Design "echoSentinel" — a high-grade naval defence web application for underwater
acoustic surveillance. It detects and classifies underwater sounds (vessels, marine
animals, natural sounds, anthropogenic noise) from hydrophone recordings using AI.
The UI must feel like a classified sonar command console: precise, calm, powerful.

THEME — "Abyssal Sonar":
- Background: near-black deep ocean gradient, #030B10 at top fading to #06131C —
  never pure black, always a hint of deep blue-green depth. Subtle animated grid of
  faint dots (sonar chart paper) at 3% opacity across all screens.
- Primary accent: radiant cyan #00F0FF, with glow — every interactive element emits
  a soft cyan outer glow (box-shadow blur 20-30px at 25% opacity) that intensifies
  on hover.
- Secondary accents: teal #2DD4BF, deep blue #0EA5E9. Alert/danger: sonar red
  #FF4D5E. Warning: amber #FFB454.
- Class color code used EVERYWHERE (chips, timeline segments, charts):
  vessel = radiant cyan #00F0FF, marine animal = bio-green #34F5C5,
  natural sound = deep blue #38BDF8, other anthropogenic = signal amber #FFB454.
- Surfaces: glassmorphism panels — rgba(10, 25, 35, 0.55), backdrop blur 16px,
  1px border of rgba(0, 240, 255, 0.15), corner radius 14px. Panels feel like
  layered glass HUD screens floating over the abyss.
- Typography: headings in Space Grotesk (geometric, technical), data/numbers in
  IBM Plex Mono (monospace — coordinates, timestamps, confidence scores always
  monospace). Uppercase micro-labels with 0.15em letter-spacing for section tags.
- Iconography: thin-line 1.5px stroke icons, cyan, with subtle glow.
- Motif: SOUND AND VIBRATION. Waveforms, concentric sonar rings, oscilloscope
  traces, and frequency bars appear as decorative and functional elements
  throughout. Background hero elements: slow expanding concentric circles (sonar
  pings) and a gently undulating waveform line across section dividers.

MOTION LANGUAGE (apply to every screen — animations must be buttery smooth, 60fps,
transform/opacity only):
- Easing: cubic-bezier(0.22, 1, 0.36, 1) ("power out") for entrances;
  cubic-bezier(0.4, 0, 0.2, 1) for micro-interactions.
- Page load: elements cascade in with 24px upward slide + fade, staggered 70ms
  apart, 600ms duration. HUD panels "power on" with a brief 150ms brightness bloom.
- Scroll: every section reveals on scroll with staggered rise-and-fade; large
  numbers count up from 0 when they enter the viewport; decorative waveform lines
  draw themselves left-to-right (stroke-dashoffset animation) as you scroll;
  background sonar rings parallax slower than foreground content (depth).
- Idle ambience: a sonar sweep line rotates slowly in hero radar elements (8s per
  revolution); waveform decorations oscillate gently; glow on primary buttons
  breathes (4s pulse, ±10% intensity). Subtle — never distracting.
- Hover: cards lift 4px with deepened glow, 200ms; buttons fill with a cyan sweep
  from left, 250ms; table rows underline-glow.
- Transitions between screens: 350ms crossfade with 12px vertical drift.
- Numbers/metrics: odometer-style rolling digits on update.

Layout: left vertical nav rail (72px, icons + glow indicator for active screen),
top status bar with system clock (monospace, UTC), model status LED (pulsing green
dot = model loaded), and mission name. Content area max-width 1440px. Generous
spacing — this is a calm, confident console, not a crowded dashboard.

---

## SCREEN 1 — Command Overview (Home)

Hero band: large radar/sonar circle on the right (300px) with rotating sweep line,
concentric rings expanding outward every 3s and fading; small blips appear where
recent detections happened. On the left: "ECHOSENTINEL" wordmark, micro-label
"UNDERWATER DOMAIN AWARENESS", one-line mission statement, and two CTAs:
"Analyze Recordings" (primary, glowing) and "Live Monitor" (ghost button).
A thin animated waveform line runs under the hero, oscillating slowly.

Below, a 4-stat row (glass tiles, count-up numbers, monospace):
"Files Analyzed 1,284", "Events Detected 8,392", "Detection IER 0.41",
"Avg Processing 3.2s / min of audio". Each tile has a small sparkline and its
class-colored icon.

Then "Recent Detections" — a horizontal scrolling strip of event cards: each card
shows a mini waveform thumbnail with the detected segment highlighted in its class
color, class chip (VESSEL / MARINE ANIMAL / NATURAL / ANTHROPOGENIC), confidence
as a monospace percentage, timestamp, and source file. Cards slide in staggered
on scroll.

Bottom: "System Status" panel — model version, classes supported (4 color-coded
chips), pipeline diagram as a horizontal flow (Audio → PCEN → Neural Net →
Events → JSON) with a small pulse of light traveling along the flow line every
few seconds.

## SCREEN 2 — Upload & Analyze

Center stage: a large drop zone styled as a sonar aperture — dashed circular ring
that ripples with concentric cyan circles when a file is dragged over it. Text:
"Drop hydrophone recordings" + supported formats (.wav, any sample rate/bit
depth, up to 250 MB+). Browse button beneath.

Below: the processing queue — each file becomes a glass row with: filename
(monospace), duration, sample rate badge, a live progress bar styled as a
scanning waveform (the waveform of the actual file fills left-to-right in cyan as
analysis progresses, with a bright scan line at the frontier), status text
("Streaming block 14/40 · detecting…"), and a cancel icon. Completed rows flip to
show a summary: number of events found, per-class colored count chips, and an
"Open results" arrow. Queue rows enter with a slide-stagger; completion triggers
a brief ring-pulse on the row.

Side panel (right, collapsible): analysis settings — model selector, per-class
sensitivity sliders with cyan glow thumbs, "recall-biased (defence default)"
toggle, output format (PS-12 JSON) note.

## SCREEN 3 — Analysis Results (the core screen)

Top bar: file name (monospace), duration, sample rate, "Export JSON" primary
button, "Report" ghost button.

Main element: full-width interactive timeline, two synced lanes:
1) Waveform lane — the audio waveform in dim cyan.
2) Spectrogram lane — dark spectrogram with cyan-to-white intensity.
Overlaid on both: detected event segments as translucent color blocks (class
colors) with glowing edges; hovering a segment lifts its glow and shows a HUD
tooltip: class, start–end (mm:ss), duration, confidence (monospace). A vertical
playhead line with a small sonar-ping animation at its head moves during
playback. Zoom/pan controls; the timeline pans with buttery inertial easing.

Below-left: Events table — sortable columns: #, class (colored chip), start, end,
duration, confidence (as a thin horizontal bar + monospace %). Row hover syncs a
highlight pulse on the corresponding timeline segment.

Below-right: two compact analytics cards: "Class distribution" donut (class
colors, animated draw-in on scroll, center shows total events) and "Confidence
over time" line chart with glow line and area fade.

Audio player bar pinned at bottom: play/pause with ripple, current time
(monospace), and a mini frequency-bars visualizer that dances with playback.

## SCREEN 4 — Live Monitor (streaming mode)

Full-width real-time scrolling spectrogram (right-to-left flow, newest at right
edge) with detected events materializing as colored translucent blocks the moment
they close, each arriving with a quick ring-pulse. Above it: live status strip —
"LIVE" red-dot badge pulsing, hydrophone source name, elapsed time, input level
meter styled as a vertical waveform.

Right column: live event feed — newest detection cards slide in from the top with
spring easing; each card: class chip, confidence, time, mini waveform snippet.
Critical detections (vessel above threshold) flash the card border in sonar red
once and pin an alert banner at top: "VESSEL SIGNATURE DETECTED — 14:32:07 UTC"
with acknowledge button.

Bottom row: three live gauges (glass tiles): events/hour, dominant class (big
colored label), model latency ms — all with smoothly interpolating values.

## SCREEN 5 — Mission Archive (history)

Filterable table/grid toggle of all analyzed files: date, filename, duration,
events, dominant class, IER-mode badge. Filters as glowing chips (by class, date
range, confidence). Grid mode shows waveform-thumbnail cards. Clicking navigates
to Screen 3. Sticky filter bar condenses on scroll (height shrink, 250ms). Rows
paginate with fade-through.

## SCREEN 6 — System & Model

Model card panel: "PANNs CNN14 + PCEN" with a stylized network diagram whose
nodes pulse sequentially left-to-right; training metadata (epoch, dataset size,
classes); metrics row (IER, miss rate, false-alarm rate) as radial gauges that
sweep in on scroll. Threshold calibration table (per-class high/low, monospace)
with edit affordance. Docker/offline status badge: "AIR-GAPPED READY" with a
shield icon. Version history timeline down the page with scroll-reveal steps.

---

## Notes for whoever implements the generated design

- All animation: transform + opacity only (GPU-composited); no layout thrash.
  Respect `prefers-reduced-motion` (disable ambient loops, keep fades).
- Recommended stack: React + Tailwind + Framer Motion (scroll: `whileInView`,
  springs for cards; `useScroll` for parallax); waveform/spectrogram via
  wavesurfer.js; charts via Recharts with custom glow filters.
- The backend contract is the PS-12 JSON (`results.json`): categories ids 1-4 map
  to the class colors above; annotations carry start/end seconds + score, which
  drive the timeline blocks and tables directly.
