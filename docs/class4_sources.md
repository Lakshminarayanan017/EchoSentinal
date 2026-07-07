# Free sources for `other_anthropogenic` training clips

The dataset currently has **no** real class-4 (`other_anthropogenic`) audio. This class means
**human-made underwater sound that is NOT a ship**: active sonar pings, seismic airgun surveys,
pile driving / underwater construction, chains & anchors, underwater explosions, diver equipment.
(Ship engine/propeller sound is class 1 `vessel` — keep them separate.)

Download clips into: `Dataset\Other Anthropogenic\` (WAV preferred; MP3 is fine, the loader
converts). Then re-run `python scripts/00_audit_dataset.py`.

## Primary sources (best first)

1. **DOSITS Audio Gallery** — https://dosits.org/galleries/audio-gallery/
   The single best source. Curated underwater sound library with dedicated
   "Anthropogenic (People) Sounds" section: sonar (mid/low frequency active), seismic airguns,
   pile driving, vessel comparison clips, explosions, drilling. Free for education/research.
   → Download everything in the anthropogenic section except pure ship recordings.

2. **Freesound.org** — https://freesound.org (CC-licensed, free account)
   Search terms that work well:
   - `sonar ping` / `active sonar` / `submarine sonar`
   - `pile driving` / `pile driver`
   - `underwater explosion` / `depth charge`
   - `anchor chain` / `chain winch`
   - `hydrophone construction` / `underwater drilling`
   - `scuba breathing` / `diver regulator`
   Prefer recordings tagged `hydrophone` / `underwater`; a few in-air mechanical clips are still
   useful (the synthesizer band-limits and mixes them under ocean noise).

3. **NOAA SanctSound data portal** — https://sanctsound.ioos.us/ (also via NCEI archive)
   Real sanctuary hydrophone recordings with detection labels, including vessel and
   anthropogenic categories. Bigger download effort but domain-matched (real ocean background).

4. **Ocean Networks Canada** — https://data.oceannetworks.ca/ (free account)
   Hydrophone archive; search around known construction/ROV/vessel activity. Also their
   SoundCloud/listening-room highlights include anthropogenic events.

5. **MBARI Soundscape Listening Room** — https://www.mbari.org/soundscape-listening-room/
   Curated deep-sea hydrophone clips (MARS observatory) — some anthropogenic events.

## Bonus (expands other classes, optional)

- **Watkins Marine Mammal Sound Database** — https://cis.whoi.edu/science/B/whalesounds/
  Classic, free, well-labeled marine mammal calls → strengthens `marine_animal` (put in a new
  folder, e.g. `Dataset\Marine Animals Watkins\`, and add it to `configs/data.yaml`).
- **ShipsEar** (request via authors) and **DeepShip** (GitHub) — research vessel datasets →
  strengthens `vessel`.

## What to aim for

- 30–60 clips for class 4 is already transformative (we currently have ~0).
- Variety beats volume: a few each of sonar pings, airgun shots, pile-driving strikes,
  chain/metal impacts, explosions.
- Any sample rate / bit depth is fine — the pipeline standardizes everything to 32 kHz mono.
- Note the license of each Freesound clip if this project is ever published.

## Fallback

Independently of downloads, the scene synthesizer (`src/echosentinel/data/scene_synth.py`,
Phase 2) procedurally generates class-4 signatures: tonal sonar pings/sweeps with reverb tails,
impulsive clanks, airgun-like low-frequency pulses. Real clips + procedural together give the
best coverage.
