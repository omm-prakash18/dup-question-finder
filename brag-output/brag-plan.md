# Brag Plan: dup-question-finder 🔍

## What is this app?
A production-grade semantic duplicate question detection and vector retrieval engine combining fine-tuned Sentence-BERT (`all-mpnet-base-v2`), FAISS IVFFlat indexing, SQLite metadata caching, and FastAPI.

## The angle
High-tech precision engineering meets instant semantic search. Real-time NLP that finds duplicates at sub-10ms latency across 1M+ questions with 84% accuracy.

## Hook (first 2-3 seconds)
Big bold typography on sleek dark background: "Ever asked a question that was answered 1,000 times before?"
Dynamic search query input box appears: typing *"How do I learn Python from scratch?"*

## Key moments (the middle)
- **Scene 2 (Bi-Encoder Dense Search)**: Query instantly projects into 768-dimensional embedding space; FAISS IVFFlat indexes scan 1M+ vectors in parallel.
- **Scene 3 (Instant Semantic Match & Confidence)**: Ranked duplicate questions slide in with high cosine confidence scores (`0.94` — "How should a beginner start learning Python?").
- **Scene 4 (Battle-Tested Metrics)**: 56/56 verified test suites passing (100%), p50 latency 9.45ms, and sub-second batch ingestion.

## Outro / punchline
"Stop answering the same question twice. dup-question-finder is live."
GitHub repository badge + FastAPI Swagger docs preview.

## User flow worth showing
1. User types ambiguous or paraphrased question.
2. SBERT bi-encoder generates dense normalized embeddings on the fly.
3. FAISS retrieves top-K nearest neighbors in <10ms.
4. FastAPI serves validated JSON payload with similarity confidence.

## Tone
- Preset: `polished`
- Creative direction: Sleek high-tech AI infrastructure launch
- Interpretation: Fast-paced, confident, clean typography, dark neon aesthetic, glowing accents, and crisp motion.

## Format: landscape — 1920x1080
## Duration: 18 seconds

## Visual identity (from the project)
- Background: `#0B0F19` (Deep Slate / Dark Mode)
- Card Background: `#111827` / `#1F2937`
- Accent Primary: `#00F2FE` (Electric Cyan)
- Accent Secondary: `#8A2BE2` (Transformers Purple)
- Accent Success: `#10B981` (Emerald Green)
- Text Primary: `#F9FAFB` (Crisp White)
- Text Secondary: `#9CA3AF` (Muted Gray)
- Display font: `Inter`, `Outfit`, sans-serif
- Body font: `JetBrains Mono`, `Inter`, monospace

## Share copy (draft)
"Just shipped dup-question-finder 🔍 — a high-performance NLP pipeline & FastAPI microservice powered by fine-tuned Sentence-BERT and FAISS for sub-millisecond semantic duplicate question retrieval!"

## Audio direction
- Role: Modern cinematic electronic synth bed with subtle punchy UI accents
- Music: Upbeat energetic tech synth bed
- SFX posture: Restrained, crisp keystrokes and whooshes on match reveals

## Storyboard

### Scene 1 — The Problem & Query Hook — 3.5s
Dark canvas `#0B0F19`. Center glows with subtle cyan ambiance.
Headline reveals: **"Stop answering the same question twice."**
A glowing search bar enters. Simulated typing: *"What is the best way to learn Python from scratch?"*
- Sequential/interaction: Search bar drops in, cursor blinks, query types out smoothly.
- Audio intent: Subtle keystrokes and low synth swell.
- Transition mood: Clean zoom/cut → Scene 2

### Scene 2 — SBERT + FAISS Vector Engine — 4.0s
Visual transition into the neural architecture layer:
- Badge: `Sentence-BERT (all-mpnet-base-v2)` → `768-Dim Dense Vectors`
- Glowing FAISS index animation with 500,000+ indexed question nodes.
- Sequential/interaction: 3 tech specs pulse in sequence: `Bi-Encoder Embedding` → `IVFFlat Index` → `Sub-10ms Lookup`.
- Audio intent: Rising electronic pulse.
- Transition mood: Slide right → Scene 3

### Scene 3 — Instant Duplicate Match Cards — 4.5s
Live search interface recreation with top results card:
- Match #1: *"How should a beginner start learning Python?"* — `94.2% Match` (Green Pill)
- Match #2: *"Best resources for Python beginners"* — `88.5% Match`
- Match #3: *"What is the best way to master Python?"* — `76.1% Match`
- Sequential/interaction: Ranked result cards cascade in one by one.
- Audio intent: Crisp tick on each card appearance.
- Transition mood: Fast wipe → Scene 4

### Scene 4 — Production Performance & Metrics — 3.5s
Dark grid with neon metric cards:
- **84.0% Accuracy** (vs 72.0% Baseline)
- **9.45ms p50 Latency** (Load Tested)
- **56 / 56 Unit & Golden Tests Passed (100%)**
- Sequential/interaction: Metric counters tick up quickly to final values.
- Audio intent: Satisfying low thud on stat lock-in.
- Transition mood: Soft fade → Scene 5

### Scene 5 — Outro & Open Source Call to Action — 2.5s
Final screen with project title and repo link:
- **`dup-question-finder 🔍`**
- Tagline: *"Production-Grade Semantic Retrieval with PyTorch + FAISS + FastAPI"*
- GitHub: `github.com/omm-prakash18/dup-question-finder`
- Sequential/interaction: Logo and badges glow into focus.
- Audio intent: Clean resolving chord and fade out.

**Music mood for this video:** High-tech electronic synth with crisp modern pacing.
**Audio summary:** Focused tech launch soundscape that moves smoothly from problem hook to live semantic search and performance stats.
