# PestCast — Product Requirements Document

**"Waze for agricultural pests — crowdsourced, real-time, climate-correlated."**

---

## 1. Problem Statement

### The Core Problem
Agricultural pest outbreaks cause $200B+ in annual crop losses globally. Climate change is accelerating pest migration — species are invading new regions faster than any institution can track. The fall armyworm alone spread from the Americas to all of sub-Saharan Africa in under 3 years, causing $13B in annual damage.

### Why Existing Solutions Fail
- **USDA/Extension services** publish pest reports weekly or biweekly. Pest fronts move daily. The latency is fatal.
- **Satellite remote sensing** can detect crop damage after it's happened but can't identify pest species or provide early warning.
- **Commercial scouting services** cost $5–15/acre, pricing out smallholder farmers entirely.
- **Existing pest ID apps** (Plantix, Picture Insect) identify pests but do nothing with the data — no aggregation, no prediction, no alerts. They're individual tools, not a network.

### The Gap
There is no system that combines real-time pest identification, crowdsourced geographic intelligence, and climate-correlated predictive alerts. Farmers have no way to know what's coming toward their fields until it arrives.

---

## 2. Product Vision

PestCast turns every farmer's smartphone into a node in a distributed pest surveillance network. Farmers snap a photo → get instant pest ID (offline) → the sighting feeds a live crowdsourced heatmap → climate data projects where the pest front is heading → neighboring farms get early warning alerts hours to days before infestation.

### Success Metrics
| Metric | Target |
|--------|--------|
| On-device inference latency | < 500ms on mid-range Android |
| Classification accuracy (top-3) | > 85% across target species |
| Time-to-alert for neighboring farms | < 6 hours from first cluster detection |
| Offline functionality | Full ID capability with 0 connectivity |

### Hackathon Demo Success Criteria
- Live on-device pest classification on a physical phone
- Sightings appear on heatmap in real-time
- Climate overlay visualizes wind/temp data
- Prediction zone renders on map showing projected pest movement
- End-to-end flow completes in under 60 seconds during demo

---

## 3. User Personas

### Primary: Smallholder Farmer (Maria)
- Farms 15 acres of corn and vegetables in California's Central Valley
- Uses a mid-range Android phone ($150–300)
- Intermittent cell service in fields
- Checks weather apps daily, moderately tech-comfortable
- **Need:** "I lost 30% of my corn to armyworm last season. My neighbor saw them a week before me — I wish I'd known."

### Secondary: Agricultural Extension Agent (David)
- Covers a 5-county region, responsible for pest advisories
- Currently drives to fields manually to scout
- **Need:** "I'm one person covering 500 farms. I need real-time eyes across my whole region."

### Tertiary: Crop Insurance Analyst (Sarah)
- Assesses pest-related claims across a portfolio
- **Need:** "I need spatiotemporal pest pressure data to price risk accurately."

---

## 4. Feature Requirements

### P0 — Must Have for Demo

#### F1: On-Device Pest Classification
- **Description:** Camera capture → instant pest species identification running entirely on-phone
- **Input:** Photo from device camera or gallery
- **Output:** Top-3 species predictions with confidence scores, common name, severity indicator
- **Model:** MobileNetV3 or EfficientNet-Lite, quantized to INT8 via TFLite
- **Training data:** IP102 dataset (75,000+ images, 102 species) — fine-tune on 15–20 priority species for demo
- **Offline:** Full functionality with zero connectivity
- **Performance:** < 500ms inference on Snapdragon 600-series equivalent

#### F2: Sighting Reporting & Sync
- **Description:** Each classification generates a sighting report: species, confidence, GPS, timestamp, optional severity tag
- **Offline queue:** Sightings stored locally and batch-synced when connectivity returns
- **Backend:** Supabase (Postgres + Realtime subscriptions)
- **Schema:**
  ```
  sightings {
    id: uuid
    species: string
    confidence: float
    latitude: float
    longitude: float
    severity: enum(low, medium, high, critical)
    image_url: string (optional)
    device_id: string
    created_at: timestamp
    synced: boolean
  }
  ```

#### F3: Live Pest Heatmap
- **Description:** Web/mobile map showing aggregated pest sightings as a density heatmap
- **Tech:** Mapbox GL JS with heatmap layer
- **Features:**
  - Filter by species, time range, severity
  - Cluster markers at low zoom, heatmap at high zoom
  - Real-time updates via Supabase Realtime subscriptions
  - Color scale: green (low) → yellow → orange → red (critical density)

#### F4: Climate Data Overlay
- **Description:** Toggle-able layer showing temperature, humidity, wind speed/direction from weather APIs
- **Data source:** Open-Meteo API (free, no key required) or NOAA GFS
- **Display:** Wind direction arrows, temperature gradient shading, humidity contours
- **Purpose:** Visual correlation between climate conditions and pest movement

#### F5: Predictive Alert Zone
- **Description:** Projected pest movement zone rendered as a semi-transparent polygon on the map
- **Logic (hackathon-scope):**
  1. Identify clusters of 3+ sightings of same species within 10km and 48 hours
  2. Calculate cluster centroid and movement vector from sequential sighting timestamps
  3. Project forward 24–72 hours, weighted by wind direction and temperature gradient
  4. Render projected zone as expanding polygon in direction of movement
- **Alert:** Push notification or SMS to farms within projected zone
- **Note:** This is a simplified heuristic for demo. Production would use a proper spatiotemporal epidemiological model.

### P1 — Nice to Have

#### F6: Community Verification
Low-confidence classifications get flagged for peer review. Other farmers in the area can confirm or correct the species ID, building a feedback loop for model improvement.

#### F7: Pest Profile Cards
Tapping a species on the heatmap shows an info card: lifecycle, crops affected, recommended treatment, climate conditions that favor spread.

#### F8: Historical Timeline Playback
Slider to scrub through time and watch pest sightings animate across the map — powerful demo moment showing migration over weeks.

---

## 5. Technical Architecture

```
┌─────────────────────────────────────────────────┐
│                  MOBILE APP                      │
│              (React Native + Expo)               │
│                                                  │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │  Camera   │→│  TFLite   │→│  Sighting     │  │
│  │  Capture  │  │  Model    │  │  Report +     │  │
│  │           │  │  (on-     │  │  Offline      │  │
│  │           │  │  device)  │  │  Queue        │  │
│  └──────────┘  └───────────┘  └──────┬───────┘  │
│                                       │          │
└───────────────────────────────────────┼──────────┘
                                        │ sync
                                        ▼
┌─────────────────────────────────────────────────┐
│                SUPABASE BACKEND                  │
│                                                  │
│  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  Postgres DB  │  │  Realtime Subscriptions  │  │
│  │  (sightings)  │  │  (push to dashboard)     │  │
│  └──────────────┘  └──────────────────────────┘  │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│              WEB DASHBOARD                       │
│           (React + Mapbox GL JS)                 │
│                                                  │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │ Heatmap   │  │  Climate  │  │  Prediction  │  │
│  │ Layer     │  │  Overlay  │  │  Zone        │  │
│  │ (sighting │  │  (Open-   │  │  (movement   │  │
│  │  density) │  │  Meteo)   │  │   vector +   │  │
│  │           │  │           │  │   wind proj) │  │
│  └──────────┘  └───────────┘  └──────────────┘  │
└─────────────────────────────────────────────────┘
```

### Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Mobile framework | React Native + Expo | Fast to scaffold, TFLite plugin available |
| ML model | EfficientNet-Lite0, INT8 quantized | Best accuracy/latency tradeoff for mobile |
| Training data | IP102 (subset: 15–20 species) | Largest public pest image dataset |
| Backend | Supabase | Free tier, Postgres + Realtime out of the box, fast setup |
| Map | Mapbox GL JS | Heatmap layer, 3D terrain, free tier generous |
| Weather API | Open-Meteo | Free, no API key, global coverage |
| Prediction model | Heuristic (kernel density + wind vector projection) | Sufficient for demo, honest about limitations |

---

## 6. Data Strategy

### Pre-Seeded Demo Data
To make the demo compelling, pre-load the database with 100–150 synthetic sightings that tell a realistic story:
- Fall armyworm cluster moving NE across a California county over 10 days
- Aphid hotspot emerging near a river valley (humidity correlation)
- Scattered low-confidence sightings of a new invasive species
- Sightings spread across 30+ simulated "devices" to show network effect

### Seed Data Requirements
- Geographically realistic (follow actual farmland, not random coordinates)
- Temporally sequential (show clear migration directionality)
- Species distribution matches real-world prevalence
- Include a mix of high and low confidence scores

---

## 7. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Model accuracy is poor on target species | Medium | High | Reduce to 5–8 species, use top-3 instead of top-1. Honesty about limitations beats a bad demo |
| TFLite integration fails in React Native | Medium | Critical | Backup: run model via local Flask server on the phone. Or pre-record the on-device demo as video |
| Live demo crashes | Medium | High | Record a backup video walkthrough before presenting. Practice the pivot to video gracefully |
| Mapbox heatmap performance issues | Low | Medium | Pre-aggregate sightings into hex bins server-side instead of rendering raw points |
| Supabase Realtime drops during demo | Low | High | Pre-load all seed data so the map looks full regardless. Insert a sighting 30s before demo to pre-warm the connection |

---

## 8. Post-Hackathon Roadmap

- **v1.1:** Community verification + gamification (pest scout leaderboards)
- **v1.2:** Proper spatiotemporal epidemiological model replacing heuristic
- **v2.0:** Federated learning for continuous model improvement from field data
- **v2.1:** Data API product for crop insurance, seed companies, commodity traders
- **v3.0:** Integration with drone/satellite imagery for canopy-level damage assessment

## 8. Task Breakdown — 5 People

### Philosophy
Two people own the ML pipeline end-to-end — this is the technical centerpiece and the hardest thing to get right. Three people own the full-stack application. Pitch and demo prep are shared responsibilities — everyone contributes to what they built, and the team rehearses together.

---

### Person 1 & Person 2: ML Engineers — "The Model"
**Joint goal:** A trained, quantized, on-device pest classification model integrated into the mobile app

**Person 1 — Training & Optimization**

| Phase | Task |
|-------|------|
| Data Prep | Download IP102 dataset. Curate a subset of 15–20 high-priority species (fall armyworm, aphids, whitefly, etc.). Clean and balance the dataset — oversample rare classes, remove mislabeled images. Split into train/val/test (80/10/10) |
| Training | Fine-tune EfficientNet-Lite0 in PyTorch or TF/Keras using transfer learning. Freeze base layers, train classifier head for 15–20 epochs. Track val accuracy and loss. Aim for >85% top-3 accuracy on test set |
| Quantization | Convert trained model to TFLite. Apply post-training INT8 quantization. Validate that quantized accuracy doesn't drop more than 2–3% from full-precision. Export `pest_model.tflite` + `labels.json` |
| Benchmarking | Measure inference latency on target phone (aim < 500ms). If too slow: reduce input resolution from 224→160, try MobileNetV3-Small instead, or prune low-importance channels. Document accuracy/latency tradeoffs |

**Person 2 — Mobile Integration & On-Device Pipeline**

| Phase | Task |
|-------|------|
| App Scaffold | Set up React Native (Expo) project. Build navigation: Camera Screen → Results Screen → Sighting Submission. Configure camera permissions and gallery access |
| Inference Pipeline | Integrate `react-native-tflite` or equivalent plugin. Build the preprocessing pipeline: capture image → resize to model input dimensions → normalize pixel values → run inference → parse output tensor into top-3 predictions with confidence scores |
| Results UI | Build the results screen: species name, confidence bar visualization, severity tag picker (low/medium/high/critical), "Report Sighting" button |
| Sighting Flow | On submit: capture GPS coordinates via device location API, construct sighting object, POST to Supabase endpoint (coordinate with Person 3). Implement offline queue — store pending sightings in local state, batch sync when connectivity returns |
| Integration | Receive `.tflite` model from Person 1 as soon as it's ready. Run end-to-end tests on physical phone with 10+ pest images. Test airplane mode (offline classification). Measure and display inference latency on-screen for demo |
| Demo Prep | Polish camera UI, fix edge cases (blurry photos, low light). Ensure the phone is charged, app is stable, and airplane mode demo is rehearsed. Record a backup screen capture of the full flow |

**Shared responsibilities between Person 1 & 2:**
- Decide model architecture together (EfficientNet-Lite0 vs MobileNetV3 — benchmark both early)
- Person 1 can hand off intermediate model checkpoints so Person 2 can start integration before final training is done
- Both contribute to the technical Q&A: model architecture, quantization tradeoffs, accuracy metrics, edge deployment rationale

---

### Person 3: Backend & Data Engineer — "The Plumbing"
**Goal:** Supabase backend live with seed data, accepting submissions, pushing Realtime updates

| Phase | Task |
|-------|------|
| Backend Setup | Create Supabase project. Build `sightings` table with schema (see §4). Enable Realtime on the table. Configure Row Level Security for public inserts and reads |
| API Layer | Build endpoints: insert sighting, query sightings by bounding box + time range + species filter. Test with Postman/curl. Share Supabase URL + anon key with Person 2 (mobile) and Person 4 (dashboard) |
| Seed Data | Write a seed script generating 100–150 realistic synthetic sightings. Use real California Central Valley farmland coordinates. Create 2–3 pest "narratives": a fall armyworm migration moving NE over 10 days, an aphid cluster near a river valley, scattered low-confidence detections of a newer invasive species. Spread across 30+ simulated device IDs. Mix of confidence levels and severities |
| Prediction Endpoint | Build a serverless function or Supabase Edge Function for the prediction logic: query recent sightings for a species → identify clusters (3+ sightings within 10km/48hrs) → compute centroid + movement vector from timestamps → return a GeoJSON polygon projected 48–72 hours forward weighted by wind direction (fetched from Open-Meteo). This can be a straightforward heuristic — that's fine for demo |
| Integration | Help Person 2 debug mobile → backend sync. Help Person 4 debug Realtime subscriptions. Verify that a new sighting from the phone appears on the dashboard within seconds. Load test with rapid inserts to ensure Realtime doesn't choke |
| Demo Prep | Ensure seed data is loaded and looks good on the map. Write a 2-minute explanation of the backend architecture and data pipeline for their portion of the pitch |

---

### Person 4: Frontend & Visualization Engineer — "The Map"
**Goal:** Web dashboard with heatmap, climate overlay, prediction zone, and real-time updates

| Phase | Task |
|-------|------|
| Map Foundation | Scaffold React web app. Integrate Mapbox GL JS. Set up base map centered on Central Valley, CA with satellite/terrain hybrid style. Load sightings from Supabase and render as a heatmap layer with color scale (green → yellow → orange → red) |
| Filtering & Interaction | Add species filter dropdown, time range selector, severity toggle. Cluster markers at low zoom levels, switch to heatmap at higher zoom. Add click-to-inspect: tap a cluster to see individual sighting details |
| Real-Time Layer | Subscribe to Supabase Realtime — new sightings animate onto the map live with a pulse effect. This is a key demo moment: submit from phone → appears on dashboard in real time |
| Climate Overlay | Fetch wind speed, wind direction, temperature, and humidity from Open-Meteo API for the demo region. Render as a toggle-able layer: wind direction arrows via Mapbox symbol layer, temperature gradient as a color fill. This shows the visual correlation between weather patterns and pest movement |
| Prediction Zone | Fetch the GeoJSON prediction polygon from Person 3's endpoint. Render as a pulsing semi-transparent red/orange polygon on the map. Add an alert banner at the top: "⚠️ Fall armyworm projected to reach Davis, CA within 48 hours." If time permits, add a timeline scrubber to animate sighting history over time |
| Polish & Deploy | Add legend, loading states, smooth layer transitions. Make it responsive. Deploy to Vercel for a stable demo URL. Record a backup screen capture of the full dashboard walkthrough |
| Demo Prep | Write a 2-minute walkthrough of the dashboard for their portion of the pitch. Bookmark the deployed URL on the demo laptop |

---

### Person 5: Climate Intelligence & Data Science — "The Brain"
**Goal:** The climate data pipeline, prediction model, and the scientific narrative that makes this more than a reporting tool

| Phase | Task |
|-------|------|
| Climate Data Pipeline | Fetch historical + forecast weather data from Open-Meteo API for the demo region: wind speed, wind direction, temperature, humidity — past 14 days + 3-day forecast. Transform into map-ready formats (GeoJSON point grid with wind arrow bearings, temperature contour polygons). Provide this to Person 4 as a clean API or static JSON |
| Prediction Model | Build the core prediction heuristic as a standalone function. Inputs: clustered sighting data + wind vectors + temperature field. Logic: identify directional movement from sequential sighting timestamps, weight by wind direction and speed, apply temperature threshold filter (pests active above certain temps), output a projected polygon. Test with the seed data to make sure the prediction zone points in a visually sensible direction. Hand this off to Person 3 to deploy as an endpoint |
| Scientific Validation | Research the actual climate dependencies of demo pest species. Fall armyworm: migrates with warm fronts, active above 15°C, follows wind corridors. Aphids: population explosions above 60% humidity. Document these so the team can cite real science in the pitch and Q&A |
| Pest Profile Content | Write pest profile cards for the 3–5 main demo species: common name, scientific name, crops affected, climate conditions favoring spread, recommended farmer response. Person 4 can render these as info cards on the dashboard if time permits |
| Integration | Work with Person 3 to ensure the prediction function returns clean GeoJSON. Work with Person 4 to ensure climate overlay renders correctly and the visual story is coherent: sighting cluster → wind arrows → prediction zone all pointing the same direction |
| Demo Prep | Write a 1-minute explanation of the climate-prediction layer for their portion of the pitch. Prepare answers for climate-focused judge questions (Coline): why climate change worsens pest migration, carbon impact of crop loss, Global South applicability |

---

### Shared: Pitch & Demo (Everyone)

Pitch prep is not one person's job. Each person presents what they built.

| Task | Who |
|------|-----|
| Write the opening hook (problem statement, stats, vision) | Person 5 (strongest climate narrative) + Person 3 (data/numbers) |
| Live demo — phone classification | Person 2 holds the phone, Person 1 narrates the technical choices |
| Live demo — dashboard walkthrough | Person 4 drives the screen, Person 5 narrates the climate/prediction layer |
| Closing statement (market opportunity, roadmap) | Person 3 |
| Build 3–5 backup slides (problem, architecture, market) | Person 5 builds them, everyone reviews |
| Record backup demo video | Person 2 (phone screen capture) + Person 4 (dashboard screen capture) |
| Q&A assignments | Person 1 answers Henry's model/edge-AI questions. Person 5 answers Coline's climate questions. Person 3 answers Austin's business/architecture questions. Person 2 and 4 support on technical details |
| Full team rehearsal | Everyone. Run it twice. Time it. Identify where it might break. Practice the pivot to backup video |

---

### Dependency Map

```
Person 1 (Model Training) ──delivers .tflite──→ Person 2 (Mobile Integration)
Person 3 (Backend)        ──delivers Supabase URL + key──→ Person 2 (Mobile)
Person 3 (Backend)        ──delivers Supabase URL + key──→ Person 4 (Dashboard)
Person 5 (Climate Data)   ──delivers weather JSON/API──→ Person 4 (Dashboard)
Person 5 (Prediction Fn)  ──delivers function──→ Person 3 (deploys as endpoint)
Person 3 (Prediction EP)  ──delivers GeoJSON endpoint──→ Person 4 (renders zone)
Everyone                  ──converges for──→ Integration testing + demo rehearsal
```

**Critical path:** Person 3 (backend) must be ready first — Person 2 and Person 4 both need the Supabase endpoint to do real integration. Person 1 should deliver an intermediate model checkpoint to Person 2 as early as possible so mobile integration isn't blocked on final training.

---