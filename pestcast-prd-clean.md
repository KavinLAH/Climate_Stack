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
