"""
FOSC Inference API
==================
YOLOv11-based detector for Fiber Optic Splice Closures (FOSC)
and snowshoe lashing anchors on utility/telephone poles.

Endpoints:
  GET  /health            — liveness check
  GET  /status            — model status + version
  POST /detect            — detect FOSC in a single image URL
  POST /detect/batch      — detect FOSC in multiple image URLs
  POST /scan/tile         — scan all Mapillary sequences in a zoom-14 tile
  GET  /results           — return accumulated detection GeoJSON
  DELETE /results         — clear accumulated results

Environment variables:
  MAPILLARY_TOKEN         — Mapillary API access token
  MODEL_PATH              — path to YOLOv11 .pt weights (default: model/fosc_v1.pt)
  CONFIDENCE_THRESHOLD    — min detection confidence 0-1 (default: 0.45)
  PORT                    — server port (default: 8000)
"""

import os
import json
import math
import time
import asyncio
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fosc")

# ── Config ────────────────────────────────────────────────────────────────
MAPILLARY_TOKEN   = os.environ.get("MAPILLARY_TOKEN", "")
MODEL_PATH        = os.environ.get("MODEL_PATH", "model/fosc_v1.pt")
CONFIDENCE        = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.45"))
PORT              = int(os.environ.get("PORT", "8000"))
MAPILLARY_BASE    = "https://graph.mapillary.com"

# ── Model loader ──────────────────────────────────────────────────────────
model = None
model_status = "not_loaded"
model_version = "placeholder_v0"

def load_model():
    global model, model_status, model_version
    model_file = Path(MODEL_PATH)
    if model_file.exists():
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_file))
            model_status = "loaded"
            model_version = model_file.stem
            log.info(f"Model loaded: {model_file}")
        except Exception as e:
            model_status = f"error: {e}"
            log.error(f"Model load failed: {e}")
    else:
        # Placeholder — download YOLOv11n as base until custom weights arrive
        try:
            from ultralytics import YOLO
            log.info("Custom model not found — loading YOLOv11n base (placeholder)")
            model = YOLO("yolo11n.pt")
            model_status = "placeholder"
            model_version = "yolo11n_placeholder"
            log.info("Placeholder model ready. Deploy fosc_v1.pt to activate custom detection.")
        except Exception as e:
            model_status = f"error: {e}"
            log.warning(f"Placeholder model also failed: {e}. /detect will return empty results.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(
    title="FOSC Inference API",
    description="YOLOv11-based Fiber Optic Splice Closure detector for CV farmland road scanning",
    version="1.0.0",
    lifespan=lifespan
)

# ── In-memory results store ───────────────────────────────────────────────
# Persisted to /tmp/fosc_results.geojson on each write
RESULTS_PATH = Path("/tmp/fosc_results.geojson")
detections_store = []

def save_results():
    geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count": len(detections_store),
            "model": model_version,
            "confidence_threshold": CONFIDENCE
        },
        "features": detections_store
    }
    RESULTS_PATH.write_text(json.dumps(geojson, indent=2))

def add_detection(lat: float, lon: float, confidence: float,
                  cls: str, image_id: str, image_url: str,
                  sequence_id: str = "", road_note: str = ""):
    feat = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {
            "class": cls,
            "confidence": round(confidence, 3),
            "image_id": image_id,
            "image_url": image_url,
            "sequence_id": sequence_id,
            "road_note": road_note,
            "detected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ring_radius_mi": 1.5,
            "note": "FOSC detected on utility/telephone pole. 1.5mi prospecting perimeter applies."
        }
    }
    detections_store.append(feat)
    save_results()
    return feat

# ── Schemas ───────────────────────────────────────────────────────────────
class DetectRequest(BaseModel):
    image_url: str
    image_id: Optional[str] = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    sequence_id: Optional[str] = ""
    save: Optional[bool] = True   # save confirmed detections to store

class BatchDetectRequest(BaseModel):
    images: list[DetectRequest]
    save: Optional[bool] = True

class TileScanRequest(BaseModel):
    tile_x: int
    tile_y: int
    zoom: int = 14
    max_images: Optional[int] = 200
    save: Optional[bool] = True

# ── Mapillary helpers ─────────────────────────────────────────────────────
def tile_to_bbox(x: int, y: int, z: int):
    """Return (west, south, east, north) bbox for a tile"""
    n = 2 ** z
    lon_w = x / n * 360 - 180
    lon_e = (x + 1) / n * 360 - 180
    lat_n = math.degrees(math.atan(math.sinh(math.pi * (1 - 2*y/n))))
    lat_s = math.degrees(math.atan(math.sinh(math.pi * (1 - 2*(y+1)/n))))
    return lon_w, lat_s, lon_e, lat_n

async def fetch_mapillary_images(bbox_str: str, limit: int = 200) -> list[dict]:
    """Fetch image metadata from Mapillary within a bbox string 'west,south,east,north'"""
    if not MAPILLARY_TOKEN:
        return []
    url = (
        f"{MAPILLARY_BASE}/images"
        f"?access_token={MAPILLARY_TOKEN}"
        f"&fields=id,geometry,sequence_id,thumb_1024_url"
        f"&bbox={bbox_str}"
        f"&limit={limit}"
    )
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get(url)
            if r.status_code == 200:
                data = r.json()
                return data.get("data", [])
            else:
                log.warning(f"Mapillary API {r.status_code}: {r.text[:200]}")
                return []
        except Exception as e:
            log.error(f"Mapillary fetch error: {e}")
            return []

def run_inference(image_url: str) -> list[dict]:
    """Run YOLO inference on an image URL. Returns list of detections."""
    if model is None:
        return []
    try:
        results = model.predict(
            source=image_url,
            conf=CONFIDENCE,
            verbose=False,
            stream=False
        )
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                # When custom model is loaded, map class names properly
                # Placeholder model: class names will be COCO — filter to pole-like objects
                conf = float(box.conf[0])
                detections.append({
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": box.xyxy[0].tolist()
                })
        return detections
    except Exception as e:
        log.error(f"Inference error on {image_url}: {e}")
        return []

def is_fosc_class(cls_name: str) -> bool:
    """
    When custom model is loaded: 'fosc', 'snowshoe', 'splice_closure' are valid.
    When placeholder (COCO) model: we pass through everything and flag as placeholder.
    """
    if model_status == "placeholder":
        return True  # placeholder — caller sees model_status
    fosc_classes = {"fosc", "snowshoe", "splice_closure", "fiber_splice", "aerial_closure"}
    return cls_name.lower() in fosc_classes

# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_status": model_status, "version": model_version}

@app.get("/status")
def status():
    return {
        "model_status": model_status,
        "model_version": model_version,
        "model_path": MODEL_PATH,
        "confidence_threshold": CONFIDENCE,
        "detections_stored": len(detections_store),
        "mapillary_token_set": bool(MAPILLARY_TOKEN),
        "placeholder_note": (
            "Custom FOSC model not yet loaded. Deploy fosc_v1.pt to MODEL_PATH to activate. "
            "Provide labeled FOSC/snowshoe images to train the custom model."
        ) if model_status == "placeholder" else None
    }

@app.post("/detect")
async def detect(req: DetectRequest):
    """Run inference on a single image URL"""
    t0 = time.time()
    raw = run_inference(req.image_url)

    fosc_hits = [d for d in raw if is_fosc_class(d["class"])]
    saved = []

    if req.save and fosc_hits and req.lat is not None and req.lon is not None:
        for hit in fosc_hits:
            feat = add_detection(
                lat=req.lat,
                lon=req.lon,
                confidence=hit["confidence"],
                cls=hit["class"],
                image_id=req.image_id or "",
                image_url=req.image_url,
                sequence_id=req.sequence_id or ""
            )
            saved.append(feat)

    return {
        "image_url": req.image_url,
        "inference_ms": round((time.time() - t0) * 1000),
        "model_status": model_status,
        "raw_detections": len(raw),
        "fosc_detections": len(fosc_hits),
        "detections": fosc_hits,
        "saved": len(saved),
        "warning": "Placeholder model active — retrain with labeled FOSC images" if model_status == "placeholder" else None
    }

@app.post("/detect/batch")
async def detect_batch(req: BatchDetectRequest):
    """Run inference on multiple images"""
    results = []
    for img in req.images:
        img.save = req.save
        r = await detect(img)
        results.append(r)
        await asyncio.sleep(0.05)  # brief yield

    total_fosc = sum(r["fosc_detections"] for r in results)
    return {
        "images_processed": len(results),
        "total_fosc_detections": total_fosc,
        "model_status": model_status,
        "results": results
    }

@app.post("/scan/tile")
async def scan_tile(req: TileScanRequest, background_tasks: BackgroundTasks):
    """
    Scan all Mapillary images within a zoom-14 tile.
    Runs inference on each image, saves FOSC detections.
    Returns immediately with job_id; check /results for output.
    """
    lon_w, lat_s, lon_e, lat_n = tile_to_bbox(req.tile_x, req.tile_y, req.zoom)
    bbox_str = f"{lon_w},{lat_s},{lon_e},{lat_n}"

    images = await fetch_mapillary_images(bbox_str, limit=req.max_images)

    if not images:
        return {
            "tile": f"{req.zoom}/{req.tile_x}/{req.tile_y}",
            "bbox": bbox_str,
            "images_found": 0,
            "message": "No Mapillary images found in this tile"
        }

    # Run in background
    job_id = f"tile_{req.tile_x}_{req.tile_y}_{req.zoom}_{int(time.time())}"

    async def process():
        count = 0
        fosc_count = 0
        for img in images:
            img_id = img.get("id", "")
            thumb = img.get("thumb_1024_url", "")
            coords = img.get("geometry", {}).get("coordinates", [None, None])
            lon_img, lat_img = (coords[0], coords[1]) if len(coords) == 2 else (None, None)
            seq_id = img.get("sequence_id", "")

            if not thumb:
                continue

            raw = run_inference(thumb)
            fosc_hits = [d for d in raw if is_fosc_class(d["class"])]

            if fosc_hits and lat_img and req.save:
                for hit in fosc_hits:
                    add_detection(
                        lat=lat_img,
                        lon=lon_img,
                        confidence=hit["confidence"],
                        cls=hit["class"],
                        image_id=img_id,
                        image_url=thumb,
                        sequence_id=seq_id
                    )
                    fosc_count += 1

            count += 1
            await asyncio.sleep(0.1)

        log.info(f"[{job_id}] Done: {count} images, {fosc_count} FOSC detections")

    background_tasks.add_task(process)

    return {
        "job_id": job_id,
        "tile": f"{req.zoom}/{req.tile_x}/{req.tile_y}",
        "bbox": bbox_str,
        "images_queued": len(images),
        "status": "processing",
        "message": "Tile scan running in background. Poll /results for detections."
    }

@app.get("/results")
def get_results():
    """Return all accumulated FOSC detections as GeoJSON FeatureCollection"""
    return {
        "type": "FeatureCollection",
        "metadata": {
            "count": len(detections_store),
            "model_status": model_status,
            "model_version": model_version,
            "confidence_threshold": CONFIDENCE,
            "ring_radius_mi": 1.5,
            "note": "Each detection has a 1.5-mile prospecting perimeter for adjacent farmland parcels"
        },
        "features": detections_store
    }

@app.delete("/results")
def clear_results():
    """Clear all stored detections"""
    count = len(detections_store)
    detections_store.clear()
    if RESULTS_PATH.exists():
        RESULTS_PATH.unlink()
    return {"cleared": count}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
