# CanopyMRV System Context & Architecture Summary

## 1. Documentation Summary

### 1.1 `docs/ARCHITECTURE.md`
- **Subnet Objective:** Decentralized data annotation subnet (NetUID 498 on Bittensor Finney testnet) building commercial-grade object detection and delineation datasets for Climate MRV (Monitoring, Reporting, and Verification).
- **Core Pipeline:**
  - **Miners:** High-throughput annotation engines running local vision models (YOLOv8, self-hosted servers, or VLMs) to detect vegetation and delineate tree canopies, forest hazards, and land-cover classes. Upload JSON annotations to Cloudflare R2 and return signed presigned URIs to validators.
  - **Validators:** Manage image corpus (mix of secret Golden Set images held locally and unlabeled production images), camouflage images (strip EXIF/metadata, SHA-256 hash filenames, inject network jitter), dispatch tasks via `AnnotationTask` synapse, score miners on golden ground truth, fuse miner submissions using Bayesian aggregation, and export commercial datasets.
- **Security & Anti-Gaming:**
  - Metadata stripping and opaque SHA-256 hashing prevent miners from distinguishing golden images by name or EXIF.
  - Micro-delay jitter prevents timing side-channels.
  - Bayesian fusion weights unlabeled predictions by historical golden fidelity, neutralizing Sybils.
- **Dual-Flywheel Incentive:**
  - `Total Reward = alpha * Fidelity_Score + (1 - alpha) * Adoption_Bonus` (default `alpha = 0.7`).
  - Fidelity is computed strictly on secret Golden Set images using IoU match, class match, hallucination penalty, and net weight agreement.
  - Adoption bonus rewards consensus agreement on unlabeled images.

### 1.2 `MINER.md`
- **Miner Role:**
  - Receives unlabeled Sentinel-2 RGB chips (1024×1024 px).
  - Runs vision detection/segmentation to detect tree canopies, clusters, and forest hazards:
    `individual_tree`, `group_of_trees`, `tree`, `intact_forest`, `degraded_forest`, `deforestation`, `regrowth`, `plantation`, `wetland`, `water`, `agriculture`, `urban`, `fire_scar`, `bare_land`, `cloud`.
  - Produces annotations with flexible geometries: bounding boxes `[x1, y1, x2, y2]`, polygons `[[x1, y1], [x2, y2], ...]`, area, weight, net_weight, tree_coverage_percentage.
  - Uploads `annotations.json` conforming to schema `annotations.v1` to Cloudflare R2 under `miners/annotations/<task_id>/annotations.json`.
- **Supported Backends:**
  - `self_hosted`: Local FastAPI server (`server.py` / `scripts/reference_self_hosted_server.py`) exposing `/train`, `/train/status/{job_id}`, and `/infer`.
  - `yolo_local`: Local GPU fine-tuning of YOLOv8 (`yolov8n.pt`).
  - `openai_vision`: OpenAI GPT-4o vision backend.

### 1.3 `VALIDATOR.md`
- **Validator Role:**
  - Loads Climate MRV corpus: 500 Sentinel-2 RGB raw chips (`climate_raw_000.jpg` to `499.jpg`) from `data/climate_mrv/samples/raw/` (or GEE streaming).
  - Loads 100 private ground-truth golden chips (`climate_tree_000.jpg` to `099.jpg`) with ground truth from `golden_labels.json` stored strictly locally on the validator's machine (never published to R2).
  - Dispatches `AnnotationTask` synapses with camouflaged image URLs (presigned R2 or local).
  - Evaluates miner responses on Golden Set images using `AnnotationFidelityScorer`.
  - Fuses predictions on unlabeled images and exports commercial dataset to R2.

---

## 2. Current Miner Inference Entrypoint

- **Entrypoint:** `neurons/miner.py` → `ModelTrainingAnnotationEngine` (for `self_hosted`) → POST to `http://localhost:8081/infer` served by `server.py` (`scripts/reference_self_hosted_server.py`).
- **Inference Engine:** `EcologicalVisionEngine` in `template/miner/ecological_reasoning.py`.
- **Model Weights Loaded:**
  1. **Detector:** `models/tree_detection.pt` (specialized YOLOv8 tree detection model, fallback to `yolov8s-worldv2.pt` or `yolov8n.pt`).
  2. **Semantic Segmenter:** `models/tcd-segformer-mit-b2` (SegFormer model for tree canopy delineation, loaded via HuggingFace `AutoImageProcessor` and `SegformerForSemanticSegmentation`).
  3. **Multimodal VLM:** `models/qwen2.5-vl-3b` (Qwen2.5-VL-3B-Instruct loaded via `transformers` with `Qwen2_5_VLForConditionalGeneration` in `bfloat16`).
- **Prompt Used (Qwen2.5-VL):**
  - Prompt: `'Respond in compact JSON only: {"landscape": "urban"|"agricultural"|"forest"|"coastal_wetland", "has_mangroves": true|false, "has_fields": true|false, "has_trees": true|false}'`
  - Used for macro-landscape scene reasoning to set ecological context flags (`has_mangroves`, `has_fields`).
- **Post-Processing & Polygon Extraction:**
  - **Bboxes:** Predicted by YOLOv8 (`results[0].boxes`). If detector produces zero boxes, SegFormer semantic mask contours (area >= 25) are converted to bounding box proposals. Fallback: grid-spectral scan.
  - **Classification:** Bio-spectral & morphological reasoning (`_classify_crop`) classifies crops using color indices (ExG = 2G - R - B, NDVI approximation, aspect ratio, size) combined with Qwen landscape flags (`has_mangroves`).
  - **Polygon Generation:**
    - If SegFormer mask is present: contours are extracted from `seg_mask[y1:y2, x1:x2]` using `cv2.findContours` and approximated via `cv2.approxPolyDP(cnt, epsilon=0.015 * arcLength, closed=True)`.
    - Fallback: OpenCV ExG Otsu thresholding or `cv2.minAreaRect` oriented bounding box (OBB).
    - Final fallback: 4-point rectangle `[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]`.
  - **Field Parcel Extraction:** Morphological filtering on Excess Green (ExG > 10.0, low local variance, gray in [40, 210]) to extract agricultural field polygons.
  - **Ecological Carbon Weight:** `poly_area / img_area * multiplier` where `mangrove: 3.5`, `dense_tree: 1.8`, `ordinary_tree: 1.0`, `field: 0.7`, `plant: 0.4`.
- **Polygon Format Emitted:**
  - List of 2D vertex coordinates: `[[x1, y1], [x2, y2], ...]` in absolute pixel coordinates (NOT GeoJSON geometry object, NOT COCO RLE, NOT oriented bbox format `[cx, cy, w, h, angle]`, but explicit polygon contour vertices).

---

## 3. Validator Scoring Function

Implemented in `template/hazard/annotation_eval.py` (`AnnotationFidelityScorer`):

- **IoU Definition:**
  - Axis-aligned 2D bounding box IoU:
    $$\text{IoU} = \frac{\text{area}(\text{box}_A \cap \text{box}_B)}{\text{area}(\text{box}_A) + \text{area}(\text{box}_B) - \text{area}(\text{box}_A \cap \text{box}_B)}$$
  - Computed strictly on `[x1, y1, x2, y2]` bounding boxes. **Polygon/mask IoU is not evaluated by the validator.**
- **Matching Rule:**
  - Greedy 1-to-1 matching: For each ground truth box, searches all unused miner items for the maximum IoU. If matched, the miner item index is marked as used. If no match exists, IoU = 0.0 and Class = 0.0.
- **Class / Species Fidelity:**
  - Evaluated via `_class_match_score`:
    - Normalized to canonical carbon class (`mangrove`, `dense_tree`, `ordinary_tree`, `field`, `plant`).
    - Exact class match: `1.0`.
    - Partial credit:
      - `{dense_tree, ordinary_tree}`: `0.75`.
      - `{plant, ordinary_tree, dense_tree}`: `0.50`.
      - Otherwise: `0.0`.
- **Hallucination Penalty:**
  - `hallucinated = len(miner_items) - len(used_miner_idx)`.
  - `rel_hallucinated = hallucinated / max(1, len(gt_annotations))`.
  - `penalty = 0.5 ** rel_hallucinated` (default base `hallucination_penalty = 0.5`).
- **Net Canopy Weight Agreement:**
  - `miner_net_weight = sum(item.weight)`.
  - `gt_net_weight = sum((gt_box_area / img_area) * multiplier)`.
  - `weight_diff = abs(miner_net_weight - gt_net_weight)`.
  - `net_weight_agreement = max(0.0, 1.0 - (weight_diff / max(gt_net_weight, 0.05)))`.
  - `coverage_factor = 0.70 + 0.30 * net_weight_agreement`.
- **Final Golden Fidelity Score:**
  - `fidelity_raw = 0.60 * mean_matched_iou + 0.40 * mean_matched_class`.
  - `fidelity = min(1.0, max(0.0, fidelity_raw * penalty * coverage_factor))`.
- **Confidence Handling:**
  - The scorer completely ignores miner confidence scores. No thresholding is applied by the validator. All submitted annotations are matched; extra low-confidence predictions simply incur the hallucination penalty.

---

## 4. Task Chip Format

- **Size:** 1024×1024 pixels (in `data/climate_mrv/samples/` and `MINER.md`), or 256×256 pixels in GEE streaming mode.
- **Bands:** 3-band RGB (Sentinel-2 L2A Harmonized: B4 Red, B3 Green, B2 Blue).
- **GSD (Ground Sample Distance):** 10 meters per pixel (Sentinel-2 native resolution).
- **CRS:** EPSG:4326 (WGS84) in GEE exports, but raw chips served to miners are standard JPEG images with metadata stripped.
- **Serving Mechanism:**
  - Stored in Cloudflare R2 under `dataset/raw/climate_raw_XXX.jpg`.
  - Validator serves presigned authenticated S3/R2 URLs inside `AnnotationTask.annotation_images`.
  - Miners download via HTTP GET, process locally, upload `annotations.json` to `miners/annotations/<task_id>/annotations.json`, and return presigned R2 URI.

---

## 5. What the Current Stack Actually Is

In plain language:

The current CanopyMRV miner stack is a **hybrid multi-stage pipeline**:
1. **Scene Context:** `Qwen2.5-VL-3B` runs macro-landscape ecological classification on the full image to determine landscape type and flags (`has_mangroves`, `has_fields`, `has_trees`).
2. **Canopy Segmentation:** `SegFormer-mit-b2` (pretrained on OAM-TCD `restor/tcd-segformer-mit-b2`) predicts a 2D semantic canopy mask.
3. **Bounding Box Proposals:** A fine-tuned `YOLOv8` detector (`models/tree_detection.pt` from `solafune/tree-detection`) predicts bounding boxes. If YOLO detects no boxes, connected components from the SegFormer mask are extracted as box proposals.
4. **Bio-Spectral Crop Refinement:** Each detected bounding box crop is classified into the carbon taxonomy (`mangrove`, `dense_tree`, `ordinary_tree`, `plant`) using OpenCV color indices (ExG, NDVI proxy) conditioned on Qwen's macro flags.
5. **Mask/Polygon Extraction:** For each box, if the SegFormer mask is active in the crop, `cv2.findContours` and `cv2.approxPolyDP` delineate the polygon contour. Otherwise, OpenCV ExG thresholding / OBB or bbox coordinates are used.
6. **Agricultural Field Parcels:** A separate OpenCV morphological filter on Excess Green (ExG) extracts field parcel polygons.
7. **Carbon Weight Calculation:** Area ratios multiplied by ecological carbon factors are summed to produce `net_weight` and `tree_coverage_percentage`.

### Component Table

| Component | Model ID | Task | Output Type | GSD Assumed |
|---|---|---|---|---|
| Macro-Landscape Reasoner | `Qwen2.5-VL-3B-Instruct` | Scene classification (landscape type, mangrove/field/tree flags) | JSON structured metadata | 10 m/px (Sentinel-2) |
| Canopy Segmenter | `restor/tcd-segformer-mit-b2` | Semantic canopy delineation | 2D pixel mask | 0.1 m/px (TCD native, but applied to 10 m S2 chips) |
| Crown Detector | `solafune/tree-detection` (YOLOv8) | Object detection for tree crowns | Bounding boxes `[x1, y1, x2, y2]` | ~0.1–0.5 m/px aerial/satellite |
| Crop Classifier | Heuristic Bio-spectral (OpenCV ExG/NDVI + Qwen scene context) | Vegetation classification into carbon taxonomy | Class label + confidence | 10 m/px |
| Polygon Extractor | SegFormer mask + `cv2.approxPolyDP` / `cv2.minAreaRect` | Flexible canopy contour delineation | List of `[x, y]` polygon vertices | 10 m/px |
| Field Extractor | OpenCV ExG + morphological filtering | Field parcel delineation | List of `[x, y]` polygon vertices | 10 m/px |
