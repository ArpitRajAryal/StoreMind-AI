# StoreMind AI — Retail Inventory & Incident Prototype

StoreMind AI is a computer vision and retail-operations prototype exploring how shelf observations, inventory records, point-of-sale events, and camera activity could be combined to support store staff.

The current prototype demonstrates:

- Shelf-state monitoring
- Inventory discrepancy detection
- Restock and reorder task generation
- POS and visible-product-movement reconciliation
- Human review of unmatched movement events
- Optional ResNet18 transfer learning for shelf-state classification
- Optional preservation of short video clips for staff review

> **Important:** The operational records in the current notebook are simulated.  
> The prototype does not determine theft, automatically alter inventory, or contact police or other third parties.

---

## Concept

A retail store may have several sources of information describing what is happening to a product:

- Inventory management systems
- Point-of-sale transactions
- Shelf cameras
- Staff observations
- Backroom stock checks
- Product movement events

StoreMind explores how these signals could be reconciled into useful staff tasks rather than treated independently.

For example:

    Camera observes empty shelf
            +
    Inventory system reports stock available
            ↓
    CHECK_SHELF_AND_BACKROOM

Or:

    Visible product movement
            +
    No matching POS transaction within time window
            ↓
    REVIEW_UNMATCHED_MOVEMENT
            ↓
    Human review

An unmatched event is **not automatically classified as theft**.

---

## Current Prototype

### 1. Simulated Retail Data

The notebook currently uses simulated records representing:

- SKUs
- Products
- On-hand stock
- Verified backroom stock
- Shelf observations
- Observation confidence
- Product movement events
- POS transactions

These records are used only to demonstrate the system architecture and decision logic.

---

## 2. Shelf Task Generation

Shelf observations are reconciled with inventory information to generate staff actions.

Example actions include:

| Action | Meaning |
|---|---|
| `OK` | No action required |
| `VERIFY_OBSERVATION` | Camera confidence is too low |
| `PROPOSE_REORDER` | Shelf is empty and system stock is zero |
| `RESTOCK_FROM_BACKROOM` | Shelf stock is low and verified backroom stock exists |
| `CHECK_SHELF_AND_BACKROOM` | Shelf appears empty but the inventory system reports stock |
| `VERIFY_LOW_SHELF` | Shelf appears below its target level |

The system creates tasks rather than automatically changing inventory records.

---

## 3. POS and Product-Movement Reconciliation

The prototype also compares visible product movements against POS transactions.

A movement that matches a sale within a configurable time window is recorded as:

    MATCHED_POS

A movement without a matching transaction becomes:

    REVIEW_UNMATCHED_MOVEMENT

The system then asks a staff member to verify possible explanations such as:

- Delayed POS transactions
- Returns
- Staff transfers
- Restocking activity
- Recording errors
- Other legitimate activity

This approach keeps incident decisions under human review.

---

## 4. Staff Dashboard

The prototype creates:

- A shelf-management task queue
- An incident-review queue
- A simple visualization of shelf task distribution

The current dashboard uses simulated SKUs and should not be interpreted as model accuracy or commercial performance data.

---

## 5. Shelf-State Computer Vision

The notebook includes an optional real-image training pipeline.

A shelf image can be classified into three states:

    empty
    low
    stocked

The current approach uses **ResNet18 transfer learning** with ImageNet pretrained weights.

The pretrained feature extractor is frozen and the final classification layer is replaced with a three-class output layer.

### Expected Image Manifest

Real training data should use a CSV containing:

    image_path
    sku
    shelf_id
    capture_session
    label

Example:

    image_path,sku,shelf_id,capture_session,label
    images/img001.jpg,SKU001,A1,session_01,stocked
    images/img002.jpg,SKU001,A1,session_01,low
    images/img003.jpg,SKU001,A1,session_02,empty

`capture_session` is used to help prevent near-duplicate frames from appearing in both training and validation data.

---

## Important Computer Vision Limitation

The shelf-state classifier answers:

> "Does this shelf area appear empty, low, or stocked?"

It does **not** independently answer:

> "Which SKU is this?"

or:

> "Exactly how many units are present?"

A production StoreMind system would therefore require additional computer-vision components such as:

    Camera
      ↓
    SKU / Product Detection
      ↓
    Product Tracking / Counting
      ↓
    Shelf-State Analysis
      ↓
    Inventory + POS Reconciliation
      ↓
    Staff Task / Human Review

---

## 6. Video Review

The notebook includes an optional helper using `ffmpeg` that can preserve a short section of video around an event.

For example:

    10 seconds before event
            +
    event
            +
    15 seconds after event

The resulting clip can then be presented to an authorised staff reviewer.

The prototype does not upload, transmit, or automatically distribute footage.

---

## 7. Human Review

StoreMind is designed around human verification.

Reviewers can record outcomes such as:

- `sale_found`
- `staff_transfer`
- `return_or_restock`
- `unclear`
- `other`

This avoids automatically inferring wrongdoing from incomplete data.

---

# Potential Datasets and Research Resources

Several public datasets could support different parts of a future StoreMind system.

None of these datasets directly provides all the data required by the prototype, so they are best treated as research, experimentation, or pre-training resources.

---

## Out-of-Stock Detection — Roboflow

[Out-of-Stock Detection Dataset](https://universe.roboflow.com/empty-space-detection-capstone/out-of-stock-detection?utm_source=chatgpt.com)

**Potential use:** Detect empty spaces on retail shelves.

This dataset contains an `empty` detection class and could be useful for experimenting with locating empty shelf regions.

### Limitation

The current StoreMind image classifier uses three shelf-state classes:

    empty
    low
    stocked

Because this dataset only detects empty spaces, it cannot directly train the notebook's three-class `empty / low / stocked` classifier without additional labelled data.

---

## SKU110K

[SKU110K — GitHub](https://github.com/eg4000/SKU110K_CVPR19)

**Potential use:** Detect and count visible products on densely packed retail shelves.

SKU110K is useful for experimenting with object detection in highly crowded retail shelf environments.

### Limitation

Its product bounding boxes use a generic product class rather than identifying individual store SKUs.

StoreMind would ultimately need SKU-aware recognition to associate visual detections with inventory and POS records.

The dataset authors also restrict the data to academic, non-commercial use, so its licence must be considered before commercial use.

---

## RetailAction

[RetailAction — Hugging Face](https://huggingface.co/datasets/standard-cognition/RetailAction?utm_source=chatgpt.com)

**Potential use:** Explore video-based recognition of retail interactions such as:

    take
    put
    touch

This could support research into product-movement detection and customer-item interaction.

### Limitation

RetailAction is significantly larger and more complex than the current prototype, containing approximately 21,000 annotated multi-view samples.

It does not provide matching StoreMind-style POS transaction records, so transaction reconciliation would still require a separate data source or custom dataset.

Its separate dataset licence should be reviewed before use.

---

## FreshRetailNet-50K

[FreshRetailNet-50K — Hugging Face](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)

**Potential use:** Experiment with sales, inventory, and stockout modelling.

This dataset could support the non-visual side of StoreMind, including:

- Stockout prediction
- Inventory modelling
- Demand analysis
- Replenishment research

### Limitation

FreshRetailNet-50K primarily contains tabular sales and stock information.

It does not contain shelf photographs directly linked to those records, so it cannot independently train the computer-vision component of StoreMind.

---

## How These Datasets Could Fit Together

A future StoreMind research pipeline could potentially combine ideas from several dataset types:

| Dataset / Data Source | StoreMind Component |
|---|---|
| Out-of-Stock Detection | Empty shelf detection |
| SKU110K | Product detection and counting |
| RetailAction | Take / put / touch detection |
| FreshRetailNet-50K | Inventory and stockout modelling |
| Custom store shelf images | Empty / low / stocked classifier |
| Store POS records | Transaction reconciliation |
| Store inventory records | On-hand stock reconciliation |

No single public dataset currently represents the complete StoreMind problem.

A realistic deployment would therefore likely require a custom multimodal dataset linking:

    Camera observations
    +
    SKU identities
    +
    Product movements
    +
    Shelf state
    +
    Inventory
    +
    POS transactions
    +
    Timestamps

---

# Technology

The current prototype uses:

- Python
- Pandas
- NumPy
- Matplotlib
- PyTorch
- TorchVision
- ResNet18
- scikit-learn
- Pillow
- FFmpeg

---

# Installation

Clone the repository:

    git clone https://github.com/YOUR-USERNAME/StoreMind-AI.git
    cd StoreMind-AI

Install the main Python dependencies:

    pip install numpy pandas matplotlib torch torchvision scikit-learn pillow

FFmpeg is also required if you want to use the optional video-clipping functionality.

---

# Running the Prototype

Open:

    StoreMind_Inventory_Incident_Prototype.ipynb

using Jupyter Notebook, JupyterLab, VS Code, or another compatible notebook environment.

The simulated operational prototype can run without any external retail dataset.

The computer-vision training section is skipped automatically if no real image manifest is available.

---

# Current Limitations

This repository is currently a prototype rather than a production retail surveillance system.

It has not yet been validated on real store operations.

Important limitations include:

- Operational records are simulated
- No production SKU detector is implemented
- No real-time camera stream integration
- No live POS integration
- No live inventory-system integration
- No validated unit-counting model
- No production authentication or access controls
- No real-world accuracy or business-impact measurements
- No automatic incident determination

Before deployment, the system would also require appropriate privacy, security, data-retention, access-control, and legal review.

---

# Future Development

Planned areas for further development include:

1. Collect labelled shelf images across different cameras, stores, lighting conditions, and time periods.
2. Train and evaluate the `empty / low / stocked` shelf classifier.
3. Introduce SKU-aware object detection.
4. Add product tracking and visible-unit counting.
5. Integrate read-only inventory and POS data.
6. Synchronise timestamps between camera and transaction systems.
7. Build a proper staff operations dashboard.
8. Add reviewer audit logs and evidence-management controls.
9. Evaluate alert precision, recall, false-positive rates, and staff workload.
10. Test the complete pipeline in a controlled retail pilot.

---

## Status

**Prototype / Research Project**

The project currently demonstrates the architecture and core reconciliation logic required for a larger retail computer-vision system.
