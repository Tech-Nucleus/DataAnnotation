# Subnet Glossary

This glossary defines key terms and concepts used across the decentralized annotation subnet codebase, documentation, and protocols.

---

### **A**
* **Adoption Bonus**: An incentive mechanism rewarding miners whose submitted annotations closely match the final aggregated consensus on Unlabeled images.
* **AnnotationFidelityScorer**: The validator component responsible for scoring miner annotation submissions against the ground truth of Golden Set images using IoU and class overlap.
* **AnnotationTask**: The Bittensor synapse protocol containing task information, a challenge nonce, and a list of image URLs that miners need to annotate.

### **B**
* **Bayesian Fusion**: A probabilistic consensus algorithm that aggregates multiple noisy annotations on unlabeled images into a high-confidence ground truth. Votes are weighted based on each miner's historical reliability on Golden Set images.
* **Bounding Box (BBox)**: Spatial coordinates defining the location of an object. Formatted as `[x_min, y_min, x_max, y_max]` representing bounding box corners in pixels.

### **C**
* **Camouflage Pipeline**: A series of security measures (EXIF metadata stripping, SHA-256 filename hashing, network request jitter) designed to prevent miners from identifying secret Golden Set images.
* **ConsensusScorer**: The component that computes spatial overlap and category agreement among miners on Unlabeled images to generate the consensus.

### **D**
* **DatasetAssembler**: The validator subsystem that coordinates the fusion of miner submissions, resolves spatial/class conflicts, maps classes to severities, and exports the finalized commercial dataset.
* **Dual-Flywheel Reward**: The incentive distribution model that combines Golden Set accuracy (Fidelity) and Unlabeled consensus agreement (Adoption) to compute on-chain miner weights.

### **G**
* **Golden Set (Golden Images)**: A secret, curated set of images with highly accurate, human-verified labels. Used by validators as a holdout to measure miner reliability.
* **Golden holdout ratio**: The fraction of images in an annotation step that are randomly drawn from the Golden Set (defaults to 10%, configured via `--neuron.flywheel_golden_ratio` or environment variable `VALIDATOR_GOLDEN_RATIO`).

### **I**
* **Injection Plan**: The step-by-step arrangement of Golden and Unlabeled images sent in a single annotation round.
* **Intersection over Union (IoU)**: A geometric metric measuring the overlap of two bounding boxes, computed as the area of intersection divided by the area of union.

### **S**
* **Severity Tier**: The danger level associated with a hazard (Low, Medium, High, Critical). Severities are assigned deterministically server-side based on the hazard class, completely bypassing miner input.
* **Sybil Attack**: An attack vector where a single malicious actor runs multiple miner instances to manipulate consensus. Mitigated in this subnet through fidelity-weighted aggregation.

### **U**
* **Unlabeled Images**: Images in the annotation pool that do not have ground truth labels. Miners label these images, and their annotations are fused to compile the final dataset.
