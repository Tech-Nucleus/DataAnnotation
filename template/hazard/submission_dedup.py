"""Anti-plagiarism helpers for annotation-only miner submissions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence, Tuple

from template.protocol import PerImageAnnotationItem


def fingerprint_annotation_items(items: Sequence[PerImageAnnotationItem]) -> str:
    """Stable hash over sorted annotation rows."""

    rows = []
    for it in sorted(
        items,
        key=lambda x: (x.hazard_class.lower(), tuple(x.bounding_box)),
    ):
        rows.append(
            {
                "hazard_class": it.hazard_class.strip().lower(),
                "bounding_box": [float(b) for b in it.bounding_box],
            }
        )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def full_submission_fingerprint(
    records: Mapping[str, Sequence[PerImageAnnotationItem]],
) -> str:
    """Hash of per-image fingerprints for the whole round payload."""

    parts = {iid: fingerprint_annotation_items(items) for iid, items in sorted(records.items())}
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class AnnotationDuplicateTracker:
    """Within-round index: first UID wins for identical annotation structure per image."""

    _image_fingerprint_to_uid: Dict[str, Dict[str, int]] = field(default_factory=dict)
    _full_fp_to_uid: Dict[str, int] = field(default_factory=dict)

    def check_and_register(
        self,
        uid: int,
        records: Mapping[str, Sequence[PerImageAnnotationItem]],
    ) -> Tuple[bool, str]:
        """Return (ok, reason). ``ok`` is False if this UID duplicates an earlier one."""

        full_fp = full_submission_fingerprint(records)
        prior_full = self._full_fp_to_uid.get(full_fp)
        if prior_full is not None and prior_full != uid:
            return False, (
                f"annotations payload matches uid {prior_full} (full-submission fingerprint)"
            )

        for image_id, items in records.items():
            fp = fingerprint_annotation_items(items)
            bucket = self._image_fingerprint_to_uid.get(image_id, {})
            owner = bucket.get(fp)
            if owner is not None and owner != uid:
                return False, (
                    f"duplicate annotation structure on image_id={image_id} "
                    f"(first uid={owner})"
                )

        if prior_full is None:
            self._full_fp_to_uid[full_fp] = uid
        for image_id, items in records.items():
            fp = fingerprint_annotation_items(items)
            bucket = self._image_fingerprint_to_uid.setdefault(image_id, {})
            if fp not in bucket:
                bucket[fp] = uid
        return True, ""
