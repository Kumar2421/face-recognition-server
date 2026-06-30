from pydantic import BaseModel, Field, AliasChoices
from typing import Any, List, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class FaceIndex:
    subject_ids: List[str]
    mean_embeddings: np.ndarray

class FaceSearchRequest(BaseModel):
    image_b64: str = Field(..., validation_alias=AliasChoices("image_b64", "image", "images_b64"))
    camera: Optional[str] = None
    reid_id: Optional[str] = None
    frame_time: Optional[float] = None

class FaceSearchResponse(BaseModel):
    subject_id: str
    similarity: float
    meta: Optional[dict] = None

class FaceAddRequest(BaseModel):
    subject_id: str
    images_b64: List[str] = []
    image_urls: List[str] = []
    branch: Optional[str] = None
    created_at: Optional[str] = None
    ts: Optional[float] = None

class GroupCreateRequest(BaseModel):
    group_id: str
    name: Optional[str] = None
    meta: Optional[dict] = None

class GroupResponse(BaseModel):
    group_id: str
    name: Optional[str] = None
    meta: Optional[dict] = None

class GroupListResponse(BaseModel):
    groups: List[GroupResponse]

class BranchCreateRequest(BaseModel):
    branch_id: str
    name: Optional[str] = None
    meta: Optional[dict] = None

class BranchResponse(BaseModel):
    branch_id: str
    name: Optional[str] = None
    meta: Optional[dict] = None
    subject_count: Optional[int] = 0
    enrollment_count: Optional[int] = 0

class BranchListResponse(BaseModel):
    branches: List[BranchResponse]

class FaceAddResponse(BaseModel):
    subject_id: str
    num_images: int
    num_embedded: int
    embedding_dim: Optional[int] = None
    meta: Optional[dict] = None

class FaceSearchTopKRequest(BaseModel):
    image_b64: str = Field(..., validation_alias=AliasChoices("image_b64", "image", "images_b64"))
    top_k: int = 5
    branch: Optional[str] = None
    day: Optional[str] = None
    from_day: Optional[str] = None
    to_day: Optional[str] = None
    since_ts: Optional[float] = None
    until_ts: Optional[float] = None

class FaceSearchTopKItem(BaseModel):
    subject_id: str
    similarity: float
    point_id: str
    image_id: Optional[str] = None
    thumb_path: Optional[str] = None

class FaceSearchTopKResponse(BaseModel):
    results: List[FaceSearchTopKItem]
    query_thumb_path: Optional[str] = None

class FaceRecognizeRequest(BaseModel):
    image_b64: str = Field(..., validation_alias=AliasChoices("image_b64", "image", "images_b64"))
    top_k: int = 5
    min_similarity: Optional[float] = None
    branch: Optional[str] = None
    day: Optional[str] = None
    from_day: Optional[str] = None
    to_day: Optional[str] = None
    since_ts: Optional[float] = None
    until_ts: Optional[float] = None

class FaceRecognizeResponse(BaseModel):
    matched: bool
    subject_id: Optional[str] = None
    similarity: Optional[float] = None
    results: List[FaceSearchTopKItem] = []
    meta: Optional[dict] = None

class FaceQualityResult(BaseModel):
    ok: bool
    quality: Optional[dict] = None
    det_score: Optional[float] = None
    bbox: Optional[List[float]] = None

class QualityCheckResponse(BaseModel):
    ok: bool
    total_quality: Optional[str] = None
    faces: List[FaceQualityResult] = []
    annotated_image: Optional[str] = None
    timing: Optional[dict] = None

class RecognitionEventResponse(BaseModel):
    event_id: str
    ts: float
    camera: str = ""
    branch: Optional[str] = None
    source_path: str
    decision: str
    subject_id: Optional[str] = None
    similarity: Optional[float] = None
    processing_ms: Optional[int] = None
    model_ms: Optional[int] = None
    rejected_reason: Optional[str] = None
    bbox: Optional[List[float]] = None
    det_score: Optional[float] = None
    image_path: str
    thumb_path: str
    image_saved_at: Optional[float] = None
    meta: Optional[dict] = None
    feedback_label: Optional[str] = None
    feedback_note: Optional[str] = None
    feedback_updated_at: Optional[float] = None

class RecognitionEventsListResponse(BaseModel):
    items: List[RecognitionEventResponse]
    cursor: Optional[float] = None

class SearchEventResponse(BaseModel):
    event_id: str
    ts: float
    query_image_path: str
    query_thumb_path: str
    top_subject_id: Optional[str] = None
    top_similarity: Optional[float] = None
    results: List[FaceSearchTopKItem] = []
    meta: Optional[dict] = None

class SearchEventsListResponse(BaseModel):
    items: List[SearchEventResponse]
    cursor: Optional[float] = None

class SearchEventsStatsResponse(BaseModel):
    match: int
    no_match: int
    total: int

class RecognitionStatsResponse(BaseModel):
    total: int
    match: int
    no_match: int
    rejection: int
    unique_matches: int = 0
    by_camera: dict

class EventFeedbackRequest(BaseModel):
    label: Optional[str] = None
    note: Optional[str] = None

class EventFeedbackResponse(BaseModel):
    event_id: str
    updated: bool
    feedback_label: Optional[str] = None
    feedback_note: Optional[str] = None
    feedback_updated_at: Optional[float] = None

class FeedbackStatsResponse(BaseModel):
    total: int
    labeled: int
    unlabeled: int
    tp: int
    fp: int
    fn: int
    ignore: int
    fp_rate_match: Optional[float] = None
    by_decision: dict

class RecognitionFetchRequest(BaseModel):
    image_b64: Optional[str] = Field(None, validation_alias=AliasChoices("image_b64", "image", "images_b64"))
    image_url: Optional[str] = Field(None, validation_alias=AliasChoices("image_url", "url"))
    camera: str
    source_path: Optional[str] = None
    ts: Optional[float] = None
    top_k: int = 5
    min_similarity: Optional[float] = None
    process_all_faces: bool = False

class FaceCompareRequest(BaseModel):
    image1_b64: Optional[str] = None
    image2_b64: Optional[str] = None
    image1_url: Optional[str] = None
    image2_url: Optional[str] = None

class FaceCompareResponse(BaseModel):
    similarity: float
    match: bool
    confidence: str
    meta: Optional[dict] = None

class FaceSubjectsResponse(BaseModel):
    subjects: List[str]

class FaceDeleteSubjectResponse(BaseModel):
    subject_id: str
    deleted: bool

class PrivacyExtractRequest(BaseModel):
    image_b64: str = Field(..., validation_alias=AliasChoices("image_b64", "image", "images_b64"))
    recognition: bool = False
    top_k: int = Field(1, validation_alias=AliasChoices("top_k", "top_n"))
    branch: Optional[str] = None
    day: Optional[str] = None
    from_day: Optional[str] = None
    to_day: Optional[str] = None
    since_ts: Optional[float] = None
    until_ts: Optional[float] = None

class PrivacyCropItem(BaseModel):
    bbox: Optional[List[float]] = None
    quality: Optional[dict] = None
    image_b64: str
    recognition: Optional[FaceRecognizeResponse] = None

class PrivacyExtractResponse(BaseModel):
    results: List[PrivacyCropItem]

class PrivacyBlurRequest(BaseModel):
    # base64 string or http(s) URL
    image_b64: str = Field(..., validation_alias=AliasChoices("image_b64", "image", "images_b64", "url"))
    # [x1, y1, x2, y2] of the face to KEEP unblurred
    bbox: Optional[List[float]] = None
    # If true, blur every detected face (including the bbox target)
    blur_all: bool = False

class PrivacyBlurResponse(BaseModel):
    image_b64: str
    faces_total: int
    blurred_count: int
    kept_bbox: Optional[List[float]] = None

class KeyCreateRequest(BaseModel):
    name: Optional[str] = None
    # Optional explicit key value; if omitted the server generates one.
    api_key: Optional[str] = Field(None, validation_alias=AliasChoices("api_key", "key"))

class KeyInfo(BaseModel):
    key_id: str
    name: Optional[str] = None
    access_key: str
    created_at: Optional[str] = None
    active: bool = True
    # Masked key for listing; raw value returned only once on creation.
    api_key: Optional[str] = None
    api_key_masked: Optional[str] = None

class KeyListResponse(BaseModel):
    keys: List[KeyInfo]

class KeyDeleteResponse(BaseModel):
    key_id: str
    deleted: bool
