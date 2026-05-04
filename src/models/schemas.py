from pydantic import BaseModel, Field, AliasChoices
from typing import Any, List, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class FaceIndex:
    subject_ids: List[str]
    mean_embeddings: np.ndarray

class FaceSearchRequest(BaseModel):
    image_b64: str
    camera: Optional[str] = None
    reid_id: Optional[str] = None
    frame_time: Optional[float] = None

class FaceSearchResponse(BaseModel):
    subject_id: str
    similarity: float
    meta: Optional[dict] = None

class FaceAddRequest(BaseModel):
    subject_id: str
    images_b64: List[str]
    group_id: Optional[str] = None

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

class FaceAddResponse(BaseModel):
    subject_id: str
    num_images: int
    num_embedded: int
    embedding_dim: Optional[int] = None
    meta: Optional[dict] = None

class FaceSearchTopKRequest(BaseModel):
    image_b64: str = Field(..., validation_alias=AliasChoices("image_b64", "image", "images_b64"))
    top_k: int = 5
    group_id: Optional[str] = None

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
    image_b64: str
    top_k: int = 5
    min_similarity: Optional[float] = None
    group_id: Optional[str] = None

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
    camera: str
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
    url: str
    camera: str
    source_path: Optional[str] = None
    ts: Optional[float] = None
    top_k: int = 5
    min_similarity: Optional[float] = None
    process_all_faces: bool = False

class FaceSubjectsResponse(BaseModel):
    subjects: List[str]

class FaceDeleteSubjectResponse(BaseModel):
    subject_id: str
    deleted: bool
