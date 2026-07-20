"""
DeepGuard — services/detection/service.py

Service coordinating face extraction, deep learning inference, and database persistence
for deepfake detection tasks.
"""

import uuid
import logging
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
from sqlalchemy.ext.asyncio import AsyncSession
from database.models import DetectionResultDB
from repositories.sqlite.detection import SQLiteDetectionRepository
from repositories.sqlite.model import SQLiteModelRepository
from models.factory import ModelFactory
from datasets.preprocessors.face_extractor import FaceExtractor
from utils.explainability import ExplainabilityEngine

logger = logging.getLogger(__name__)


class DetectionService:
    """Orchestrates image/video decoding, preprocessing, model inference, and database logging."""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session
        self.detection_repo = SQLiteDetectionRepository(db_session)
        self.model_repo = SQLiteModelRepository(db_session)
        # Use cv2 Haar Cascade as a lightweight detector for serving/unit tests
        self.face_extractor = FaceExtractor(backend="opencv_haar", image_size=224, fallback_to_full=True)
        self._model: torch.nn.Module | None = None
        self._model_config: dict | None = None

    async def get_model(self) -> torch.nn.Module:
        """Resolve and load the active ViT inference model."""
        if self._model is not None:
            return self._model

        active_model_record = await self.model_repo.get_active()

        # Load configuration
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "configs" / "model_config.yaml"
        with open(config_path) as f:
            model_config = yaml.safe_load(f)

        self._model_config = model_config
        self._model = ModelFactory.create_model(model_config)

        # Attempt to load active model weights if a registered model version is active.
        if active_model_record is not None:
            registry_path = (active_model_record.registry_path or "").strip()
            if registry_path:
                model_path = Path(registry_path)
                if not model_path.is_absolute():
                    model_path = project_root / registry_path.lstrip("./")

                if model_path.exists() and model_path.suffix.lower() in {".pt", ".pth", ".bin", ".ckpt"}:
                    try:
                        checkpoint = torch.load(model_path, map_location="cpu")
                        if isinstance(checkpoint, dict):
                            if "model_state_dict" in checkpoint:
                                self._model.load_state_dict(checkpoint["model_state_dict"])
                            else:
                                self._model.load_state_dict(checkpoint)
                        else:
                            self._model.load_state_dict(checkpoint)
                        logger.info("Loaded active model weights from %s", model_path)
                    except Exception as exc:
                        logger.warning(
                            "Failed to load active model weights from %s: %s. Using default model initialization.",
                            model_path,
                            str(exc),
                        )
                elif model_path.exists():
                    logger.info(
                        "Active model registry path '%s' is not a supported PyTorch checkpoint extension. "
                        "Loaded architecture without checkpoint weights.",
                        model_path,
                    )
                else:
                    logger.warning(
                        "Active model registry path '%s' does not exist. Using default weights instead.",
                        model_path,
                    )
            else:
                logger.info("Active model version found but registry path is empty. Using default weights.")

        self._model.eval()
        return self._model

    async def get_onnx_session(self):
        """Resolve and load the ONNX inference session."""
        use_onnx = False
        if self._model_config:
            use_onnx = self._model_config.get("inference", {}).get("use_onnx", False)
        
        import os
        if os.getenv("INFERENCE_USE_ONNX", "").lower() in ("true", "1"):
            use_onnx = True

        if not use_onnx:
            return None

        project_root = Path(__file__).resolve().parents[2]
        onnx_path = project_root / "weights" / "model.onnx"
        if not onnx_path.exists():
            logger.warning("ONNX model weights not found at %s. Falling back to PyTorch model.", onnx_path)
            return None

        import onnxruntime as ort
        # Use CPUExecutionProvider for standard CPU environments
        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"]
        )
        return session

    async def detect_image(self, file_bytes: bytes, filename: str) -> DetectionResultDB:
        """Execute deepfake detection on image bytes."""
        record_id = str(uuid.uuid4())
        record = DetectionResultDB(
            id=record_id,
            filename=filename,
            media_type="image",
            status="processing",
            created_at=datetime.utcnow()
        )
        await self.detection_repo.add(record)

        try:
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image bytes.")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.face_extractor.extract_faces(img_rgb)
            faces_count = len(faces)

            model = await self.get_model()

            # Normalization stats
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

            face_tensors = []
            for face in faces:
                face_norm = face.astype(np.float32) / 255.0
                face_norm = (face_norm - mean) / std
                face_tensor = torch.from_numpy(face_norm).permute(2, 0, 1)
                face_tensors.append(face_tensor)

            batch_tensor = torch.stack(face_tensors)
            
            onnx_session = await self.get_onnx_session()
            if onnx_session is not None:
                input_name = onnx_session.get_inputs()[0].name
                ort_inputs = {input_name: batch_tensor.cpu().numpy()}
                ort_outs = onnx_session.run(None, ort_inputs)
                logits = torch.from_numpy(ort_outs[0])
                probs = torch.softmax(logits, dim=-1)[:, 1]
            else:
                with torch.no_grad():
                    logits = model(batch_tensor)
                    probs = torch.softmax(logits, dim=-1)[:, 1]  # Fake class index (1)

            agg_method = "mean"
            if self._model_config:
                agg_method = self._model_config.get("inference", {}).get("face_aggregation", "mean")

            probs_np = probs.cpu().numpy()
            if agg_method == "max":
                confidence = float(np.max(probs_np))
            else:
                confidence = float(np.mean(probs_np))

            threshold = 0.5
            if self._model_config:
                threshold = self._model_config.get("inference", {}).get("confidence_threshold", 0.5)

            label = 1 if confidence >= threshold else 0

            # Generate Explainable AI results
            target_face = faces[0] if faces_count > 0 else img_rgb
            explain_data = ExplainabilityEngine.generate(
                model=model,
                face_img=target_face,
                label=label,
                confidence=confidence
            )

            # Update DB entry
            record.status = "completed"
            record.label = label
            record.confidence = confidence
            record.faces_count = faces_count
            record.completed_at = datetime.utcnow()
            record.meta_info = {
                "detector": self.face_extractor._active_backend,
                "aggregation": agg_method,
                "model_name": self._model_config.get("model", {}).get("name", "vit_tiny") if self._model_config else "vit_tiny",
                "explainability": explain_data
            }
            await self.db_session.commit()
            await self.db_session.refresh(record)

        except Exception as e:
            logger.exception("Error processing image detection:")
            record.status = "failed"
            record.error_message = str(e)
            record.completed_at = datetime.utcnow()
            await self.db_session.commit()
            await self.db_session.refresh(record)

        return record

    async def detect_video(self, file_path: Path, filename: str) -> DetectionResultDB:
        """Execute deepfake detection on video file path."""
        record_id = str(uuid.uuid4())
        record = DetectionResultDB(
            id=record_id,
            filename=filename,
            media_type="video",
            status="processing",
            created_at=datetime.utcnow()
        )
        await self.detection_repo.add(record)

        try:
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                raise ValueError("Failed to open video file stream.")

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            sample_rate = max(1, int(fps))  # 1 frame per second
            frame_idx = 0
            face_tensors = []
            faces_count = 0

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

            first_face_img = None
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = self.face_extractor.extract_faces(frame_rgb)
                    faces_count += len(faces)
                    for face in faces:
                        if first_face_img is None:
                            first_face_img = face
                        face_norm = face.astype(np.float32) / 255.0
                        face_norm = (face_norm - mean) / std
                        face_tensor = torch.from_numpy(face_norm).permute(2, 0, 1)
                        face_tensors.append(face_tensor)

                frame_idx += 1

            cap.release()

            if len(face_tensors) == 0:
                raise ValueError("No frames or faces were extracted from the video.")

            model = await self.get_model()
            batch_size = 16
            probs_list = []
            onnx_session = await self.get_onnx_session()
            if onnx_session is not None:
                input_name = onnx_session.get_inputs()[0].name
                for idx in range(0, len(face_tensors), batch_size):
                    batch = torch.stack(face_tensors[idx : idx + batch_size])
                    ort_inputs = {input_name: batch.cpu().numpy()}
                    ort_outs = onnx_session.run(None, ort_inputs)
                    logits = torch.from_numpy(ort_outs[0])
                    probs = torch.softmax(logits, dim=-1)[:, 1]
                    probs_list.extend(probs.cpu().numpy().tolist())
            else:
                with torch.no_grad():
                    for idx in range(0, len(face_tensors), batch_size):
                        batch = torch.stack(face_tensors[idx : idx + batch_size])
                        logits = model(batch)
                        probs = torch.softmax(logits, dim=-1)[:, 1]
                        probs_list.extend(probs.cpu().numpy().tolist())

            agg_method = "mean"
            if self._model_config:
                agg_method = self._model_config.get("inference", {}).get("face_aggregation", "mean")

            if agg_method == "max":
                confidence = float(np.max(probs_list))
            else:
                confidence = float(np.mean(probs_list))

            threshold = 0.5
            if self._model_config:
                threshold = self._model_config.get("inference", {}).get("confidence_threshold", 0.5)

            label = 1 if confidence >= threshold else 0

            # Generate Explainable AI results on first detected face
            explain_data = {}
            if first_face_img is not None:
                explain_data = ExplainabilityEngine.generate(
                    model=model,
                    face_img=first_face_img,
                    label=label,
                    confidence=confidence
                )

            record.status = "completed"
            record.label = label
            record.confidence = confidence
            record.faces_count = faces_count
            record.completed_at = datetime.utcnow()
            record.meta_info = {
                "detector": self.face_extractor._active_backend,
                "aggregation": agg_method,
                "frames_processed": len(face_tensors),
                "model_name": self._model_config.get("model", {}).get("name", "vit_tiny") if self._model_config else "vit_tiny",
                "explainability": explain_data
            }
            await self.db_session.commit()
            await self.db_session.refresh(record)

        except Exception as e:
            logger.exception("Error processing video detection:")
            record.status = "failed"
            record.error_message = str(e)
            record.completed_at = datetime.utcnow()
            await self.db_session.commit()
            await self.db_session.refresh(record)

        return record
