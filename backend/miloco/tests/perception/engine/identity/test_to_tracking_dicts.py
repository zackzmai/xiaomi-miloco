"""_to_tracking_dicts class_id 映射修复的验证测试。

验证修复后 _to_tracking_dicts 根据 ObjectType 正确映射 class_id 和 box_type：
- ObjectType.HUMAN_BODY → class_id=0 (HUMAN) + "human_body" box
- ObjectType.PET → class_id=1 (CAT) + "pet_body" box
"""

from __future__ import annotations

from miloco.perception.engine.identity.identity import _to_tracking_dicts
from miloco.perception.engine.types import ObjectType, TrackedObject, TrackingBoxInfo


class TestToTrackingDictsClassIdMapping:
    """验证 _to_tracking_dicts 根据 ObjectType 正确映射 class_id。"""

    def test_human_type_maps_to_class_id_0(self):
        """ObjectType.HUMAN_BODY → class_id=0 (HUMAN)。"""
        obj = TrackedObject(
            type=ObjectType.HUMAN_BODY,
            face_id="none",
            track_id=0,
            box_info=[TrackingBoxInfo(frame_index=0, boxes={"human_body": (100, 200, 150, 200)})],
        )
        result = _to_tracking_dicts([obj])
        assert result[0]["class_id"] == 0
        assert result[0]["bbox"] == (100, 200, 150, 200)
        assert result[0]["xyxy"] == (100, 200, 250, 400)

    def test_pet_type_maps_to_class_id_1(self):
        """ObjectType.PET → class_id=1 (CAT)。"""
        obj = TrackedObject(
            type=ObjectType.PET,
            face_id="none",
            track_id=1,
            box_info=[TrackingBoxInfo(frame_index=0, boxes={"pet_body": (300, 400, 80, 60)})],
        )
        result = _to_tracking_dicts([obj])
        assert result[0]["class_id"] == 1  # CLASS_CAT
        assert result[0]["bbox"] == (300, 400, 80, 60)
        assert result[0]["xyxy"] == (300, 400, 380, 460)

    def test_pet_type_uses_pet_body_box(self):
        """PET 类型应该从 "pet_body" key 取 bbox，而不是 "human_body"。"""
        obj = TrackedObject(
            type=ObjectType.PET,
            face_id="none",
            track_id=1,
            box_info=[TrackingBoxInfo(frame_index=0, boxes={"pet_body": (200, 300, 50, 40)})],
        )
        result = _to_tracking_dicts([obj])
        assert result[0]["bbox"] == (200, 300, 50, 40)

    def test_pet_type_fallback_to_human_body(self):
        """PET 类型如果没有 "pet_body" key，应该回退到 "human_body"。"""
        obj = TrackedObject(
            type=ObjectType.PET,
            face_id="none",
            track_id=1,
            box_info=[TrackingBoxInfo(frame_index=0, boxes={"human_body": (100, 200, 50, 40)})],
        )
        result = _to_tracking_dicts([obj])
        assert result[0]["bbox"] == (100, 200, 50, 40)

    def test_mixed_types_correct_mapping(self):
        """混合类型正确映射。"""
        objects = [
            TrackedObject(
                type=ObjectType.HUMAN_BODY,
                face_id="none",
                track_id=0,
                box_info=[TrackingBoxInfo(frame_index=0, boxes={"human_body": (100, 200, 150, 200)})],
            ),
            TrackedObject(
                type=ObjectType.PET,
                face_id="none",
                track_id=1,
                box_info=[TrackingBoxInfo(frame_index=0, boxes={"pet_body": (300, 400, 80, 60)})],
            ),
        ]
        result = _to_tracking_dicts(objects)
        assert result[0]["class_id"] == 0
        assert result[1]["class_id"] == 1
