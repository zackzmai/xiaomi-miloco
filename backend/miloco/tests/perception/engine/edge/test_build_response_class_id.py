"""_build_response class_id → ObjectType 映射修复的验证测试。

验证修复后 _build_response 根据 class_id 正确映射 ObjectType 和 box_type：
- CLASS_HUMAN (0) → ObjectType.HUMAN_BODY + "human_body"
- CLASS_CAT (1) / CLASS_DOG (2) → ObjectType.PET + "pet_body"
"""

from __future__ import annotations

from miloco.perception.engine.identity.tracking_service import _build_response
from miloco.perception.engine.types import ObjectType


class TestBuildResponseClassIdMapping:
    """验证 _build_response 根据 class_id 正确映射类型。"""

    def test_human_class_id_maps_to_human_body(self):
        """class_id=0 (human) → HUMAN_BODY + human_body box。"""
        results = [{"id": 0, "xyxy": (100, 200, 300, 400), "class_id": 0}]
        resp = _build_response(results, n_frames=6, fps=2)
        obj = resp.object_info[0]
        assert obj.type == ObjectType.HUMAN_BODY
        assert "human_body" in obj.box_info[0].boxes

    def test_cat_class_id_maps_to_pet(self):
        """class_id=1 (cat) → PET + pet_body box。"""
        results = [{"id": 1, "xyxy": (200, 300, 300, 380), "class_id": 1}]
        resp = _build_response(results, n_frames=6, fps=2)
        obj = resp.object_info[0]
        assert obj.type == ObjectType.PET
        assert "pet_body" in obj.box_info[0].boxes

    def test_dog_class_id_maps_to_pet(self):
        """class_id=2 (dog) → PET + pet_body box。"""
        results = [{"id": 2, "xyxy": (300, 100, 450, 250), "class_id": 2}]
        resp = _build_response(results, n_frames=6, fps=2)
        obj = resp.object_info[0]
        assert obj.type == ObjectType.PET
        assert "pet_body" in obj.box_info[0].boxes

    def test_missing_class_id_defaults_to_human(self):
        """无 class_id 字段时回退为 HUMAN_BODY（向后兼容）。"""
        results = [{"id": 0, "xyxy": (100, 200, 300, 400)}]
        resp = _build_response(results, n_frames=6, fps=2)
        obj = resp.object_info[0]
        assert obj.type == ObjectType.HUMAN_BODY
        assert "human_body" in obj.box_info[0].boxes

    def test_mixed_human_and_pet(self):
        """人 + 宠物混合结果正确映射。"""
        results = [
            {"id": 0, "xyxy": (100, 100, 300, 400), "class_id": 0},
            {"id": 1, "xyxy": (400, 300, 500, 380), "class_id": 1},
            {"id": 2, "xyxy": (200, 350, 320, 430), "class_id": 2},
        ]
        resp = _build_response(results, n_frames=6, fps=2)
        assert resp.object_info[0].type == ObjectType.HUMAN_BODY
        assert resp.object_info[1].type == ObjectType.PET
        assert resp.object_info[2].type == ObjectType.PET

    def test_pet_box_coords_correct(self):
        """宠物的 bbox xyxy → xywh 转换正确。"""
        results = [{"id": 1, "xyxy": (100, 200, 180, 260), "class_id": 1}]
        resp = _build_response(results, n_frames=4, fps=1)
        box = resp.object_info[0].box_info[0]
        assert box.boxes["pet_body"] == (100, 200, 80, 60)

    def test_pet_face_id_is_none(self):
        """宠物的 face_id 始终为 'none'。"""
        results = [{"id": 1, "xyxy": (100, 200, 180, 260), "class_id": 1}]
        resp = _build_response(results, n_frames=4, fps=1)
        assert resp.object_info[0].face_id == "none"
