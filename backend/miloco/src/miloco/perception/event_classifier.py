# Copyright (C) 2025 Xiaomi Corporation
# This software may be used and distributed according to the terms of the Xiaomi Miloco License Agreement.

"""有意义事件分类纯函数.

判断一次推理(全屋视图 result)是否有意义 + 计算 has_* 标志位.

注:result 是 `_merge_results` 合并后的全屋视图;classifier 不关心 device 归属
(device_ids 由 client.py 从 processor.py 入参直接传入);MatchedRule / Suggestion
本来就是全屋视角概念,不区分单摄像头.
"""

from __future__ import annotations

from miloco.perception.types import RealtimePerceptionResult


def classify(result: RealtimePerceptionResult) -> dict:
    """计算 has_* 标志位 + 整体是否"有意义".

    入表条件(任一为真即入表):
    - has_rule_hit:`result.matched_rules` 非空
    - has_suggestion:`result.suggestions` 非空
    - has_asr:存在至少一个 `Speech.needs_response=True AND is_complete=True`
      (家人闲聊 / 未说完的指令不算)
    - has_person:identity engine 检测到至少一个人类 track
      (confirmed = 已知成员, unknown = 陌生人, pending = 识别中)
    - has_pet:identity engine 检测到至少一个宠物 track
      (当前官方 track_human_only=True 时恒 False;放开后自动生效)

    Returns:
        {
            "is_meaningful": bool,     # 任一 has_* 为 true
            "has_rule_hit": bool,
            "has_suggestion": bool,
            "has_asr": bool,
            "has_person": bool,
            "has_pet": bool,
        }
    """
    has_rule_hit = bool(result.matched_rules)
    has_suggestion = bool(result.suggestions)
    has_asr = any(
        s.needs_response and s.is_complete for s in result.speeches
    )
    has_person = result.has_person
    has_pet = result.has_pet

    return {
        "is_meaningful": (
            has_rule_hit or has_suggestion or has_asr or has_person or has_pet
        ),
        "has_rule_hit": has_rule_hit,
        "has_suggestion": has_suggestion,
        "has_asr": has_asr,
        "has_person": has_person,
        "has_pet": has_pet,
    }
