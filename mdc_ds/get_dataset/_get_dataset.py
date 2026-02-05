from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from datasets import Dataset


def get_dataset(
    name: Literal[
        "mcv-scripted-zh-TW-v24.0",
        "mcv-scripted-zh-CN-v24.0",
        "mcv-scripted-zh-HK-v24.0",
        "common-voice-v24-english-en-au-subset-fo-0447e8a6",
    ],
    split: Literal["train", "test", "validation"] = "train",
) -> "Dataset":
    if name == "mcv-scripted-zh-TW-v24.0":
        from .mcv_scripted_zh_tw_v24_0 import get_dataset as get_dataset_zh_tw

        return get_dataset_zh_tw(name=name, split=split)

    elif name == "mcv-scripted-zh-CN-v24.0":
        from .mcv_scripted_zh_cn_v24_0 import get_dataset as get_dataset_zh_cn

        return get_dataset_zh_cn(name=name, split=split)

    elif name == "mcv-scripted-zh-HK-v24.0":
        from .mcv_scripted_zh_hk_v24_0 import get_dataset as get_dataset_zh_hk

        return get_dataset_zh_hk(name=name, split=split)

    elif name == "common-voice-v24-english-en-au-subset-fo-0447e8a6":
        from .common_voice_v24_english_en_au_subset_fo_0447e8a6 import (
            get_dataset as get_dataset_en_au,
        )

        return get_dataset_en_au(name=name, split=split)

    else:
        raise ValueError(f"Invalid dataset name: {name}")
