"""包标记文件。

`src` 下是纯计算逻辑与 I/O，不含第三方依赖；CLI 通过 `python -m src.cli` 调用。
"""

__all__ = ["catalog", "sizing", "tp_ep", "token_per_watt", "tco",
           "form_validator", "doc_assembler", "cli"]
