"""
Bound malformed USB streams without admitting damaged frames or losing boundary recovery.
"""

import pytest
from support.frames import make_frame

from based_thermals.camera_backends.duo.camera import (
    YUY2_FRAME_BYTES,
    FrameAssembler,
    NegotiatedFrameAssembler,
)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("recovery", ["fid", "eof"])
def test_missing_frame_boundaries_cannot_grow_memory_indefinitely(
    strict: bool, recovery: str,
) -> None:
    """
    Discard overflow through the next boundary, then accept a complete independent frame.
    """
    assembler = NegotiatedFrameAssembler(YUY2_FRAME_BYTES) if strict else FrameAssembler()
    source = make_frame()
    if strict:
        source += bytes(YUY2_FRAME_BYTES - len(source))
    assert assembler.feed(b"\x02\x80" + source) is None
    packet = b"\x02\x80" + bytes(8134)
    for _ in range(2000):
        assert assembler.feed(packet) is None
        assert assembler.buffered_bytes <= YUY2_FRAME_BYTES * 2
    assert assembler.buffered_bytes == 0
    assert assembler.rejected["size_mismatch"] == 1
    if recovery == "eof":
        assert assembler.feed(b"\x02\x82" + source) is None
        assert assembler.feed(b"\x02\x82" + source) == source
    else:
        assert assembler.feed(b"\x02\x83" + source) == source
