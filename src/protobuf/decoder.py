"""Protobuf wire-format decoder for the douyin Frontier binary frame protocol.

The Frontier frame uses standard Protobuf encoding with the following schema:

    message Frame {
      required int64  SeqID           = 1;   // tag=8,  wire=0 (varint)
      required int64  LogID           = 2;   // tag=16, wire=0 (varint)
      required int32  service         = 3;   // tag=24, wire=0 (varint)
      required int32  method          = 4;   // tag=32, wire=0 (varint)
      repeated Header headers         = 5;   // tag=42, wire=2 (length-delimited)
      optional string payloadEncoding = 6;   // tag=50, wire=2
      optional string payloadType     = 7;   // tag=58, wire=2
      optional bytes  payload         = 8;   // tag=66, wire=2
      optional string LogIDNew        = 9;   // tag=74, wire=2
      optional string serverTiming    = 10;  // tag=82, wire=2
      optional string msgID           = 11;  // tag=90, wire=2
      optional int32  frameType       = 12;  // tag=96, wire=0 (varint)
    }

    message Header {
      required string key   = 1;  // tag=10, wire=2
      required string value = 2;  // tag=18, wire=2
    }
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FrameHeader:
    key: str = ""
    value: str = ""


@dataclass
class Frame:
    """Decoded Frontier binary frame."""
    seq_id: int = 0
    log_id: int = 0
    service: int = 0
    method: int = 0
    headers: list[FrameHeader] = field(default_factory=list)
    payload_encoding: str = ""
    payload_type: str = ""
    payload: bytes = b""
    log_id_new: str = ""
    server_timing: str = ""
    msg_id: str = ""
    frame_type: int = 0

    @property
    def payload_json(self) -> Optional[dict]:
        """If payload is JSON text, decode it. Otherwise return None."""
        if self.payload_encoding == "json" or not self.payload_encoding:
            import json
            try:
                return json.loads(self.payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None
        return None


# ── Wire format primitives ────────────────────────────────────────────

WIRE_VARINT = 0
WIRE_LENGTH_DELIMITED = 2


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Read a varint from data at offset. Returns (value, new_offset)."""
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if not (byte & 0x80):
            break
    return value, offset


def _read_signed_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Read a zigzag-encoded signed varint. Returns (value, new_offset)."""
    raw, offset = _read_varint(data, offset)
    # ZigZag decode
    value = (raw >> 1) ^ -(raw & 1)
    return value, offset


def _read_string(data: bytes, offset: int, length: int) -> tuple[str, int]:
    """Read a UTF-8 string of given length. Returns (str, new_offset)."""
    end = offset + length
    return data[offset:end].decode("utf-8", errors="replace"), end


def _read_bytes(data: bytes, offset: int, length: int) -> tuple[bytes, int]:
    """Read raw bytes of given length. Returns (bytes, new_offset)."""
    return data[offset:offset + length], offset + length


def decode_frame(data: bytes) -> Frame:
    """Decode a single Frontier binary frame from bytes.

    Args:
        data: Raw bytes from WebSocket message.

    Returns:
        Decoded Frame object.
    """
    frame = Frame()
    offset = 0
    limit = len(data)

    while offset < limit:
        tag_raw, offset = _read_varint(data, offset)
        field_number = tag_raw >> 3
        wire_type = tag_raw & 0x07

        if wire_type == WIRE_VARINT:
            if field_number == 1:
                frame.seq_id, offset = _read_varint(data, offset)
            elif field_number == 2:
                frame.log_id, offset = _read_varint(data, offset)
            elif field_number == 3:
                frame.service, offset = _read_varint(data, offset)
            elif field_number == 4:
                frame.method, offset = _read_varint(data, offset)
            elif field_number == 12:
                frame.frame_type, offset = _read_varint(data, offset)
            else:
                # Skip unknown varint field
                _, offset = _read_varint(data, offset)

        elif wire_type == WIRE_LENGTH_DELIMITED:
            length, offset = _read_varint(data, offset)
            if field_number == 5:
                # Repeated Header sub-message
                header = _decode_header(data, offset, length)
                frame.headers.append(header)
            elif field_number == 6:
                frame.payload_encoding, _ = _read_string(data, offset, length)
            elif field_number == 7:
                frame.payload_type, _ = _read_string(data, offset, length)
            elif field_number == 8:
                frame.payload, _ = _read_bytes(data, offset, length)
            elif field_number == 9:
                frame.log_id_new, _ = _read_string(data, offset, length)
            elif field_number == 10:
                frame.server_timing, _ = _read_string(data, offset, length)
            elif field_number == 11:
                frame.msg_id, _ = _read_string(data, offset, length)
            # else skip unknown
            offset += length

        else:
            # Skip unknown wire type — for groups (3,4) or future types
            if wire_type == 0:
                _, offset = _read_varint(data, offset)
            elif wire_type == 2:
                length, offset = _read_varint(data, offset)
                offset += length
            elif wire_type == 5:
                offset += 4  # 32-bit fixed
            elif wire_type == 1:
                offset += 8  # 64-bit fixed
            else:
                break  # safety

    return frame


def _decode_header(data: bytes, offset: int, length: int) -> FrameHeader:
    """Decode a Header sub-message (field 1=key, field 2=value)."""
    end = offset + length
    hdr = FrameHeader()
    while offset < end:
        tag_raw, offset = _read_varint(data, offset)
        field_number = tag_raw >> 3
        wire_type = tag_raw & 0x07
        if wire_type == 2:  # length-delimited string
            flen, offset = _read_varint(data, offset)
            val, _ = _read_string(data, offset, flen)
            offset += flen
            if field_number == 1:
                hdr.key = val
            elif field_number == 2:
                hdr.value = val
        else:
            # Skip unknown wire type
            if wire_type == 0:
                _, offset = _read_varint(data, offset)
            elif wire_type == 5:
                offset += 4
            elif wire_type == 1:
                offset += 8
    return hdr


def encode_frame(frame: Frame) -> bytes:
    """Encode a Frame back to binary (for sending).

    This is a simplified encoder — only encodes non-empty/non-zero fields.
    For full parity with the JS encoder, use protobuf library.
    """
    parts: list[bytes] = []

    def write_varint_tag(field: int, wire: int, value: int):
        tag = (field << 3) | wire
        parts.append(_encode_varint(tag))
        parts.append(_encode_varint(value))

    def write_length_delimited(field: int, payload: bytes):
        tag = (field << 3) | 2
        parts.append(_encode_varint(tag))
        parts.append(_encode_varint(len(payload)))
        parts.append(payload)

    if frame.seq_id:
        write_varint_tag(1, 0, frame.seq_id)
    if frame.log_id:
        write_varint_tag(2, 0, frame.log_id)
    if frame.service:
        write_varint_tag(3, 0, frame.service)
    if frame.method:
        write_varint_tag(4, 0, frame.method)
    for h in frame.headers:
        hdr_bytes = (
            _encode_varint((1 << 3) | 2) + _encode_varint(len(h.key.encode())) + h.key.encode() +
            _encode_varint((2 << 3) | 2) + _encode_varint(len(h.value.encode())) + h.value.encode()
        )
        write_length_delimited(5, hdr_bytes)
    if frame.payload_encoding:
        write_length_delimited(6, frame.payload_encoding.encode())
    if frame.payload_type:
        write_length_delimited(7, frame.payload_type.encode())
    if frame.payload:
        write_length_delimited(8, frame.payload)
    if frame.log_id_new:
        write_length_delimited(9, frame.log_id_new.encode())
    if frame.server_timing:
        write_length_delimited(10, frame.server_timing.encode())
    if frame.msg_id:
        write_length_delimited(11, frame.msg_id.encode())
    if frame.frame_type:
        write_varint_tag(12, 0, frame.frame_type)

    return b"".join(parts)


def _encode_varint(value: int) -> bytes:
    """Encode an unsigned integer as a varint.

    Raises ValueError for negative values, which would otherwise
    cause an infinite loop (Python's >> on negative ints preserves sign).
    """
    if value < 0:
        raise ValueError(f"varint does not support negative values: {value}")
    result = []
    while value > 0x7F:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value & 0x7F)
    return bytes(result) if result else b"\x00"
