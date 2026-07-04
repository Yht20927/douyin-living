# -*- coding: utf-8 -*-
"""Unit tests for protobuf decoder — varint, encode/decode round-trip."""

import pytest
from src.protobuf.decoder import (
    _read_varint,
    _encode_varint,
    _read_signed_varint,
    decode_frame,
    encode_frame,
    Frame,
    FrameHeader,
)


class TestVarint:
    def test_encode_zero(self):
        assert _encode_varint(0) == b"\x00"

    def test_encode_single_byte(self):
        assert _encode_varint(1) == b"\x01"
        assert _encode_varint(127) == b"\x7f"

    def test_encode_multi_byte(self):
        # 128 = 0b10000000 → needs two bytes
        assert _encode_varint(128) == b"\x80\x01"
        # 300 needs two bytes
        assert _encode_varint(300) == b"\xac\x02"

    def test_read_varint_zero(self):
        val, offset = _read_varint(b"\x00", 0)
        assert val == 0
        assert offset == 1

    def test_read_varint_single_byte(self):
        val, offset = _read_varint(b"\x7f", 0)
        assert val == 127
        assert offset == 1

    def test_read_varint_multi_byte(self):
        val, offset = _read_varint(b"\x80\x01", 0)
        assert val == 128
        assert offset == 2

    def test_encode_decode_round_trip(self):
        """Encoding then decoding should return the original value."""
        for v in [0, 1, 127, 128, 300, 10000, 2**32 - 1, 2**63 - 1]:
            encoded = _encode_varint(v)
            decoded, offset = _read_varint(encoded, 0)
            assert decoded == v, f"Round-trip failed for {v}"
            assert offset == len(encoded)

    def test_encode_negative_raises(self):
        with pytest.raises(ValueError, match="negative"):
            _encode_varint(-1)

    def test_encode_negative_large_raises(self):
        with pytest.raises(ValueError, match="negative"):
            _encode_varint(-99999999)


class TestSignedVarint:
    def test_read_signed_zero(self):
        val, _ = _read_signed_varint(b"\x00", 0)
        assert val == 0

    def test_read_signed_negative(self):
        # -1 is zigzag-encoded as 1
        val, _ = _read_signed_varint(b"\x01", 0)
        assert val == -1

    def test_read_signed_positive(self):
        # 1 is zigzag-encoded as 2
        val, _ = _read_signed_varint(b"\x02", 0)
        assert val == 1


class TestFrameRoundTrip:
    def test_empty_frame(self):
        frame = Frame()
        data = encode_frame(frame)
        decoded = decode_frame(data)
        assert decoded.seq_id == 0
        assert decoded.log_id == 0

    def test_frame_with_fields(self):
        frame = Frame(
            seq_id=42,
            log_id=100,
            service=1,
            method=2,
            payload_type="msg",
            payload=b"hello",
        )
        data = encode_frame(frame)
        decoded = decode_frame(data)
        assert decoded.seq_id == 42
        assert decoded.log_id == 100
        assert decoded.service == 1
        assert decoded.method == 2
        assert decoded.payload_type == "msg"
        assert decoded.payload == b"hello"

    def test_frame_with_headers(self):
        frame = Frame(
            headers=[
                FrameHeader(key="k1", value="v1"),
                FrameHeader(key="k2", value="v2"),
            ]
        )
        data = encode_frame(frame)
        decoded = decode_frame(data)
        assert len(decoded.headers) == 2
        assert decoded.headers[0].key == "k1"
        assert decoded.headers[0].value == "v1"
        assert decoded.headers[1].key == "k2"
        assert decoded.headers[1].value == "v2"

    def test_frame_with_all_fields(self):
        frame = Frame(
            seq_id=1, log_id=2, service=3, method=4,
            headers=[FrameHeader(key="x", value="y")],
            payload_encoding="json",
            payload_type="msg",
            payload=b'{"a":1}',
            log_id_new="log123",
            server_timing="10ms",
            msg_id="msg456",
            frame_type=1,
        )
        data = encode_frame(frame)
        decoded = decode_frame(data)
        assert decoded.seq_id == 1
        assert decoded.log_id == 2
        assert decoded.service == 3
        assert decoded.method == 4
        assert len(decoded.headers) == 1
        assert decoded.payload_encoding == "json"
        assert decoded.payload_type == "msg"
        assert decoded.payload == b'{"a":1}'
        assert decoded.log_id_new == "log123"
        assert decoded.server_timing == "10ms"
        assert decoded.msg_id == "msg456"
        assert decoded.frame_type == 1

    def test_payload_json_property(self):
        frame = Frame(
            payload_encoding="json",
            payload=b'{"hello": "world"}',
        )
        assert frame.payload_json == {"hello": "world"}

    def test_payload_json_invalid_returns_none(self):
        frame = Frame(
            payload_encoding="json",
            payload=b"not json",
        )
        assert frame.payload_json is None


class TestDecodeEdgeCases:
    def test_empty_bytes(self):
        frame = decode_frame(b"")
        assert frame.seq_id == 0

    def test_truncated_varint(self):
        """Truncated varint at end of message should not crash."""
        frame = decode_frame(b"\x80")  # MSB set but no continuation
        # Should return whatever was decoded (best-effort)
        assert isinstance(frame, Frame)
