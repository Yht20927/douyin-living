# -*- coding: utf-8 -*-
"""Quick verification of refactored modules."""

from src.models import RoomInfo, StreamInfo, DanmakuMessage, RoomStatus
from src.signer import Signer
from src.params import Params
from src.auth import Auth


def testModels():
    s = StreamInfo(
        flvUrls={"HD1": "http://x.com/a.flv"},
        hlsUrls={"HD1": "http://x.com/a.m3u8"},
        defaultQuality="hd",
        streamId="123",
        sessionId="abc",
    )
    r = RoomInfo(webRid="123456", roomId="789", title="Test", status=RoomStatus.LIVING, userCount=100, stream=s)
    m = DanmakuMessage(roomId="123456", msgType="chat", content="hello")
    assert r.title == "Test"
    assert s.flvUrl("HD1") == "http://x.com/a.flv"
    assert m.msgType == "chat"
    print("✓ models")


def testSigner():
    ms = Signer.generateMsToken()
    assert len(ms) == 107
    sig = Signer.generateLiveSignature("123", "456")
    assert isinstance(sig, str) and len(sig) > 0
    print("✓ signer")


def testParams():
    p = Params()
    p.withPlatform().withMsToken()
    qs = p.build()
    assert "aid=6383" in qs
    assert "msToken=" in qs
    print("✓ params")


def testAuth():
    a = Auth.fromString("k1=v1; k2=v2")
    assert a.cookie["k1"] == "v1"
    assert a.cookie["k2"] == "v2"
    print("✓ auth")


if __name__ == "__main__":
    for t in [testModels, testSigner, testParams, testAuth]:
        try:
            t()
        except Exception as e:
            print(f"✗ {t.__name__}: {e}")
            raise
    print("All tests passed ✓")
