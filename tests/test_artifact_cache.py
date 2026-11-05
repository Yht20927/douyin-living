# -*- coding: utf-8 -*-
"""Tests for artifact cache metadata and reuse decisions."""

import json

from src.contracts import (
    ArtifactMeta,
    artifactMetaPath,
    fingerprintMatches,
)
from src.pipeline.artifacts import ArtifactCache


def _write(path, text="payload"):
    path.write_text(text, encoding="utf-8")
    return str(path)


def _cache(**kwargs):
    options = {"settingsHashValue": "settings-1", "generatorVersion": "test-gen"}
    options.update(kwargs)
    return ArtifactCache(**options)


class TestArtifactMetaContract:
    def test_sidecar_path(self, tmp_path):
        assert artifactMetaPath(tmp_path / "a.aac") == str(tmp_path / "a.aac") + ".meta.json"

    def test_save_load_roundtrip(self, tmp_path):
        meta = ArtifactMeta(
            artifactPath=str(tmp_path / "a.json"),
            step="1_audio_extract",
            sourceHashes={"src": {"sizeBytes": 1}},
            settingsHash="s",
            paramsHash="p",
            params={"x": 1},
            elapsedMs=12,
        )
        path = meta.save()
        loaded = ArtifactMeta.load(path)
        assert loaded is not None
        assert loaded.step == "1_audio_extract"
        assert loaded.params == {"x": 1}
        assert loaded.elapsedMs == 12

    def test_load_ignores_unknown_fields(self, tmp_path):
        path = tmp_path / "a.meta.json"
        path.write_text(
            json.dumps({"artifactPath": "a", "step": "s", "future": 1}),
            encoding="utf-8",
        )
        loaded = ArtifactMeta.load(path)
        assert loaded is not None
        assert loaded.artifactPath == "a"

    def test_load_invalid_returns_none(self, tmp_path):
        path = tmp_path / "a.meta.json"
        path.write_text("not json", encoding="utf-8")
        assert ArtifactMeta.load(path) is None
        path.write_text("[]", encoding="utf-8")
        assert ArtifactMeta.load(path) is None
        path.write_text('{"step": "s"}', encoding="utf-8")
        assert ArtifactMeta.load(path) is None

    def test_fingerprint_matches_semantics(self):
        base = {"sizeBytes": 3, "mtimeNs": 1, "sha256": "abc"}
        assert fingerprintMatches(base, dict(base))
        assert not fingerprintMatches(base, {**base, "sizeBytes": 4})
        assert not fingerprintMatches(base, {**base, "sha256": "def"})
        # Large files may skip hashing: size + mtime decide identity.
        largeA = {"sizeBytes": 10, "mtimeNs": 100, "sha256": None}
        largeB = {"sizeBytes": 10, "mtimeNs": 200, "sha256": None}
        assert fingerprintMatches(largeA, dict(largeA))
        assert not fingerprintMatches(largeA, largeB)
        assert not fingerprintMatches(None, largeA)


class TestArtifactCache:
    def test_record_then_hit(self, tmp_path):
        source = _write(tmp_path / "source.bin", "source")
        artifact = _write(tmp_path / "audio.aac", "audio")
        cache = _cache()

        meta = cache.record(artifact, "1_audio_extract", [source], params={"k": 1})
        assert meta.params == {"k": 1}
        assert meta.settingsHash == "settings-1"

        decision = cache.check(artifact, "1_audio_extract", [source], params={"k": 1})
        assert decision.hit is True
        assert decision.meta is not None
        assert cache.stats() == {"hits": 1, "misses": 0, "writes": 1}

    def test_missing_artifact_and_missing_meta(self, tmp_path):
        source = _write(tmp_path / "source.bin")
        cache = _cache()

        assert cache.check(tmp_path / "nope", "step", [source]).hit is False

        artifact = _write(tmp_path / "out.json", "{}")
        assert cache.check(artifact, "step", [source]).hit is False
        assert "metadata" in cache.check(artifact, "step", [source]).reason

    def test_settings_params_and_version_changes_miss(self, tmp_path):
        source = _write(tmp_path / "source.bin")
        artifact = _write(tmp_path / "out.json", "{}")
        cache = _cache()
        cache.record(artifact, "step", [source], params={"a": 1})

        assert not _cache(settingsHashValue="other").check(
            artifact, "step", [source], params={"a": 1}
        ).hit
        assert not cache.check(artifact, "step", [source], params={"a": 2}).hit
        assert not _cache(generatorVersion="other-gen").check(
            artifact, "step", [source], params={"a": 1}
        ).hit
        assert not cache.check(artifact, "different-step", [source], params={"a": 1}).hit

    def test_source_changed_or_missing(self, tmp_path):
        source = tmp_path / "source.bin"
        _write(source, "one")
        artifact = _write(tmp_path / "out.json", "{}")
        cache = _cache()
        cache.record(artifact, "step", [str(source)])

        _write(source, "two")
        assert not cache.check(artifact, "step", [str(source)]).hit

        cache2 = _cache()
        cache2.record(artifact, "step", [str(source)])
        source.unlink()
        decision = cache2.check(artifact, "step", [str(source)])
        assert decision.hit is False
        assert "source missing" in decision.reason

    def test_source_set_change(self, tmp_path):
        first = _write(tmp_path / "a.bin")
        second = _write(tmp_path / "b.bin")
        artifact = _write(tmp_path / "out.json", "{}")
        cache = _cache()
        cache.record(artifact, "step", [first])

        decision = cache.check(artifact, "step", [first, second])
        assert decision.hit is False
        assert "source set changed" in decision.reason

    def test_artifact_modified_after_record(self, tmp_path):
        source = _write(tmp_path / "source.bin")
        artifact = tmp_path / "out.json"
        _write(artifact, "one")
        cache = _cache()
        cache.record(str(artifact), "step", [source])

        _write(artifact, "two")
        assert not cache.check(str(artifact), "step", [source]).hit

    def test_validator_can_reject_a_matching_artifact(self, tmp_path):
        source = _write(tmp_path / "source.bin")
        artifact = _write(tmp_path / "out.json", "{}")
        cache = _cache()
        cache.record(artifact, "step", [source])

        decision = cache.check(
            artifact,
            "step",
            [source],
            validator=lambda: (False, "artifact has no features"),
        )
        assert decision.hit is False
        assert decision.reason == "artifact has no features"
        assert cache.stats()["hits"] == 0

    def test_invalidate_removes_sidecar_only(self, tmp_path):
        source = _write(tmp_path / "source.bin")
        artifact = _write(tmp_path / "out.json", "{}")
        cache = _cache()
        cache.record(artifact, "step", [source])

        cache.invalidate(artifact)
        assert (tmp_path / "out.json").exists()
        assert not (tmp_path / "out.json.meta.json").exists()
        assert not cache.check(artifact, "step", [source]).hit
        # Invalidating a missing sidecar is a no-op.
        cache.invalidate(artifact)
