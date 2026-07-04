# -*- coding: utf-8 -*-
"""Unit tests for Signer — msToken, webId generation."""

import pytest
from src.signer import Signer


class TestGenerateMsToken:
    def test_default_length(self):
        token = Signer.generateMsToken()
        assert len(token) == 107

    def test_custom_length(self):
        token = Signer.generateMsToken(length=50)
        assert len(token) == 50

    def test_only_valid_characters(self):
        token = Signer.generateMsToken(length=200)
        valid = set(
            "ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789="
        )
        assert all(c in valid for c in token)

    def test_multiple_calls_produce_different_tokens(self):
        tokens = {Signer.generateMsToken() for _ in range(10)}
        # With 107 chars each from 64-char alphabet, collisions are
        # astronomically unlikely — all 10 should be distinct.
        assert len(tokens) == 10


class TestGenerateWebId:
    def test_default_length(self):
        wid = Signer.generateWebId()
        assert len(wid) == 19

    def test_custom_length(self):
        wid = Signer.generateWebId(length=10)
        assert len(wid) == 10

    def test_only_digits(self):
        wid = Signer.generateWebId(length=50)
        assert wid.isdigit()

    def test_different_calls_produce_different_ids(self):
        ids = {Signer.generateWebId() for _ in range(10)}
        assert len(ids) == 10


class TestGenerateMsTokenValue:
    def test_alias_returns_same_length(self):
        assert len(Signer.generateMsTokenValue()) == 107
