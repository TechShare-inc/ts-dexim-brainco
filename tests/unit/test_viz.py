"""Unit tests for dexim.brainco.viz.subscriber.ActionMessage."""

from __future__ import annotations

import pytest

from dexim.brainco.viz.subscriber import ActionMessage


class TestActionMessage:
    def test_from_dict(self):
        msg = ActionMessage.from_dict({"q": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        assert msg.q == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_from_legacy_list(self):
        msg = ActionMessage.from_dict([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        assert msg.q == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

    def test_direct_construction(self):
        msg = ActionMessage(q=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        assert len(msg.q) == 6

    def test_empty_q(self):
        msg = ActionMessage(q=[])
        assert msg.q == []
