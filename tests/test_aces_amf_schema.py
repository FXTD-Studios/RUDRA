"""RUDRA's AMF sidecars validate against the Academy's AMF v2.0 schema.

native/tests/test_aces_amf.cpp holds aces_amf_xml() byte-equal to these
goldens; this holds the goldens to the official XSD (tests/data/aces_amf).
"""
from pathlib import Path

import pytest

etree = pytest.importorskip("lxml.etree")

ROOT = Path(__file__).resolve().parents[1]
XSD = ROOT / "tests" / "data" / "aces_amf" / "acesMetadataFile.xsd"
GOLDENS = sorted((ROOT / "native" / "tests" / "golden" / "amf").glob("*.amf"))


@pytest.fixture(scope="module")
def schema():
    return etree.XMLSchema(etree.parse(str(XSD)))


def test_there_are_goldens():
    assert len(GOLDENS) >= 3


@pytest.mark.parametrize("amf", GOLDENS, ids=lambda p: p.name)
def test_golden_is_schema_valid(schema, amf):
    doc = etree.parse(str(amf))
    assert schema.validate(doc), schema.error_log


def test_a_broken_file_is_caught(schema, tmp_path):
    bad = GOLDENS[0].read_text().replace("<aces:majorVersion>2</aces:majorVersion>", "")
    p = tmp_path / "bad.amf"
    p.write_text(bad)
    assert not schema.validate(etree.parse(str(p)))
