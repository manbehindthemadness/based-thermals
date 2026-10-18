"""
Readable annotation formulas, stable references and bounded table rendering without external I/O.
"""

import json
from copy import deepcopy

import numpy as np
import pytest
from support.pipeline import raw_pipeline

from based_thermals.markup.expressions import evaluate, render, render_item
from based_thermals.markup.labels import IDENTITY
from based_thermals.markup.model import annotation, read_annotations, serialize
from based_thermals.markup.paint import draw_object, masks_for
from based_thermals.markup.runtime import ReadoutRuntime
from based_thermals.markup.tables import cells
from based_thermals.markup.variables import catalog, pipeline_value, references
from based_thermals.pipeline import node, validate_pipeline


@pytest.mark.parametrize("source,expected", [
    ("[Hot] - [Cold] :.1f", "8.0"), ("([Hot] + [Cold]) / 2 :.2f", "16.00"),
    ("max([Hot], [Cold]) :d", "20"), ("sqrt(abs([Cold] - [Hot])) :.2f", "2.83"),
    ("round([Hot] / 3, 2)", "6.67"), ("ceil([Cold] / 5) + floor([Hot] / 3)", "9"),
    ("[Hot] // 3 + [Cold] % 5", "8"), ("2 ** 4", "16"),
    ("[Clock.Time]", "12:34:56"), ("[Room: desk] :.1f", "3.0"),
])
def test_readable_math_and_formats(source: str, expected: str) -> None:
    """
    Evaluate familiar arithmetic with multiple references, named functions and explicit formatting.
    """
    values = {"Hot": 20., "Cold": 12., "Clock.Time": "12:34:56", "Room: desk": 3.}
    assert evaluate(source, values) == expected


@pytest.mark.parametrize("source", [
    "__import__('os').system('touch /tmp/markup-must-not-execute')", "[Hot].__class__",
    "(lambda: 7)()", "{'x': 1}", "sum([Hot])", "open(1)",
    "[Hot][0]", "2 ** 100000", "1 / 0", "sqrt(-1)", "1e999", "True + 1",
    "[Missing] + ().__class__", "[Hot] :999999999f", "+" * 1500 + "1",
])
def test_invalid_or_executable_formulas_are_localized_errors(source: str) -> None:
    """
    Attributes, Python calls, containers and unbounded arithmetic can never reach execution.
    """
    assert render("Value {{ " + source + " }} end", {"Hot": 20.}) == "Value #ERR end"


def test_alias_bindings_unavailable_values_and_literal_text() -> None:
    """
    Retain human names separately from stable identities and never treat missing values as zero.
    """
    text = "Delta: {{ [Desk maximum] - [Reference] :.1f }} C"
    bindings = {"Desk maximum": "Measurement.1.Maximum", "Reference": "Measurement.2.Average"}
    assert render(text, {"Measurement.1.Maximum": 30., "Measurement.2.Average": 22.}, bindings) == "Delta: 8.0 C"
    assert render(text, {"Measurement.1.Maximum": 30.}, bindings) == "Delta: -- C"
    assert render("Static text", {}) == "Static text"
    assert render("{{ [x for x in (1, 2)] }}", {}) == "--"


def test_pipeline_picker_references_survive_reordering_and_tab_moves() -> None:
    """
    Resolve settings by saved node identity, with JSON-pointer escaping for Custom config fields.
    """
    first, second = node("software", "gamma", amount=1.8), node("software", "brightness", amount=7.)
    document = raw_pipeline(first, second)
    entries = catalog(document, {"Temperature.Unit": "F"}, [{"number": 1, "name": "Desk [top]"}])
    entry = next(entry for entry in entries if entry["key"] == f"Pipeline/{first['id']}/params/amount")
    assert "gamma" in entry["group"].lower() and entry["value"] == 1.8
    markup = annotation("text")
    markup.update(text="{{ [Amplification] :.1f }}", bindings={"Amplification": entry["key"]}, points=[[.1, .1]])
    document["software"].insert(-1, node("software", "markup", annotations=serialize([markup])))
    document["software"].remove(first)
    document["branches"]["B"] = [node("software", "source"), first]
    assert references(document) == {entry["key"]}
    assert pipeline_value(document, entry["key"]) == 1.8
    restored = validate_pipeline(json.loads(json.dumps(document)))
    assert restored == document
    first["params"]["config"] = json.dumps({"a/b~c": [3, 4]})
    assert pipeline_value(document, f"Pipeline/{first['id']}/params/config/a~1b~0c/1") == 4
    document["branches"]["B"].remove(first)
    assert pipeline_value(document, entry["key"]) is None


def test_legacy_bindings_and_table_limits_are_validated() -> None:
    """
    Older annotations remain readable, while table sizes and binding maps have explicit budgets.
    """
    old = annotation("text")
    del old["bindings"]
    assert read_annotations(json.dumps([old]))[0]["bindings"] == {}
    for bindings in ({"bad[name]": "x"}, {"good": []}, {"x": "y" * 513}):
        with pytest.raises(ValueError, match="bindings"):
            serialize([{**old, "bindings": bindings}])
    for text in ("\n" * 32, "\t" * 16):
        with pytest.raises(ValueError, match="tables"):
            serialize([{**annotation("table"), "text": text}])


def test_table_grid_dynamic_cells_preserve_input_and_cache_budgets() -> None:
    """
    Draw real cell grids, independent colors and changing formulas without altering the saved text.
    """
    item = annotation("table")
    item.update(points=[[.1, .1], [.9, .9]], text="Name\tValue\nHot\t{{ [Temperature.Maximum] :.1f }}",
                border_color="#ff0000", text_color="#00ff00")
    original = deepcopy(item)
    source = np.full((240, 320, 3), 35, np.uint8)
    first = source.copy()
    current = render_item(item, {"Temperature.Maximum": 25.})
    draw_object(first, current, IDENTITY)
    assert np.any(first[..., 2] > source[..., 2]) and np.any(first[..., 1] > source[..., 1])
    cache = masks_for((240, 320), current, IDENTITY)
    assert masks_for((240, 320), current, IDENTITY) is cache
    second = source.copy()
    draw_object(second, render_item(item, {"Temperature.Maximum": 99.}), IDENTITY)
    assert not np.array_equal(first, second) and np.all(source == 35) and item == original
    injected = render_item({**item, "text": "Name\t{{ [x] }}"}, {"x": "one\ntwo\tthree"})
    assert cells(injected["text"]) == [["Name", "one two three"]]


def test_runtime_counters_are_bounded_and_do_not_count_repeated_frames() -> None:
    """
    Track observed receipts rather than drawing loops, with monotonic timers and stale rate gaps.
    """
    runtime = ReadoutRuntime()
    for index in range(100):
        runtime.observe(index / 25, index / 25)
        runtime.observe(index / 25, index / 25)
    values = runtime.values(4.)
    assert runtime.frames == 100 and len(runtime.receipts) == 25
    assert values["Capture.FPS"] == pytest.approx(25.) and values["Session.Elapsed"] == 4.
    assert runtime.values(6.)["Capture.FPS"] == 0.
    assert runtime.values(3661.)["Session.Timer"] == "01:01:01"
