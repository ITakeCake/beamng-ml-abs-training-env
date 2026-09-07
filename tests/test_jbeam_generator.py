"""jbeam_generator writes the multi-model ML-ABS mod: one parent part per
supported car (declaring the model sub-slot + a controller-less placeholder
child) plus one child part per exported trained model. Pure string/dict
generation + real zip scanning, no game imports."""
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jbeam_generator import (
    find_abs_slot_types, generate_parent_jbeam, generate_model_jbeam,
    MODEL_SLOT_TYPE, PLACEHOLDER_PART_NAME,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _make_zip(path, model, slot_type):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"vehicles/{model}/info.json",
                    json.dumps({"Name": model.upper(), "Type": "Car"}))
        zf.writestr(f"vehicles/{model}/{model}_brakes.jbeam", json.dumps({
            f"{model}_ABS": {
                "information": {"name": "Anti-Lock Braking System", "value": 250},
                "slotType": slot_type,
            }
        }))


def test_find_abs_slot_types_reads_real_zip_shape(tmp_path):
    _make_zip(tmp_path / "testcar.zip", "testcar", "testcar_ABS")
    slots = find_abs_slot_types(str(tmp_path))
    assert slots["testcar"][0] == ["testcar_ABS"]


def test_find_abs_slot_types_collects_sockets_offered_not_just_declared(tmp_path):
    """A socket can exist purely as an OFFER in another part's `slots` row --
    which is how the whole ETK family's real socket (etk_DSE_ABS) appears.
    Scanning slotType declarations alone finds a name nothing plugs into."""
    with zipfile.ZipFile(tmp_path / "offered.zip", "w") as zf:
        zf.writestr("vehicles/offered/info.json",
                    json.dumps({"Name": "Offered", "Type": "Car"}))
        zf.writestr("vehicles/offered/offered.jbeam", json.dumps({
            "offered_electrics": {
                "slotType": "offered_DSE",
                "slots": [["type", "default", "description"],
                          ["offered_DSE_ABS", "offered_stock_ABS", "ABS"]],
            }
        }))
    assert find_abs_slot_types(str(tmp_path))["offered"][0] == ["offered_DSE_ABS"]


def test_find_abs_slot_types_includes_shared_sockets_from_common_zip(tmp_path):
    """etk800's own zip never names etk_DSE_ABS; common.zip does. Missing it is
    what made the generated part load and never appear in the parts menu."""
    _make_zip(tmp_path / "etk800.zip", "etk800", "etk800_ABS")
    with zipfile.ZipFile(tmp_path / "common.zip", "w") as zf:
        zf.writestr("vehicles/common/etk/etk_dse.jbeam", json.dumps({
            "etk_DSE": {"slots": [["type", "default", "description"],
                                  ["etk_DSE_ABS", "etk_DSE_ABS", "ABS"]]}
        }))
    slots = find_abs_slot_types(str(tmp_path))["etk800"][0]
    assert "etk_DSE_ABS" in slots and "etk800_ABS" in slots


def test_find_abs_slot_types_skips_models_without_an_abs_slot(tmp_path):
    with zipfile.ZipFile(tmp_path / "noabs.zip", "w") as zf:
        zf.writestr("vehicles/noabs/info.json", json.dumps({"Name": "NoAbs", "Type": "Car"}))
        zf.writestr("vehicles/noabs/noabs.jbeam", json.dumps({"noabs_body": {"slotType": "x"}}))
    slots = find_abs_slot_types(str(tmp_path))
    assert "noabs" not in slots


def _parents(data):
    return {name: p for name, p in data.items() if "slots" in p}


def test_generate_parent_jbeam_declares_the_model_subslot_and_placeholder():
    data = json.loads(generate_parent_jbeam("etk800", "etk_DSE_ABS"))
    parents = _parents(data)
    assert len(parents) == 1
    part = next(iter(parents.values()))
    assert part["slotType"] == "etk_DSE_ABS"
    assert MODEL_SLOT_TYPE in [row[0] for row in part["slots"][1:]]
    placeholder = data[f"{PLACEHOLDER_PART_NAME}_etk800"]
    assert placeholder["slotType"] == MODEL_SLOT_TYPE


def test_one_variant_is_emitted_per_candidate_slot():
    """A car's real socket is not identifiable from its files alone, and a wrong
    guess is silent, the part loads and never appears. Unused variants are
    inert, since jbeam drops a part whose slotType nothing offers."""
    data = json.loads(generate_parent_jbeam(
        "etk800", ["etk800_ABS", "etk_DSE_ABS", "pickup_ABS"]))
    parents = _parents(data)
    assert len(parents) == 3
    assert {p["slotType"] for p in parents.values()} == {
        "etk800_ABS", "etk_DSE_ABS", "pickup_ABS"}


def test_every_variant_shares_one_child_slot_and_placeholder():
    """So a model exported ONCE is offered by whichever variant the car uses --
    otherwise each variant would need its own copy of every model."""
    data = json.loads(generate_parent_jbeam("etk800", ["etk800_ABS", "etk_DSE_ABS"]))
    placeholders = [n for n, p in data.items() if "slots" not in p]
    assert placeholders == [f"{PLACEHOLDER_PART_NAME}_etk800"]
    for part in _parents(data).values():
        assert part["slots"][1] == [MODEL_SLOT_TYPE, placeholders[0], "ML ABS Model"]


def test_all_variants_are_named_ML_ABS_so_only_one_shows_per_car():
    data = json.loads(generate_parent_jbeam("etk800", ["etk800_ABS", "etk_DSE_ABS"]))
    assert {p["information"]["name"] for p in _parents(data).values()} == {"ML ABS"}


def test_duplicate_slots_do_not_produce_duplicate_parts():
    data = json.loads(generate_parent_jbeam("bx", ["bx_ABS", "bx_ABS"]))
    assert len(_parents(data)) == 1


def test_generate_parent_jbeam_is_valid_json():
    j = generate_parent_jbeam("bx", "bx_ABS")
    json.loads(j)   # must not raise


def test_generate_model_jbeam_references_the_weights_module():
    j = generate_model_jbeam("etk800", "residual_sac_r1", "mlabs_w_residual_sac_r1")
    data = json.loads(j)
    part = next(iter(data.values()))
    assert part["slotType"] == MODEL_SLOT_TYPE
    controller_row = part["controller"][1]
    assert controller_row[0] == "MTB-ML-ABS"
    assert controller_row[1]["weights"] == "mlabs_w_residual_sac_r1"


def test_generate_model_jbeam_is_valid_json():
    j = generate_model_jbeam("bx", "run2", "mlabs_w_run2")
    json.loads(j)
