"""jbeam_generator writes the multi-model ML-ABS mod: one parent part per
supported car (declaring the model sub-slot + a controller-less placeholder
child) plus one child part per exported trained model. Pure string/dict
generation + real zip scanning -- no game imports."""
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
    assert slots["testcar"][0] == "testcar_ABS"


def test_find_abs_slot_types_skips_models_without_an_abs_slot(tmp_path):
    with zipfile.ZipFile(tmp_path / "noabs.zip", "w") as zf:
        zf.writestr("vehicles/noabs/info.json", json.dumps({"Name": "NoAbs", "Type": "Car"}))
        zf.writestr("vehicles/noabs/noabs.jbeam", json.dumps({"noabs_body": {"slotType": "x"}}))
    slots = find_abs_slot_types(str(tmp_path))
    assert "noabs" not in slots


def test_generate_parent_jbeam_declares_the_model_subslot_and_placeholder():
    j = generate_parent_jbeam("etk800", "etk_DSE_ABS")
    part_name = f"{PLACEHOLDER_PART_NAME}_etk800_parent"
    data = json.loads(j)
    part = next(iter(data.values()))
    assert part["slotType"] == "etk_DSE_ABS"
    slot_names = [row[0] for row in part["slots"][1:]]
    assert MODEL_SLOT_TYPE in slot_names


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
