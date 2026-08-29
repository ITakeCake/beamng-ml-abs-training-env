import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asset_installer as ai


def _make_assets(tmp_path):
    assets = tmp_path / "assets"
    (assets / "mods" / "mtb_ml_abs" / "lua").mkdir(parents=True)
    (assets / "mods" / "mtb_ml_abs" / "info.json").write_text('{"a":1}')
    (assets / "mods" / "mtb_ml_abs" / "lua" / "controller.lua").write_text("-- lua")
    (assets / "cars" / "etk800").mkdir(parents=True)
    (assets / "cars" / "etk800" / "Car1.pc").write_text('{"model":"etk800"}')
    return str(assets)


def test_iter_asset_files_maps_mods_and_cars_correctly(tmp_path):
    assets_dir = _make_assets(tmp_path)
    pairs = dict(ai.iter_asset_files(assets_dir))
    assert pairs[os.path.join("mods", "mtb_ml_abs", "info.json")] == \
        os.path.join("mods", "unpacked", "mtb_ml_abs", "info.json")
    assert pairs[os.path.join("cars", "etk800", "Car1.pc")] == \
        os.path.join("vehicles", "etk800", "Car1.pc")


def test_installed_status_reports_missing_ok_stale(tmp_path):
    assets_dir = _make_assets(tmp_path)
    userpath = str(tmp_path / "userpath")
    os.makedirs(userpath)

    status = ai.installed_status(assets_dir, userpath)
    assert all(v == "MISSING" for v in status.values())

    ai.install_assets(assets_dir, userpath)
    status = ai.installed_status(assets_dir, userpath)
    assert all(v == "OK" for v in status.values())

    # user/mod update changes a shipped file -> STALE
    dest_info = os.path.join(userpath, "mods", "unpacked", "mtb_ml_abs", "info.json")
    with open(dest_info, "w") as fh:
        fh.write('{"a":999}')
    status = ai.installed_status(assets_dir, userpath)
    assert status[os.path.join("mods", "unpacked", "mtb_ml_abs", "info.json")] == "STALE"


def test_install_assets_creates_dirs_and_copies_content(tmp_path):
    assets_dir = _make_assets(tmp_path)
    userpath = str(tmp_path / "userpath2")
    report = ai.install_assets(assets_dir, userpath)
    assert len(report) == 3
    dest = os.path.join(userpath, "vehicles", "etk800", "Car1.pc")
    assert os.path.isfile(dest)
    assert open(dest).read() == '{"model":"etk800"}'


def test_install_assets_reports_action_installed_then_updated(tmp_path):
    assets_dir = _make_assets(tmp_path)
    userpath = str(tmp_path / "userpath3")

    report1 = dict(ai.install_assets(assets_dir, userpath))
    assert all(v == "installed" for v in report1.values())

    report2 = dict(ai.install_assets(assets_dir, userpath))
    assert all(v == "unchanged" for v in report2.values())

    # modify a source asset -> next install reports "updated"
    src = os.path.join(assets_dir, "mods", "mtb_ml_abs", "info.json")
    with open(src, "w") as fh:
        fh.write('{"a":2}')
    report3 = dict(ai.install_assets(assets_dir, userpath))
    assert report3[os.path.join("mods", "unpacked", "mtb_ml_abs", "info.json")] == "updated"
