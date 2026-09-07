from cosim_link import configure_diag_file, diag_write


def test_diagnostics_are_routed_to_a_new_run_local_file(tmp_path):
    path = tmp_path / "run" / "cosim_diag.log"
    configure_diag_file(path)
    diag_write("hello")
    assert path.read_text(encoding="utf-8") == "DIAG hello\n"


def test_diagnostic_router_refuses_to_truncate_existing_evidence(tmp_path):
    path = tmp_path / "cosim_diag.log"
    path.write_text("old\n", encoding="utf-8")
    try:
        configure_diag_file(path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing diagnostics were accepted for truncation")
    assert path.read_text(encoding="utf-8") == "old\n"
