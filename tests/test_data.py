from cascade.data import cross_lab_replication_pairs, load_effect_matrix


def _write_matrix(path, rows):
    with open(path, "w") as fh:
        fh.write("cell_line,GENEA,GENEB,GENEC\n")
        for name, a, b, c in rows:
            fh.write(f"{name},{a},{b},{c}\n")


def test_load_effect_matrix(tmp_path):
    p = tmp_path / "broad.csv"
    _write_matrix(p, [("CL1", -1.5, 0.1, 2.0), ("CL2", -0.2, "", 0.3)])
    screens = load_effect_matrix(str(p), lab_id="broad", default_variance=0.04)
    assert len(screens) == 2
    cl1 = next(s for s in screens if s.context.cell_line == "CL1")
    assert cl1.effects["GENEA"].beta == -1.5
    assert cl1.effects["GENEA"].variance == 0.04
    # blank cell skipped
    cl2 = next(s for s in screens if s.context.cell_line == "CL2")
    assert "GENEB" not in cl2.effects
    assert "GENEA" in cl2.effects


def test_cross_lab_pairs_label_replication(tmp_path):
    broad = tmp_path / "broad.csv"
    sanger = tmp_path / "sanger.csv"
    # GENEA: strong hit in both → replicates. GENEC: hit in broad, not sanger → no.
    _write_matrix(broad, [("CL1", -1.5, 0.0, -2.0)])
    _write_matrix(sanger, [("CL1", -1.3, 0.0, 0.1)])
    sa = load_effect_matrix(str(broad), "broad")
    sb = load_effect_matrix(str(sanger), "sanger")
    pairs = cross_lab_replication_pairs(sa, sb, hit_threshold=0.5)
    by_gene = {p.gene: p for p in pairs}
    assert by_gene["GENEA"].label is True  # replicated
    assert by_gene["GENEC"].label is False  # did not replicate
    assert "GENEB" not in by_gene  # not a hit in A → no pair


def test_cross_lab_only_shared_cell_lines(tmp_path):
    broad = tmp_path / "b.csv"
    sanger = tmp_path / "s.csv"
    _write_matrix(broad, [("CL1", -1.5, 0.0, 0.0)])
    _write_matrix(sanger, [("CL2", -1.5, 0.0, 0.0)])  # different cell line
    sa = load_effect_matrix(str(broad), "broad")
    sb = load_effect_matrix(str(sanger), "sanger")
    pairs = cross_lab_replication_pairs(sa, sb, same_cell_line_only=True)
    assert pairs == []  # no shared cell line → no pairs
