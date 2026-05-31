import numpy as np
import pytest
from asl.library import load_templates


def test_groups_exemplars_by_gloss(tmp_path):
    np.save(tmp_path / "HELLO__0.npy", np.zeros((3, 2)))
    np.save(tmp_path / "HELLO__1.npy", np.ones((4, 2)))
    np.save(tmp_path / "BYE__0.npy", np.zeros((2, 2)))

    templates = load_templates(tmp_path)

    assert set(templates) == {"HELLO", "BYE"}
    assert len(templates["HELLO"]) == 2
    assert len(templates["BYE"]) == 1


def test_empty_directory_raises(tmp_path):
    with pytest.raises(ValueError):
        load_templates(tmp_path)


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_templates(tmp_path / "does_not_exist")


def test_empty_array_file_raises(tmp_path):
    np.save(tmp_path / "HELLO__0.npy", np.zeros((0, 2)))
    with pytest.raises(ValueError):
        load_templates(tmp_path)


def test_inconsistent_shape_exemplars_raise(tmp_path):
    np.save(tmp_path / "HELLO__0.npy", np.zeros((3, 2)))
    np.save(tmp_path / "HELLO__1.npy", np.zeros((3, 5)))
    with pytest.raises(ValueError):
        load_templates(tmp_path)


def test_gloss_with_double_underscore_parses(tmp_path):
    np.save(tmp_path / "THANK__YOU__0.npy", np.zeros((3, 2)))
    np.save(tmp_path / "THANK__YOU__1.npy", np.ones((4, 2)))

    templates = load_templates(tmp_path)

    assert set(templates) == {"THANK__YOU"}
    assert len(templates["THANK__YOU"]) == 2
