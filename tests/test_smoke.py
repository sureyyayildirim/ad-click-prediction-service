def test_project_structure_exists():
    import os

    assert os.path.isdir("src")
    assert os.path.isdir("tests")
    assert os.path.isfile("README.md")
