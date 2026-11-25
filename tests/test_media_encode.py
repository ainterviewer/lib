"""A guide image is read from its project's image directory and nowhere
else: its name is written by whoever edits the guide."""

from uuid import uuid4

import pytest

from ainterviewer.interview_guides.media import Image
from ainterviewer.settings import settings


def _image(name: str) -> Image:
    return Image(name=name, data=None, description="", alt="")


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.storage, "base_path", tmp_path)
    project_id = uuid4()
    images = settings.storage.project_storage.image_path(project_id)
    (images / "photo.png").write_bytes(b"png")
    (tmp_path / "secret.txt").write_bytes(b"secret")
    return project_id


def test_an_image_is_read_from_the_project_s_images(project):
    image = _image("photo.png")

    image.encode(project)

    assert image.data is not None
    assert image.data.endswith(",cG5n")


@pytest.mark.parametrize(
    "name", ["../../../secret.txt", "..\\..\\..\\secret.txt", "/etc/../secret.txt"]
)
def test_a_name_with_a_path_reads_nothing_outside(project, name):
    image = _image(name)

    with pytest.raises(FileNotFoundError):
        image.encode(project)

    assert image.data is None


@pytest.mark.parametrize("name", ["", ".", ".."])
def test_a_name_that_is_no_file_is_refused(project, name):
    with pytest.raises(FileNotFoundError):
        _image(name).encode(project)
