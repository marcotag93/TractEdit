"""Small OBJ texture association and persistence contracts."""

from pathlib import Path

import pytest
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkIOImage import vtkPNGWriter

from tractedit_pkg.mesh_texture import (
    DEFAULT_BUDGETS,
    MeshTextureError,
    prepare_obj_textures,
)
from tractedit_pkg.session.mesh_texture import (
    prepare_saved_textures,
    texture_configuration,
)


@pytest.fixture
def textured_obj(tmp_path):
    image = vtkImageData()
    image.SetDimensions(2, 2, 1)
    image.AllocateScalars(3, 3)
    image.GetPointData().GetScalars().Fill(128)
    writer = vtkPNGWriter()
    writer.SetInputData(image)
    writer.SetFileName(str(tmp_path / "surface.png"))
    writer.Write()
    path = tmp_path / "surface.obj"
    path.write_text(
        "mtllib old.mtl\nusemtl photo\n"
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n",
        encoding="ascii",
    )
    path.with_suffix(".mtl").write_text(
        "newmtl photo\nmap_Kd old.png\n", encoding="ascii",
    )
    return path


def test_missing_references_use_compatible_same_stem_assets(textured_obj):
    result = prepare_obj_textures(textured_obj, gl_maximum=16384)
    material = result["materials"]["photo"]
    assert material["library"]["path"] == str(textured_obj.with_suffix(".mtl"))
    assert result["images"][material["image_key"]]["asset"]["path"] == str(
        textured_obj.with_suffix(".png")
    )
    assert material["image_origin"] == "automatic"


def test_existing_reference_wins_over_same_stem_image(textured_obj):
    original = textured_obj.parent / "old.png"
    original.write_bytes(textured_obj.with_suffix(".png").read_bytes())
    result = prepare_obj_textures(textured_obj, gl_maximum=16384)
    assert next(iter(result["images"].values()))["asset"]["path"] == str(original)


def test_ambiguous_images_and_incompatible_library_are_not_guessed(textured_obj):
    textured_obj.with_suffix(".jpg").write_bytes(b"not a candidate to prefer")
    result = prepare_obj_textures(textured_obj, gl_maximum=16384)
    assert not result["images"]
    textured_obj.with_suffix(".mtl").write_text(
        "newmtl another\nmap_Kd surface.png\n", encoding="ascii",
    )
    result = prepare_obj_textures(textured_obj, gl_maximum=16384)
    assert not result["libraries"]


def test_manual_image_survives_saved_restore_and_explicit_reload(textured_obj):
    selected = textured_obj.parent / "selected.png"
    selected.write_bytes(textured_obj.with_suffix(".png").read_bytes())
    result = prepare_obj_textures(
        textured_obj, gl_maximum=16384, image_overrides={"photo": str(selected)},
    )
    configuration, assets = texture_configuration(result, True)
    assert configuration["materials"]["photo"]["image_origin"] == "manual"
    restored = prepare_saved_textures(
        textured_obj, configuration, assets, gl_maximum=16384,
        budgets=DEFAULT_BUDGETS,
    )
    assert Path(next(iter(restored["images"].values()))["asset"]["path"]) == selected
    image = vtkImageData()
    image.SetDimensions(3, 2, 1)
    image.AllocateScalars(3, 3)
    image.GetPointData().GetScalars().Fill(64)
    writer = vtkPNGWriter()
    writer.SetInputData(image)
    writer.SetFileName(str(selected))
    writer.Write()
    with pytest.raises(MeshTextureError, match="asset_changed"):
        prepare_saved_textures(
            textured_obj, configuration, assets, gl_maximum=16384,
            budgets=DEFAULT_BUDGETS,
        )
    reloaded = prepare_saved_textures(
        textured_obj, configuration, assets, gl_maximum=16384, refresh=True,
        budgets=DEFAULT_BUDGETS,
    )
    asset = next(iter(reloaded["images"].values()))["asset"]
    assert asset["size"] == selected.stat().st_size
