# -*- coding: utf-8 -*-
"""Borrowed tractogram arrays and explicit component extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice
from typing import Any

import numpy as np


COLOR_MAPS = ("cividis", "viridis", "RdBu")

# Matplotlib palettes quantized to the byte precision of vtkLookupTable.
_PALETTES = {
    "cividis": bytes.fromhex(
        "00224e00234f00245100255300255400265600275800285900285b00295d002a5f002a61"
        "002b62002c64002c66002d68002e6a002e6c002f6d00306f003070003170003171013271"
        "0533710833700c34700f357012357014367016377018376f1a386f1c396f1e3a6f203a6f"
        "213b6e233c6e243c6e263d6e273e6e293f6e2a3f6d2b406d2d416d2e416d2f426d31436d"
        "32436d33446d34456c35456c36466c38476c39486c3a486c3b496c3c4a6c3d4a6c3e4b6c"
        "3f4c6c404c6c414d6c424e6c434e6c444f6c45506c46516c47516c48526c49536c4a536c"
        "4b546c4c556c4d556c4e566c4f576c50576c51586d52596d535a6d545a6d555b6d555c6d"
        "565c6d575d6d585e6d595e6e5a5f6e5b606e5c616e5d616e5e626e5e636f5f636f60646f"
        "61656f62656f636670646770656870656870666970676a71686a71696b716a6c716b6d72"
        "6c6d726c6e726d6f726e6f736f7073707173717274727274727374737475747475757575"
        "7676767777767777777878777979777a7a787b7a787c7b787d7c787e7c787e7d787f7e78"
        "807f78817f788280798381798482798582798683798784788885788985788a86788b8778"
        "8c88788d88788e89788f8a78908b78918b78928c78928d78938e78948e77958f77969077"
        "9791779892779992779a93769b94769c95769d95769e96769f9775a09875a19975a29975"
        "a39a74a49b74a59c74a69c74a79d73a89e73a99f73aaa073aba072aca172ada272aea371"
        "afa471b0a571b1a570b3a670b4a76fb5a86fb6a96fb7a96eb8aa6eb9ab6dbaac6dbbad6d"
        "bcae6cbdae6cbeaf6bbfb06bc0b16ac1b26ac2b369c3b369c4b468c5b568c6b667c7b767"
        "c8b866c9b965cbb965ccba64cdbb63cebc63cfbd62d0be62d1bf61d2c060d3c05fd4c15f"
        "d5c25ed6c35dd7c45cd9c55cdac65bdbc75adcc859ddc858dec958dfca57e0cb56e1cc55"
        "e2cd54e4ce53e5cf52e6d051e7d150e8d24fe9d34eead34cebd44bedd54aeed649efd748"
        "f0d846f1d945f2da44f3db42f5dc41f6dd3ff7de3ef8df3cf9e03afbe138fce236fde334"
        "fee434fee535fee636fee838"
    ),
    "viridis": bytes.fromhex(
        "44015444025645045745055946075a46085c460a5d460b5e470d60470e61471063471164"
        "47136548146748166848176948186a481a6c481b6d481c6e481d6f481f70482071482173"
        "482374482475482576482677482878482979472a7a472c7a472d7b472e7c472f7d46307e"
        "46327e46337f463480453581453781453882443983443a83443b84433d84433e85423f85"
        "4240864241864142874144874045884046883f47883f48893e49893e4a893e4c8a3d4d8a"
        "3d4e8a3c4f8a3c508b3b518b3b528b3a538b3a548c39558c39568c38588c38598c375a8c"
        "375b8d365c8d365d8d355e8d355f8d34608d34618d33628d33638d32648e32658e31668e"
        "31678e31688e30698e306a8e2f6b8e2f6c8e2e6d8e2e6e8e2e6f8e2d708e2d718e2c718e"
        "2c728e2c738e2b748e2b758e2a768e2a778e2a788e29798e297a8e297b8e287c8e287d8e"
        "277e8e277f8e27808e26818e26828e26828e25838e25848e25858e24868e24878e23888e"
        "23898e238a8d228b8d228c8d228d8d218e8d218f8d21908d21918c20928c20928c20938c"
        "1f948c1f958b1f968b1f978b1f988b1f998a1f9a8a1e9b8a1e9c891e9d891f9e891f9f88"
        "1fa0881fa1881fa1871fa28720a38620a48621a58521a68522a78522a88423a98324aa83"
        "25ab8225ac8226ad8127ad8128ae8029af7f2ab07f2cb17e2db27d2eb37c2fb47c31b57b"
        "32b67a34b67935b77937b87838b9773aba763bbb753dbc743fbc7340bd7242be7144bf70"
        "46c06f48c16e4ac16d4cc26c4ec36b50c46a52c56954c56856c66758c7655ac8645cc863"
        "5ec96260ca6063cb5f65cb5e67cc5c69cd5b6ccd5a6ece5870cf5773d05675d05477d153"
        "7ad1517cd2507fd34e81d34d84d44b86d54989d5488bd6468ed64590d74393d74195d840"
        "98d83e9bd93c9dd93ba0da39a2da37a5db36a8db34aadc32addc30b0dd2fb2dd2db5de2b"
        "b8de29bade28bddf26c0df25c2df23c5e021c8e020cae11fcde11dd0e11cd2e21bd5e21a"
        "d8e219dae319dde318dfe318e2e418e5e419e7e419eae51aece51befe51cf1e51df4e61e"
        "f6e620f8e621fbe723fde725"
    ),
    "RdBu": bytes.fromhex(
        "67001f6a011f6d02207003207304217605217906227c07227f0823810823840924870a24"
        "8a0b258d0c25900d26930e26960f279910279c11279f1228a21328a51429a81529ab162a"
        "ae172ab1182bb3192cb41c2db61f2eb72230b82531ba2832bb2a34bd2d35be3036bf3338"
        "c13639c2383ac43b3cc53e3dc6413ec84440c94741cb4942cc4c44ce4f45cf5246d05548"
        "d25849d35a4ad55d4cd6604dd7634fd86551da6853db6b55dc6e57dd7059de735cdf765e"
        "e17860e27b62e37e64e48066e58368e6866ae8896ce98b6eea8e70eb9172ec9374ee9677"
        "ef9979f09c7bf19e7df2a17ff3a481f4a683f5a886f5aa89f5ac8bf6af8ef6b191f6b394"
        "f7b596f7b799f7b99cf8bb9ef8bda1f8bfa4f9c2a7f9c4a9f9c6acfac8affacab1fbccb4"
        "fbceb7fbd0b9fcd3bcfcd5bffcd7c2fdd9c4fddbc7fddcc9fdddcbfcdecdfcdfcffce0d0"
        "fce2d2fbe3d4fbe4d6fbe5d8fbe6dafae7dcfae8defae9dffaeae1f9ebe3f9ede5f9eee7"
        "f9efe9f9f0ebf8f1edf8f2eff8f3f0f8f4f2f7f5f4f7f6f6f6f7f7f5f6f7f3f5f6f2f5f6"
        "f0f4f6eff3f5edf2f5ecf2f5eaf1f5e9f0f4e7f0f4e6eff4e4eef4e3edf3e1edf3e0ecf3"
        "deebf2ddebf2dbeaf2dae9f2d8e9f1d7e8f1d5e7f1d4e6f1d2e6f0d1e5f0cfe4efcce2ef"
        "cae1eec7e0edc5dfecc2ddecc0dcebbddbeabbdaeab8d8e9b6d7e8b3d6e8b1d5e7aed3e6"
        "acd2e5a9d1e5a7d0e4a5cee3a2cde3a0cce29dcbe19bc9e098c8e096c7df93c6de90c4dd"
        "8dc2dc8ac0db87beda84bcd981bad87eb8d77bb6d678b4d575b2d471b0d36eaed26bacd1"
        "68abd065a9cf62a7ce5fa5cd5ca3cb59a1ca569fc9529dc84f9bc74c99c64997c54695c4"
        "4393c34291c2408fc13f8ec03e8cbf3c8abe3b88be3a87bd3885bc3783bb3681ba3480b9"
        "337eb8327cb7307ab62f79b52e77b52c75b42b73b32a71b22870b1276eb0266caf246aae"
        "2369ad2267ac2065ab1f63a81e61a51d5fa21c5c9f1b5a9c1a5899195696185493175290"
        "15508d144e8a134c8712498411478110457e0f437b0e41790d3f760c3d730a3b7009386d"
        "08366a073467063264053061"
    ),
}


def configure_component_lut(lut, name):
    colors = np.frombuffer(_PALETTES[name], dtype=np.uint8).reshape(256, 3) / 255
    lut.SetNumberOfTableValues(len(colors))
    lut.Build()
    for index, color in enumerate(colors):
        lut.SetTableValue(index, *color, 1.0)
    lut.SetNanColor(0.5, 0.5, 0.5, 1.0)


@dataclass(frozen=True)
class DataArray:
    association: str
    name: str
    values: Any = field(repr=False, compare=False)
    group: str = ""

    @property
    def buffer(self) -> np.ndarray:
        if hasattr(self.values, "_data"):
            return self.values._data
        if self.association == "point":
            return np.asarray(self.values[0]) if len(self.values) else np.empty(0)
        return np.asarray(self.values)

    @property
    def components(self) -> int:
        return int(np.prod(self.buffer.shape[1:])) or 1

    @property
    def dtype(self) -> np.dtype:
        return self.buffer.dtype

    @property
    def shape(self) -> tuple[int, int]:
        if self.association == "point":
            lengths = getattr(self.values, "_lengths", None)
            count = (
                int(lengths.sum())
                if lengths is not None
                else sum(len(values) for values in self.values)
            )
        else:
            count = len(self.values)
        return count, self.components

    def gather(self, ids, component: int) -> np.ndarray:
        if not 0 <= component < self.components:
            raise ValueError("Invalid array component")
        if self.association == "point":
            arrays = [
                np.asarray(self.values[index]).reshape(-1, self.components)[
                    :, component
                ]
                for index in ids
            ]
            return np.concatenate(arrays) if arrays else np.empty(0, dtype=self.dtype)
        indices = np.asarray(ids, dtype=np.intp)
        return np.asarray(self.values)[indices].reshape(len(indices), self.components)[
            :, component
        ]

    def chunks(self, ids, component: int):
        iterator = iter(ids)
        while batch := list(islice(iterator, 4096)):
            if self.association == "point":
                for index in batch:
                    values = np.asarray(self.values[index]).reshape(-1, self.components)
                    for start in range(0, len(values), 65536):
                        yield values[start : start + 65536, component]
            else:
                yield self.gather(batch, component)


def collect_arrays(mw) -> list[DataArray]:
    arrays = [
        DataArray(association, name, values)
        for association, mapping in (
            ("point", getattr(mw, "scalar_data_per_point", None)),
            ("streamline", getattr(mw, "data_per_streamline", None)),
        )
        for name, values in (mapping or {}).items()
        if name != "_tractedit_bboxes"
    ]
    owner = getattr(mw, "trx_file_reference", None)
    arrays.extend(
        DataArray("group", name, values, group)
        for group, mapping in (getattr(owner, "data_per_group", None) or {}).items()
        for name, values in mapping.items()
    )
    return arrays


def active_array(mw) -> DataArray | None:
    display = getattr(mw, "scalar_display", None)
    if display is None:
        return None
    return next(
        (
            array
            for array in collect_arrays(mw)
            if array.name == mw.active_scalar_name
            and array.association == display["association"]
        ),
        None,
    )


def colour_range(minimum, maximum) -> tuple[float, float] | None:
    """Return bounds usable by both the floating-point LUT and the range sliders."""
    low, high = float(minimum), float(maximum)
    if minimum == maximum:
        low, high = low - 0.5, high + 0.5
    span = high - low
    if 0 < span < np.inf and np.isfinite(low + span):
        return low, high
    return None


def validate_scalar_display(display) -> None:
    if (
        not isinstance(display, dict)
        or set(display) != {"association", "component", "colormap"}
        or display["association"] not in ("point", "streamline")
        or type(display["component"]) is not int
        or display["component"] < 0
        or display["colormap"] not in COLOR_MAPS
    ):
        raise ValueError("Invalid scalar component display")


def array_statistics(
    array: DataArray,
    component: int,
    ids,
    cancelled,
    *,
    histogram=True,
) -> dict | None:
    count = missing = 0
    low, high = np.inf, -np.inf
    for values in array.chunks(ids, component):
        if cancelled():
            return None
        valid = values[np.isfinite(values)]
        missing += len(values) - len(valid)
        count += len(valid)
        if len(valid):
            low = min(low, valid.min().item())
            high = max(high, valid.max().item())
    if not count:
        return {"count": 0, "missing": missing, "histogram": None}
    bounds = (low, high) if low != high else (low - 0.5, high + 0.5)
    if not histogram:
        histogram = None
    elif low == high:
        histogram = np.zeros(64, dtype=np.int64)
        histogram[32] = count
    elif np.issubdtype(array.dtype, np.integer):
        # Integer thresholds avoid rounding large offsets or overflowing the span.
        edges = np.array(
            [low + ((high - low) * index + 63) // 64 for index in range(1, 64)],
            dtype=array.dtype,
        )
        histogram = np.zeros(64, dtype=np.int64)
        for values in array.chunks(ids, component):
            if cancelled():
                return None
            bins = np.searchsorted(edges, values, side="right")
            histogram += np.bincount(bins, minlength=64)
    elif not np.isfinite(bounds[1] - bounds[0]):
        histogram = None
    else:
        edges = np.linspace(*bounds, 65)
        histogram = np.zeros(64, dtype=np.int64)
        for values in array.chunks(ids, component):
            if cancelled():
                return None
            histogram += np.histogram(values[np.isfinite(values)], bins=edges)[0]
    return {
        "count": count,
        "missing": missing,
        "min": low,
        "max": high,
        "histogram": histogram,
    }
