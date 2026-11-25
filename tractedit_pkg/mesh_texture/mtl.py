import math
from pathlib import Path
from ..data_contracts import MeshTextureDiagnostic, MeshTextureMap, MeshTextureMaterial
from .assets import _NUMBER, MeshTextureError, _check_cancel, _local_reference


def _tokens(record: str, *, material_record=False) -> list[tuple[str, int, int, bool]]:
    tokens = []
    position = 0
    while position < len(record):
        if record[position].isspace():
            position += 1
            continue
        if record[position] == "#":
            break
        start = position
        quoted = record[position] == '"'
        if quoted:
            position += 1
            end = record.find('"', position)
            if end < 0:
                raise MeshTextureError(
                    "unterminated_quote", "Unterminated double quote"
                )
            value = record[position:end]
            position = end + 1
            if position < len(record) and not (
                record[position].isspace() or record[position] == "#"
            ):
                raise MeshTextureError(
                    "invalid_quote", "Quoted tokens must be separated"
                )
        else:
            literal_hash = material_record and len(tokens) == 1 and tokens[0][0] in (
                "usemtl", "newmtl",
            )
            while position < len(record) and not record[position].isspace():
                if record[position] == "#" and not literal_hash:
                    break
                if record[position] == '"':
                    raise MeshTextureError(
                        "invalid_quote", "Quotes must enclose a complete token"
                    )
                position += 1
            value = record[start:position]
        tokens.append((value, start, position, quoted))
    return tokens


def _number(token: str) -> float:
    if token.lower().strip("+-") in ("nan", "inf", "infinity"):
        raise MeshTextureError("nonfinite_number", "Material numbers must be finite")
    if _NUMBER.fullmatch(token) is None:
        raise MeshTextureError("invalid_number", "Invalid material number")
    value = float(token)
    if not math.isfinite(value):
        raise MeshTextureError("nonfinite_number", "Material numbers must be finite")
    return value


def replacement_texture_map(path) -> MeshTextureMap:
    return {
        "path": _local_reference(str(path)),
        "scale": (1.0, 1.0, 1.0), "offset": (0.0, 0.0, 0.0), "clamp": False,
    }


def parse_texture_map(record: str) -> MeshTextureMap:
    tokens = _tokens(record)
    scale, offset, clamp = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), False
    seen = set()
    position = 0
    while (
        position < len(tokens)
        and not tokens[position][3]
        and tokens[position][0].startswith("-")
    ):
        option = tokens[position][0]
        position += 1
        if option not in ("-s", "-o", "-clamp"):
            raise MeshTextureError(
                "unsupported_option", f"Unsupported map_Kd option: {option}"
            )
        if option in seen:
            raise MeshTextureError(
                "duplicate_option", f"Repeated map_Kd option: {option}"
            )
        seen.add(option)
        if option == "-clamp":
            if position >= len(tokens) or tokens[position][0] not in ("on", "off"):
                raise MeshTextureError("invalid_clamp", "-clamp requires on or off")
            clamp = tokens[position][0] == "on"
            position += 1
        else:
            values = []
            while position < len(tokens) and not tokens[position][3]:
                token = tokens[position][0]
                if _NUMBER.fullmatch(token) is None and token.lower().strip(
                    "+-"
                ) not in ("nan", "inf", "infinity"):
                    break
                values.append(_number(token))
                position += 1
            if not 1 <= len(values) <= 3:
                raise MeshTextureError(
                    "option_arity", f"{option} requires one to three numbers"
                )
            default = 1.0 if option == "-s" else 0.0
            values += [default] * (3 - len(values))
            if values[2] != default:
                raise MeshTextureError(
                    "unsupported_3d_transform",
                    "Third UV transform component is unsupported",
                )
            if option == "-s":
                scale = tuple(values)
            else:
                offset = tuple(values)
    if position == len(tokens):
        raise MeshTextureError("missing_map_path", "map_Kd requires an image path")
    remaining = tokens[position:]
    if any(item[3] for item in remaining):
        if len(remaining) != 1 or not remaining[0][3]:
            raise MeshTextureError("invalid_quote", "Quote the complete image path")
        path = remaining[0][0]
    else:
        path = record[remaining[0][1] : remaining[-1][2]]
    return {
        "path": _local_reference(path),
        "scale": scale,
        "offset": offset,
        "clamp": clamp,
    }


class _Diagnostics:
    def __init__(self, budgets):
        self.budgets = budgets
        self.items: list[MeshTextureDiagnostic] = []

    def add(self, code, message):
        if len(self.items) >= self.budgets.diagnostic_count:
            if code == "ignored_mtl_record":
                return
            index = next((
                index for index, item in enumerate(self.items)
                if item["code"] == "ignored_mtl_record"
            ), None)
            if index is None:
                index = len(self.items) - 1
                code = "diagnostics_omitted"
                message = "Additional texture problems were omitted."
            del self.items[index]
        bounded = (
            str(message)
            .encode("utf-8")[: self.budgets.diagnostic_bytes]
            .decode("utf-8", errors="ignore")
        )
        self.items.append({"code": code, "message": bounded})


def _material_name(tokens):
    if (
        len(tokens) != 1
        or not tokens[0][0].isascii()
        or not tokens[0][0]
        or len(tokens[0][0]) > 99
        or any(character.isspace() for character in tokens[0][0])
    ):
        raise MeshTextureError(
            "invalid_material_name",
            "Material names require one ASCII token of at most 99 bytes",
        )
    return tokens[0][0]


def _empty_material(name, library=None) -> MeshTextureMaterial:
    return {
        "name": name,
        "library": library,
        "diffuse_map": None,
        "diffuse_color": (1.0, 1.0, 1.0),
        "image_key": None,
        "unavailable_reason": None,
    }


def _read_mtl(asset, materials, declared, counts, budgets, diagnostics, cancelled):
    current = None
    properties = set()
    with Path(asset["path"]).open("rb") as stream:
        line_number = 0
        while True:
            _check_cancel(cancelled)
            raw = stream.readline(budgets.mtl_line_bytes + 1)
            if not raw:
                break
            line_number += 1
            counts[0] += len(raw)
            if len(raw) > budgets.mtl_line_bytes:
                raise MeshTextureError(
                    "mtl_line_limit", "MTL physical line exceeds its byte limit"
                )
            if counts[0] > budgets.mtl_bytes:
                raise MeshTextureError(
                    "mtl_byte_limit", "MTL data exceeds its per-layer byte limit"
                )
            if raw.lstrip().startswith(b"newmtl"):
                current = None
            try:
                text = raw.decode("utf-8-sig" if line_number == 1 else "utf-8")
                if any(
                    ord(character) < 32 and not character.isspace()
                    for character in text
                ):
                    raise MeshTextureError(
                        "invalid_mtl_text", "Unsupported MTL control character"
                    )
                tokens = _tokens(text, material_record=True)
                if not tokens:
                    continue
                counts[1] += 1
                if counts[1] > budgets.mtl_records:
                    raise MeshTextureError(
                        "mtl_record_limit", "MTL record count exceeds its limit"
                    )
                command = tokens[0][0]
                if command == "newmtl":
                    current = None
                    name = _material_name(tokens[1:])
                    counts[2] += 1
                    if counts[2] > budgets.max_materials:
                        raise MeshTextureError(
                            "material_limit", "MTL material count exceeds its limit"
                        )
                    current = _empty_material(name, asset)
                    properties = set()
                    if name in declared:
                        materials[name]["unavailable_reason"] = "ambiguous_material"
                        current["unavailable_reason"] = "ambiguous_material"
                        diagnostics.add(
                            "ambiguous_material", f"Duplicate material name: {name}"
                        )
                    else:
                        declared.add(name)
                        materials[name] = current
                        if len(materials) > budgets.max_materials:
                            raise MeshTextureError(
                                "material_limit",
                                "Combined material count exceeds its limit",
                            )
                elif current is None:
                    diagnostics.add(
                        "material_record_without_name",
                        f"MTL record before newmtl: {command}",
                    )
                elif command in ("map_Kd", "Kd", "d", "Tr"):
                    if command in properties:
                        code = (
                            "ambiguous_map"
                            if command == "map_Kd"
                            else "duplicate_property"
                        )
                        raise MeshTextureError(
                            code, f"Repeated material property: {command}"
                        )
                    if command in ("d", "Tr") and properties.intersection({"d", "Tr"}):
                        raise MeshTextureError(
                            "conflicting_material_alpha", "d and Tr cannot be combined"
                        )
                    properties.add(command)
                    if command == "map_Kd":
                        current["diffuse_map"] = parse_texture_map(text[tokens[0][2] :])
                    else:
                        required = 3 if command == "Kd" else 1
                        if len(tokens) != required + 1:
                            raise MeshTextureError(
                                "material_arity",
                                f"{command} requires {required} numbers",
                            )
                        values = tuple(_number(token[0]) for token in tokens[1:])
                        if any(not 0 <= value <= 1 for value in values):
                            raise MeshTextureError(
                                "material_range", "Material values must lie in [0, 1]"
                            )
                        if command == "Kd":
                            current["diffuse_color"] = values
                        elif values[0] != (1 if command == "d" else 0):
                            raise MeshTextureError(
                                "unsupported_material_alpha",
                                "Only opaque d/Tr defaults are supported",
                            )
                else:
                    diagnostics.add(
                        "ignored_mtl_record", f"Ignored material directive: {command}"
                    )
            except UnicodeDecodeError as error:
                raise MeshTextureError(
                    "invalid_mtl_text", "MTL must contain valid UTF-8"
                ) from error
            except MeshTextureError as error:
                if isinstance(error, MeshTextureError) and error.code.endswith("limit"):
                    raise
                code = getattr(error, "code", "invalid_mtl_text")
                diagnostics.add(
                    code, f"{Path(asset['path']).name}:{line_number}: {error}"
                )
                if current is not None:
                    current["unavailable_reason"] = (
                        current["unavailable_reason"] or code
                    )
