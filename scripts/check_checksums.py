"""Генерация и проверка контрольных сумм (SHA-256) для data/ и models/.

Файлы в data/ и models/ хранятся в Git LFS: при клонировании и `git lfs pull`
содержимое может быть повреждено или не полностью скачано. Скрипт позволяет
создать манифест контрольных сумм (JSON) и затем проверить локальные файлы
против него.

Запуск из корня репозитория:

    uv run -m use_scripts.check_checksums generate   # создать манифест
    uv run -m use_scripts.check_checksums verify     # проверить файлы

или через Makefile:

    make checksum-gen
    make checksum-verify
"""

import hashlib
import json
import sys
from pathlib import Path

from tqdm import tqdm

CHUNK_SIZE = 1024 * 1024  # 1 MB, чтобы не грузить большие файлы целиком в память

DEFAULT_TARGETS = ("data", "models")
DEFAULT_MANIFEST = "results/checksums.json"


def file_sha256(path: Path) -> str:
    """Считает SHA-256 хеш файла, читая его блоками по CHUNK_SIZE байт."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_target_files(targets: list[Path]) -> list[Path]:
    """Собирает отсортированный список всех файлов из целевых папок."""
    files: list[Path] = []
    for target in targets:
        if not target.is_dir():
            raise FileNotFoundError(f"Целевая папка не найдена: {target}")
        files.extend(p for p in target.rglob("*") if p.is_file())
    return sorted(files)


def generate_checksums(
    targets: list[str] = list(DEFAULT_TARGETS),
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, str]:
    """Создает манифест {относительный_путь: sha256} для всех файлов из targets."""
    manifest_path = Path(manifest_path)
    target_paths = [Path(t) for t in targets]
    files = iter_target_files(target_paths)
    if not files:
        print(f"В целевых папках {targets} не найдено ни одного файла")
        return {}

    manifest: dict[str, str] = {}
    for file_path in tqdm(files, desc="Computing SHA-256"):
        manifest[file_path.as_posix()] = file_sha256(file_path)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)

    total_size = sum(p.stat().st_size for p in files)
    print(f"Манифест ({len(manifest)} файлов, {total_size / 1024 / 1024:.1f} MiB) сохранен в {manifest_path}")
    return manifest


def verify_checksums(
    targets: list[str] = list(DEFAULT_TARGETS),
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> bool:
    """Проверяет файлы из targets против манифеста.

    Возвращает True, если все контрольные суммы совпадают,
    False — если есть отсутствующие, измененные или лишние файлы.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Манифест не найден: {manifest_path}. Сначала выполните 'generate'.")

    with open(manifest_path, encoding="utf-8") as f:
        manifest: dict[str, str] = json.load(f)

    target_paths = [Path(t) for t in targets]
    actual_files = {p.as_posix() for p in iter_target_files(target_paths)}

    expected_files = set(manifest.keys())
    missing = sorted(expected_files - actual_files)
    extra = sorted(actual_files - expected_files)
    changed: list[str] = []

    for rel_path in tqdm(sorted(expected_files & actual_files), desc="Verifying SHA-256"):
        if file_sha256(Path(rel_path)) != manifest[rel_path]:
            changed.append(rel_path)

    for rel_path in missing:
        print(f"[MISSING] {rel_path}")
    for rel_path in changed:
        print(f"[CHANGED] {rel_path}")
    for rel_path in extra:
        print(f"[EXTRA]   {rel_path}")

    ok = not missing and not changed and not extra
    print(
        f"Проверено {len(expected_files & actual_files)} файлов: "
        f"измененных - {len(changed)}, отсутствующих - {len(missing)}, лишних - {len(extra)}"
    )
    print("[OK] Контрольные суммы совпадают" if ok else "[FAIL] Обнаружены расхождения с манифестом")
    return ok


def main() -> int:
    """Разбирает аргументы командной строки и запускает generate/verify."""
    if len(sys.argv) < 2 or sys.argv[1] not in {"generate", "verify"}:
        print("Использование: uv run -m use_scripts.check_checksums [generate|verify] [путь_манифеста]")
        return 2

    mode = sys.argv[1]
    manifest_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MANIFEST

    if mode == "generate":
        generate_checksums(manifest_path=manifest_path)
        return 0
    return 0 if verify_checksums(manifest_path=manifest_path) else 1


if __name__ == "__main__":
    sys.exit(main())
