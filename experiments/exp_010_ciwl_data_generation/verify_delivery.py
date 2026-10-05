"""Verify the completed result archive and its manifest; no model computation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import tarfile

work = Path(__file__).resolve().parent
archive = work / "data_results.tar.gz"
expected = "87af08fc6ad5e30f0eff6a98ad3682ce219946e6b1a0209d3b24c1f5ad980f27"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


actual = sha(archive)
assert actual == expected, "Incomplete or mismatched archive; do not extract"
with tarfile.open(archive, "r:gz") as bundle:
    for member in bundle.getmembers():
        target = (work / member.name).resolve()
        assert target.is_relative_to(work), member.name
        assert member.isfile() or member.isdir(), member.name
    bundle.extractall(work)

verified = []
for line in (work / "sha256.txt").read_text().splitlines():
    digest, name = line.split(maxsplit=1)
    path = (work / name.lstrip("*")).resolve()
    assert path.is_relative_to(work), name
    assert sha(path) == digest, name
    verified.append(name)

audit = json.loads((work / "outputs/audit.json").read_text())
assert audit["status"] == "completed_data_generation_and_audit"
assert audit["sequences"] == 5000
assert not audit["model_training_performed"]
assert not audit["ciwl_improvement_measured"]
assert (work / "exit_code").read_text().strip() == "0"
report = {
    "status": "completed_local_delivery_verified",
    "verified_utc": datetime.now(timezone.utc).isoformat(),
    "archive_sha256": actual,
    "archive_bytes": archive.stat().st_size,
    "manifest_files_verified": len(verified),
    "files": verified,
    "remote_task_exit_code": 0,
    "sequences": audit["sequences"],
    "delivery_method": "SFTP resumed original archive after SSH recovery",
    "no_data_regeneration": True,
}
(work / "local_delivery_verification.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
(work / "transfer_status.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
print("Actual generated first sequence:")
print(json.dumps(json.loads((work / "outputs/preview.json").read_text())[0]))
