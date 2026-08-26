import json
import tarfile

from posdiffusion.data.webdataset_builder import create_webdataset_from_kif


def test_create_webdataset_from_kif(tmp_path) -> None:
    kif_dir = tmp_path / "kif"
    output_dir = tmp_path / "shards"
    kif_dir.mkdir()
    (kif_dir / "game.kif").write_text(
        "1 ７六歩(77)\n2 ３四歩(33)\n3 投了\n",
        encoding="shift_jis",
    )

    create_webdataset_from_kif(kif_dir, output_dir, shard_size=1, max_moves=100)

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["total_shards"] == 1
    assert metadata["total_games"] == 1
    assert metadata["total_pairs"] == 2
    with tarfile.open(output_dir / "shard_00000.tar") as archive:
        names = set(archive.getnames())
    assert "00000_00000.positions.npy" in names
    assert "00000_00000.game_id.txt" in names
