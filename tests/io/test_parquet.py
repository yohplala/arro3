from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from arro3.io import ParquetFile, read_parquet, write_parquet


def test_parquet_round_trip():
    table = pa.table({"a": [1, 2, 3, 4]})
    # We can't use tmp_path fixture with pytest-freethreading
    with TemporaryDirectory() as tmp_path:
        tmp_path = Path(tmp_path)
        write_parquet(table, tmp_path / "test.parquet")
        table_retour = pa.table(read_parquet(tmp_path / "test.parquet"))
        assert table == table_retour


def test_parquet_round_trip_bytes_io():
    table = pa.table({"a": [1, 2, 3, 4]})
    with BytesIO() as bio:
        write_parquet(table, bio)
        bio.seek(0)
        table_retour = pa.table(read_parquet(bio))
    assert table == table_retour


def test_copy_parquet_kv_metadata():
    metadata = {"hello": "world"}
    table = pa.table({"a": [1, 2, 3]})
    # We can't use tmp_path fixture with pytest-freethreading
    with TemporaryDirectory() as tmp_path:
        tmp_path = Path(tmp_path)
        pq_path = tmp_path / "test.parquet"
        write_parquet(
            table,
            pq_path,
            key_value_metadata=metadata,
            skip_arrow_metadata=True,
        )

        # Assert metadata was written, but arrow schema was not
        pq_meta = pq.read_metadata(pq_path).metadata
        assert pq_meta[b"hello"] == b"world"
        assert b"ARROW:schema" not in pq_meta.keys()

        # When reading with pyarrow, kv meta gets assigned to table
        pa_table = pq.read_table(pq_path)
        assert pa_table.schema.metadata[b"hello"] == b"world"

        reader = read_parquet(pq_path)
        assert reader.schema.metadata[b"hello"] == b"world"


def _write_multi_row_group(path: Path, table: pa.Table, rows_per_group: int) -> None:
    write_parquet(table, path, max_row_group_size=rows_per_group)


def test_parquet_file_accessors():
    table = pa.table({"a": list(range(10)), "b": [float(i) for i in range(10)]})
    with TemporaryDirectory() as tmp_path:
        pq_path = Path(tmp_path) / "test.parquet"
        _write_multi_row_group(pq_path, table, rows_per_group=4)

        pf = ParquetFile.open(pq_path)
        assert pf.num_rows == 10
        assert pf.num_row_groups == 3
        assert pf.num_columns == 2
        arrow_schema = pa.schema(pf.schema)
        assert arrow_schema.names == ["a", "b"]


def test_parquet_file_statistics_int():
    table = pa.table({"a": list(range(10))})
    with TemporaryDirectory() as tmp_path:
        pq_path = Path(tmp_path) / "test.parquet"
        _write_multi_row_group(pq_path, table, rows_per_group=4)

        pf = ParquetFile.open(pq_path)
        stats = pa.record_batch(pf.statistics("a"))
        assert stats.schema.names == ["min", "max", "null_count"]
        assert stats.num_rows == pf.num_row_groups

        # Ground-truth: pyarrow footer metadata
        pa_pf = pq.ParquetFile(pq_path)
        col_idx = pa_pf.schema_arrow.get_field_index("a")
        for rg_idx in range(pa_pf.num_row_groups):
            col_stats = pa_pf.metadata.row_group(rg_idx).column(col_idx).statistics
            assert stats["min"][rg_idx].as_py() == col_stats.min
            assert stats["max"][rg_idx].as_py() == col_stats.max
            assert stats["null_count"][rg_idx].as_py() == col_stats.null_count


def test_parquet_file_statistics_float_with_nulls():
    table = pa.table({"a": [1.0, None, 3.0, None, 5.0, 6.0, None, 8.0]})
    with TemporaryDirectory() as tmp_path:
        pq_path = Path(tmp_path) / "test.parquet"
        _write_multi_row_group(pq_path, table, rows_per_group=4)

        pf = ParquetFile.open(pq_path)
        stats = pa.record_batch(pf.statistics("a"))
        assert stats.num_rows == 2
        assert stats["min"].to_pylist() == [1.0, 5.0]
        assert stats["max"].to_pylist() == [3.0, 8.0]
        assert stats["null_count"].to_pylist() == [2, 1]


def test_parquet_file_statistics_first_last_row_group_bounds():
    """The downstream use case: cheaply read the min/max of a column by
    slicing the first row group's min and the last row group's max."""
    timestamps = pa.array(
        [pa.scalar(i, type=pa.timestamp("us")).as_py() for i in range(12)],
        type=pa.timestamp("us"),
    )
    table = pa.table({"timestamp": timestamps})
    with TemporaryDirectory() as tmp_path:
        pq_path = Path(tmp_path) / "test.parquet"
        _write_multi_row_group(pq_path, table, rows_per_group=4)

        pf = ParquetFile.open(pq_path)
        stats = pa.record_batch(pf.statistics("timestamp"))
        lo = stats["min"][0].as_py()
        hi = stats["max"][-1].as_py()
        assert lo == timestamps[0].as_py()
        assert hi == timestamps[-1].as_py()


def test_parquet_file_statistics_unknown_column_raises():
    table = pa.table({"a": [1, 2, 3]})
    with TemporaryDirectory() as tmp_path:
        pq_path = Path(tmp_path) / "test.parquet"
        write_parquet(table, pq_path)

        pf = ParquetFile.open(pq_path)
        with pytest.raises(Exception):
            pf.statistics("not_a_column")


def test_parquet_file_open_from_file_like():
    table = pa.table({"a": [1, 2, 3, 4]})
    with BytesIO() as bio:
        write_parquet(table, bio)
        bio.seek(0)
        pf = ParquetFile.open(bio)
        assert pf.num_rows == 4
        assert pf.num_row_groups == 1
        stats = pa.record_batch(pf.statistics("a"))
        assert stats["min"][0].as_py() == 1
        assert stats["max"][0].as_py() == 4
