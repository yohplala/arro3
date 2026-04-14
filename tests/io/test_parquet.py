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


def test_parquet_file_accessors():
    table = pa.table({"a": list(range(10)), "b": [float(i) for i in range(10)]})
    with TemporaryDirectory() as tmp_path:
        pq_path = Path(tmp_path) / "test.parquet"
        write_parquet(table, pq_path, max_row_group_size=4)

        pf = ParquetFile.open(pq_path)
        assert pf.num_rows == 10
        assert pf.num_row_groups == 3
        assert pf.num_columns == 2
        assert pf.schema_arrow.names == ["a", "b"]
        assert repr(pf) == (
            "arro3.io.ParquetFile(num_rows=10, num_row_groups=3, num_columns=2)"
        )


def test_parquet_file_statistics_int():
    table = pa.table({"a": list(range(10))})
    with TemporaryDirectory() as tmp_path:
        pq_path = Path(tmp_path) / "test.parquet"
        write_parquet(table, pq_path, max_row_group_size=4)

        pf = ParquetFile.open(pq_path)
        stats = pf.statistics("a")
        assert stats.schema.names == ["min", "max", "null_count"]
        assert stats.num_rows == pf.num_row_groups

        # Ground-truth: pyarrow footer metadata
        pa_pf = pq.ParquetFile(pq_path)
        col_idx = pa_pf.schema_arrow.get_field_index("a")
        min_col = stats.column("min")
        max_col = stats.column("max")
        null_col = stats.column("null_count")
        for rg_idx in range(pa_pf.num_row_groups):
            col_stats = pa_pf.metadata.row_group(rg_idx).column(col_idx).statistics
            assert min_col[rg_idx].as_py() == col_stats.min
            assert max_col[rg_idx].as_py() == col_stats.max
            assert null_col[rg_idx].as_py() == col_stats.null_count


def test_parquet_file_statistics_float_with_nulls():
    table = pa.table({"a": [1.0, None, 3.0, None, 5.0, 6.0, None, 8.0]})
    with TemporaryDirectory() as tmp_path:
        pq_path = Path(tmp_path) / "test.parquet"
        write_parquet(table, pq_path, max_row_group_size=4)

        pf = ParquetFile.open(pq_path)
        stats = pf.statistics("a")
        assert stats.num_rows == 2
        assert stats.column("min").to_pylist() == [1.0, 5.0]
        assert stats.column("max").to_pylist() == [3.0, 8.0]
        assert stats.column("null_count").to_pylist() == [2, 1]


def test_parquet_file_statistics_missing_null_counts_flag():
    """Both branches of the `missing_null_counts_as_zero` flag: when `True`
    (default) the `null_count` field is non-nullable; when `False` it is
    nullable."""
    table = pa.table({"a": [1, 2, 3, 4]})
    with TemporaryDirectory() as tmp_path:
        pq_path = Path(tmp_path) / "test.parquet"
        write_parquet(table, pq_path)

        pf = ParquetFile.open(pq_path)

        stats_true = pf.statistics("a")
        field_true = stats_true.schema.field("null_count")
        assert field_true.nullable is False

        stats_false = pf.statistics("a", missing_null_counts_as_zero=False)
        field_false = stats_false.schema.field("null_count")
        assert field_false.nullable is True
        # The values themselves should be identical for a file whose
        # stats include null counts.
        assert (
            stats_true.column("null_count").to_pylist()
            == stats_false.column("null_count").to_pylist()
        )


def test_parquet_file_statistics_first_last_row_group_bounds():
    """Downstream use case: cheaply read the min/max of a column by
    slicing the first row group's min and the last row group's max."""
    timestamps = pa.array(
        [pa.scalar(i, type=pa.timestamp("us")).as_py() for i in range(12)],
        type=pa.timestamp("us"),
    )
    table = pa.table({"timestamp": timestamps})
    with TemporaryDirectory() as tmp_path:
        pq_path = Path(tmp_path) / "test.parquet"
        write_parquet(table, pq_path, max_row_group_size=4)

        pf = ParquetFile.open(pq_path)
        stats = pf.statistics("timestamp")
        lo = stats.column("min")[0].as_py()
        hi = stats.column("max")[-1].as_py()
        assert lo == timestamps[0].as_py()
        assert hi == timestamps[-1].as_py()


def test_parquet_file_statistics_unknown_column_raises():
    table = pa.table({"a": [1, 2, 3]})
    with TemporaryDirectory() as tmp_path:
        pq_path = Path(tmp_path) / "test.parquet"
        write_parquet(table, pq_path)

        pf = ParquetFile.open(pq_path)
        with pytest.raises(Exception, match="not_a_column"):
            pf.statistics("not_a_column")


def test_parquet_file_open_from_file_like():
    table = pa.table({"a": [1, 2, 3, 4]})
    with BytesIO() as bio:
        write_parquet(table, bio)
        bio.seek(0)
        pf = ParquetFile.open(bio)
        assert pf.num_rows == 4
        assert pf.num_row_groups == 1
        stats = pf.statistics("a")
        assert stats.column("min")[0].as_py() == 1
        assert stats.column("max")[0].as_py() == 4


def test_parquet_file_open_skip_arrow_metadata():
    """With `skip_arrow_metadata=True`, the Arrow schema is reconstructed
    from the Parquet schema only, and the embedded `ARROW:schema` KV
    metadata is not decoded into the Arrow schema's metadata map."""
    table = pa.table({"a": [1, 2, 3]})
    with TemporaryDirectory() as tmp_path:
        pq_path = Path(tmp_path) / "test.parquet"
        # Write the file *with* the embedded Arrow schema (default).
        write_parquet(table, pq_path)

        pf_default = ParquetFile.open(pq_path)
        pf_skipped = ParquetFile.open(pq_path, skip_arrow_metadata=True)

        # Both surface the same logical schema (one int64 column).
        assert pf_default.schema_arrow.names == ["a"]
        assert pf_skipped.schema_arrow.names == ["a"]
        # And both let us read statistics.
        assert pf_skipped.statistics("a").column("min")[0].as_py() == 1


def test_parquet_file_open_page_index_optional():
    """`page_index=True` uses an Optional policy: if the file has no page
    index (which is the case for a default `write_parquet` call), `open`
    must not error."""
    table = pa.table({"a": list(range(8))})
    with TemporaryDirectory() as tmp_path:
        pq_path = Path(tmp_path) / "test.parquet"
        write_parquet(table, pq_path, max_row_group_size=4)

        # Would raise if we mapped `page_index=True` to Required and the
        # file has no page index.
        pf = ParquetFile.open(pq_path, page_index=True)
        assert pf.num_row_groups == 2
        assert pf.statistics("a").column("min").to_pylist() == [0, 4]
