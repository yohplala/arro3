use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;

use arrow_array::{Array, RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow_schema::{Field, Schema};
use parquet::arrow::arrow_reader::statistics::StatisticsConverter;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
};
use parquet::arrow::arrow_writer::ArrowWriterOptions;
use parquet::arrow::async_reader::ParquetObjectReader;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, Encoding};
use parquet::file::metadata::{KeyValue, PageIndexPolicy};
use parquet::file::properties::{WriterProperties, WriterVersion};
use parquet::schema::types::ColumnPath;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyType;
use pyo3_arrow::error::PyArrowResult;
use pyo3_arrow::export::{Arro3RecordBatch, Arro3RecordBatchReader, Arro3Schema};
use pyo3_arrow::input::AnyRecordBatch;
use pyo3_arrow::{PyRecordBatchReader, PyTable};
use pyo3_object_store::PyObjectStore;

use crate::error::Arro3IoResult;
use crate::utils::{FileReader, FileWriter};

#[pyfunction]
pub fn read_parquet(file: FileReader) -> PyArrowResult<Arro3RecordBatchReader> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();

    let metadata = builder.schema().metadata().clone();
    let reader = builder.build().unwrap();

    // Add source schema metadata onto reader's schema. The original schema is not valid
    // with a given column projection, but we want to persist the source's metadata.
    let arrow_schema = Arc::new(reader.schema().as_ref().clone().with_metadata(metadata));

    // Create a new iterator with the arrow schema specifically
    //
    // Passing ParquetRecordBatchReader directly to PyRecordBatchReader::new loses schema
    // metadata
    //
    // https://docs.rs/parquet/latest/parquet/arrow/arrow_reader/struct.ParquetRecordBatchReader.html#method.schema
    // https://github.com/apache/arrow-rs/pull/5135
    let iter = Box::new(RecordBatchIterator::new(reader, arrow_schema));
    Ok(PyRecordBatchReader::new(iter).into())
}

#[pyfunction]
#[pyo3(signature = (path, *, store))]
pub fn read_parquet_async<'py>(
    py: Python<'py>,
    path: String,
    store: PyObjectStore,
) -> PyArrowResult<Bound<'py, PyAny>> {
    let fut = pyo3_async_runtimes::tokio::future_into_py(py, async move {
        Ok(read_parquet_async_inner(store.into_inner(), path).await?)
    })?;

    Ok(fut)
}

async fn read_parquet_async_inner(
    store: Arc<dyn object_store::ObjectStore>,
    path: String,
) -> Arro3IoResult<PyTable> {
    use futures::TryStreamExt;
    use parquet::arrow::ParquetRecordBatchStreamBuilder;

    let object_reader = ParquetObjectReader::new(store, path.into());
    let builder = ParquetRecordBatchStreamBuilder::new(object_reader).await?;

    let metadata = builder.schema().metadata().clone();
    let reader = builder.build()?;

    let arrow_schema = Arc::new(reader.schema().as_ref().clone().with_metadata(metadata));

    let batches = reader.try_collect::<Vec<_>>().await?;
    Ok(PyTable::try_new(batches, arrow_schema)?)
}

/// A Parquet file opened for metadata inspection.
///
/// This loads only the Parquet footer (no data pages) and exposes row-group
/// level metadata, including per-row-group statistics via [`PyParquetFile::statistics`].
#[pyclass(module = "arro3.io", name = "ParquetFile", subclass, frozen)]
pub(crate) struct PyParquetFile {
    meta: ArrowReaderMetadata,
}

#[pymethods]
impl PyParquetFile {
    /// Open a Parquet file and read its footer metadata.
    #[classmethod]
    #[pyo3(signature = (file, *, skip_arrow_metadata = false, page_index = false))]
    fn open(
        _cls: &Bound<PyType>,
        mut file: FileReader,
        skip_arrow_metadata: bool,
        page_index: bool,
    ) -> Arro3IoResult<Self> {
        // `PageIndexPolicy::Optional` matches the user expectation of
        // "load the page index if present, otherwise skip". The upstream
        // `From<bool>` impl maps `true` to `Required`, which errors if the
        // index is missing — a surprising footgun for users opting in.
        let page_index_policy = if page_index {
            PageIndexPolicy::Optional
        } else {
            PageIndexPolicy::Skip
        };
        let options = ArrowReaderOptions::new()
            .with_skip_arrow_metadata(skip_arrow_metadata)
            .with_page_index_policy(page_index_policy);
        let meta = ArrowReaderMetadata::load(&mut file, options)?;
        Ok(Self { meta })
    }

    /// The Arrow schema of this Parquet file.
    #[getter]
    fn schema_arrow(&self) -> Arro3Schema {
        self.meta.schema().clone().into()
    }

    /// The total number of rows in this Parquet file.
    #[getter]
    fn num_rows(&self) -> i64 {
        self.meta.metadata().file_metadata().num_rows()
    }

    /// The number of row groups in this Parquet file.
    #[getter]
    fn num_row_groups(&self) -> usize {
        self.meta.metadata().num_row_groups()
    }

    /// The number of columns in this Parquet file.
    #[getter]
    fn num_columns(&self) -> usize {
        self.meta.schema().fields().len()
    }

    fn __repr__(&self) -> String {
        format!(
            "arro3.io.ParquetFile(num_rows={}, num_row_groups={}, num_columns={})",
            self.num_rows(),
            self.num_row_groups(),
            self.num_columns(),
        )
    }

    /// Row-group statistics for a single column.
    ///
    /// Returns a `RecordBatch` with one row per row group and the columns
    /// `min`, `max` and `null_count`. `min` and `max` take the Arrow data
    /// type of the source column; `null_count` is a `UInt64` column.
    ///
    /// Note: struct columns are not yet supported upstream
    /// (see apache/arrow-rs#7364).
    #[pyo3(signature = (column_name, *, missing_null_counts_as_zero = true))]
    fn statistics(
        &self,
        column_name: &str,
        missing_null_counts_as_zero: bool,
    ) -> Arro3IoResult<Arro3RecordBatch> {
        let parquet_meta = self.meta.metadata();
        let converter = StatisticsConverter::try_new(
            column_name,
            self.meta.schema(),
            self.meta.parquet_schema(),
        )?
        .with_missing_null_counts_as_zero(missing_null_counts_as_zero);

        let min_values = converter.row_group_mins(parquet_meta.row_groups())?;
        let max_values = converter.row_group_maxes(parquet_meta.row_groups())?;
        let null_counts = converter.row_group_null_counts(parquet_meta.row_groups())?;

        // When `missing_null_counts_as_zero` is true, `null_counts` is
        // guaranteed to contain no nulls, so the field is non-nullable.
        let schema = Arc::new(Schema::new(vec![
            Field::new("min", min_values.data_type().clone(), true),
            Field::new("max", max_values.data_type().clone(), true),
            Field::new(
                "null_count",
                null_counts.data_type().clone(),
                !missing_null_counts_as_zero,
            ),
        ]));
        let batch =
            RecordBatch::try_new(schema, vec![min_values, max_values, Arc::new(null_counts)])?;
        Ok(batch.into())
    }
}

pub(crate) struct PyWriterVersion(WriterVersion);

impl<'py> FromPyObject<'_, 'py> for PyWriterVersion {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        let s = obj.extract::<PyBackedStr>()?;
        let version =
            WriterVersion::from_str(&s).map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self(version))
    }
}

pub(crate) struct PyCompression(Compression);

impl<'py> FromPyObject<'_, 'py> for PyCompression {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        let s = obj.extract::<PyBackedStr>()?;
        let compression =
            Compression::from_str(&s).map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self(compression))
    }
}

#[derive(Debug)]
pub(crate) struct PyEncoding(Encoding);

impl<'py> FromPyObject<'_, 'py> for PyEncoding {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        let s = obj.extract::<PyBackedStr>()?;
        let encoding =
            Encoding::from_str(&s).map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self(encoding))
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub(crate) struct PyColumnPath(ColumnPath);

impl<'py> FromPyObject<'_, 'py> for PyColumnPath {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(path) = obj.extract::<String>() {
            Ok(Self(path.into()))
        } else if let Ok(path) = obj.extract::<Vec<String>>() {
            Ok(Self(path.into()))
        } else {
            Err(PyTypeError::new_err(
                "Expected string or list of string input for column path.",
            ))
        }
    }
}

#[pyfunction]
#[pyo3(signature=(
    data,
    file,
    *,
    bloom_filter_enabled = None,
    bloom_filter_fpp = None,
    bloom_filter_ndv = None,
    column_compression = None,
    column_dictionary_enabled = None,
    column_encoding = None,
    column_max_statistics_size = None,
    compression = None,
    created_by = None,
    data_page_row_count_limit = None,
    data_page_size_limit = None,
    dictionary_enabled = None,
    dictionary_page_size_limit = None,
    encoding = None,
    key_value_metadata = None,
    max_row_group_size = None,
    max_statistics_size = None,
    skip_arrow_metadata = false,
    write_batch_size = None,
    writer_version = None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_parquet(
    data: AnyRecordBatch,
    file: FileWriter,
    bloom_filter_enabled: Option<bool>,
    bloom_filter_fpp: Option<f64>,
    bloom_filter_ndv: Option<u64>,
    column_compression: Option<HashMap<PyColumnPath, PyCompression>>,
    column_dictionary_enabled: Option<HashMap<PyColumnPath, bool>>,
    column_encoding: Option<HashMap<PyColumnPath, PyEncoding>>,
    // TODO: remove in next breaking release
    #[allow(unused_variables)] column_max_statistics_size: Option<HashMap<PyColumnPath, usize>>,
    compression: Option<PyCompression>,
    created_by: Option<String>,
    data_page_row_count_limit: Option<usize>,
    data_page_size_limit: Option<usize>,
    dictionary_enabled: Option<bool>,
    dictionary_page_size_limit: Option<usize>,
    encoding: Option<PyEncoding>,
    key_value_metadata: Option<HashMap<String, String>>,
    max_row_group_size: Option<usize>,
    // TODO: remove in next breaking release
    #[allow(unused_variables)] max_statistics_size: Option<usize>,
    skip_arrow_metadata: bool,
    write_batch_size: Option<usize>,
    writer_version: Option<PyWriterVersion>,
) -> PyArrowResult<()> {
    let mut props = WriterProperties::builder();

    if let Some(writer_version) = writer_version {
        props = props.set_writer_version(writer_version.0);
    }
    if let Some(data_page_size_limit) = data_page_size_limit {
        props = props.set_data_page_size_limit(data_page_size_limit);
    }
    if let Some(data_page_row_count_limit) = data_page_row_count_limit {
        props = props.set_data_page_row_count_limit(data_page_row_count_limit);
    }
    if let Some(dictionary_page_size_limit) = dictionary_page_size_limit {
        props = props.set_dictionary_page_size_limit(dictionary_page_size_limit);
    }
    if let Some(write_batch_size) = write_batch_size {
        props = props.set_write_batch_size(write_batch_size);
    }
    if let Some(max_row_group_size) = max_row_group_size {
        props = props.set_max_row_group_row_count(Some(max_row_group_size));
    }
    if let Some(created_by) = created_by {
        props = props.set_created_by(created_by);
    }
    if let Some(key_value_metadata) = key_value_metadata {
        props = props.set_key_value_metadata(Some(
            key_value_metadata
                .into_iter()
                .map(|(k, v)| KeyValue::new(k, v))
                .collect(),
        ));
    }
    if let Some(compression) = compression {
        props = props.set_compression(compression.0);
    }
    if let Some(dictionary_enabled) = dictionary_enabled {
        props = props.set_dictionary_enabled(dictionary_enabled);
    }
    if let Some(bloom_filter_enabled) = bloom_filter_enabled {
        props = props.set_bloom_filter_enabled(bloom_filter_enabled);
    }
    if let Some(bloom_filter_fpp) = bloom_filter_fpp {
        props = props.set_bloom_filter_fpp(bloom_filter_fpp);
    }
    if let Some(bloom_filter_ndv) = bloom_filter_ndv {
        props = props.set_bloom_filter_ndv(bloom_filter_ndv);
    }
    if let Some(encoding) = encoding {
        props = props.set_encoding(encoding.0);
    }
    if let Some(column_encoding) = column_encoding {
        for (column_path, encoding) in column_encoding.into_iter() {
            props = props.set_column_encoding(column_path.0, encoding.0);
        }
    }
    if let Some(column_compression) = column_compression {
        for (column_path, compression) in column_compression.into_iter() {
            props = props.set_column_compression(column_path.0, compression.0);
        }
    }
    if let Some(column_dictionary_enabled) = column_dictionary_enabled {
        for (column_path, dictionary_enabled) in column_dictionary_enabled.into_iter() {
            props = props.set_column_dictionary_enabled(column_path.0, dictionary_enabled);
        }
    }

    let reader = data.into_reader()?;

    let writer_options = ArrowWriterOptions::new()
        .with_properties(props.build())
        .with_skip_arrow_metadata(skip_arrow_metadata);
    let mut writer =
        ArrowWriter::try_new_with_options(file, reader.schema(), writer_options).unwrap();
    for batch in reader {
        writer.write(&batch?).unwrap();
    }
    writer.close().unwrap();
    Ok(())
}
