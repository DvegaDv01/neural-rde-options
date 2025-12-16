# s3_data_source.py - AWS S3 Integration

## Overview

This module provides read-only access to CBOE options data stored in AWS S3 by the data-engine Java application. It handles file discovery, downloading with local caching, coverage analysis, and direct integration with the preprocessing pipeline.

**Location**: `s3_data_source.py`
**Lines**: 998
**Dependencies**: Pandas, boto3 (optional)

---

## S3 Bucket Structure

The module is configured to match the data-engine Spring Boot application's output:

```
s3://maitlandoak/SPY/
├── interval_5min/
│   ├── 2024/
│   │   ├── UnderlyingOptionsIntervals_300sec_calcs_oi_2024-06-15_1000.csv
│   │   └── ...
│   └── 2025/
├── interval_15min/
├── interval_30min/
└── interval_60min/
```

---

## Module Structure

### Configuration (Lines 56-131)

#### Default Constants (Lines 66-69)

```python
DEFAULT_BUCKET = "maitlandoak"
DEFAULT_REGION = "us-east-1"
DEFAULT_PREFIX = "SPY"
```

These match the data-engine k8s/base/data-engine/configmap.yaml configuration.

#### `S3Config` Dataclass (Line 77)

Configuration container for S3 data source.

**Fields**:
| Field | Default | Description |
|-------|---------|-------------|
| `bucket` | "maitlandoak" | S3 bucket name |
| `region` | "us-east-1" | AWS region |
| `prefix` | "SPY" | Base prefix in bucket |
| `cache_dir` | "~/.nrde_cache" | Local cache directory |
| `cache_enabled` | True | Enable local caching |
| `max_concurrent_downloads` | 5 | Parallel download limit |
| `connect_timeout` | 10 | Connection timeout (seconds) |
| `read_timeout` | 60 | Read timeout (seconds) |
| `aws_access_key_id` | None | Optional explicit credentials |
| `aws_secret_access_key` | None | Optional explicit credentials |

#### `S3FileInfo` Dataclass (Line 112)

Information about a file in S3.

**Fields**: `key`, `size`, `last_modified`, `etag`

**Properties**:
- `filename`: Base filename from key
- `date`: Extracted date from filename (e.g., `2024-06-15`)

#### `CoverageReport` Dataclass (Line 133)

Coverage report for an interval and year.

**Fields**:
```python
interval: str
year: int
total_files: int
date_range: Tuple[date, date]
available_dates: Set[date]
missing_dates: Set[date]
total_size_mb: float
```

**Property**: `coverage_pct` returns percentage of expected dates with data.

---

### S3 Client Wrapper (Lines 152-253)

#### `S3Client` Class (Line 156)

Lightweight boto3 wrapper with lazy initialization.

##### `client` Property (Line 168)

Lazy initialization of boto3 S3 client with:
- Configurable timeouts
- Adaptive retry mode (3 attempts)
- Optional explicit credentials

##### `list_objects(prefix)` (Line 199)

Lists objects with automatic pagination.

Yields `S3FileInfo` for each object.

##### `download_file(key, local_path)` (Line 217)

Downloads file from S3, creating directories as needed.

##### `get_object_metadata(key)` (Line 237)

Gets metadata without downloading (uses HEAD request).

##### `object_exists(key)` (Line 250)

Checks if object exists.

---

### Cache Manager (Lines 255-353)

#### `CacheManager` Class (Line 259)

Manages local file cache for downloaded S3 objects.

**Cache structure**:
```
cache_dir/
├── SPY/
│   ├── interval_5min/
│   │   ├── 2024/
│   │   │   ├── file1.csv
│   │   │   └── file1.csv.etag
│   │   └── 2025/
│   └── interval_15min/
└── ...
```

##### `get_cache_path(s3_key)` (Line 282)

Returns local path for an S3 key.

##### `is_cached(s3_key, etag)` (Line 286)

Checks if file is cached, optionally validating ETag for freshness.

##### `store_etag(s3_key, etag)` (Line 315)

Stores ETag alongside cached file for validation.

##### `clear_cache(prefix)` (Line 322)

Clears cached files, optionally under specific prefix.

##### `get_cache_stats()` (Line 337)

Returns cache statistics: file count, total size, enabled status.

---

### S3 Data Source (Lines 355-897)

#### `S3DataSource` Class (Line 359)

Main class for accessing CBOE options data from S3.

**Interval aliases** (Line 379):
```python
INTERVAL_ALIASES = {
    '5min': 'interval_5min',
    '15min': 'interval_15min',
    '30min': 'interval_30min',
    '60min': 'interval_60min',
    '1hour': 'interval_60min',
}
```

---

#### Discovery Methods (Lines 442-553)

##### `list_intervals()` (Line 446)

Lists available data intervals.

**Returns**: `['5min', '15min', '30min', '60min']`

##### `list_years(interval)` (Line 471)

Lists available years for an interval.

**Returns**: `[2024, 2025]`

##### `list_files(interval, year, month)` (Line 497)

Lists available files with optional year/month filters.

**Returns**: List of `S3FileInfo` objects sorted by filename.

##### `get_available_dates(interval, year)` (Line 530)

Returns set of dates with available data.

---

#### Coverage Analysis (Lines 554-654)

##### `get_coverage(interval, year, exclude_weekends)` (Line 558)

Generates comprehensive coverage report.

**Logic**:
1. List all files for interval/year
2. Extract available dates
3. Generate expected dates (optionally excluding weekends)
4. Compute missing dates
5. Calculate total size

**Returns**: `CoverageReport`

##### `find_gaps(interval, start_date, end_date, exclude_weekends)` (Line 617)

Finds missing dates in a date range.

**Returns**: List of missing dates.

---

#### Data Loading (Lines 655-841)

##### `_download_file(file_info)` (Line 659)

Downloads file if not cached, returns local path.

**Logic**:
1. Check cache (with ETag validation)
2. If cached and valid, return path
3. Otherwise download and store ETag

##### `_extract_zip(zip_path)` (Line 675)

Extracts ZIP file and returns path to CSV.

##### `load_file(interval, target_date)` (Line 694)

Loads data for a specific date.

**Steps**:
1. Find matching file
2. Download (with caching)
3. Extract if ZIP
4. Load CSV

**Raises**: `FileNotFoundError` if no data available.

##### `load_date_range(interval, start_date, end_date, show_progress)` (Line 737)

Loads data for a date range.

**Features**:
- Automatic year spanning
- Progress bar (if tqdm available)
- Graceful handling of failed files
- Combines all data into single DataFrame

##### `load_latest(interval, n_days)` (Line 812)

Loads most recent N trading days of data.

**Default**: 5 days

##### `load_for_preprocessing(interval, n_days)` (Line 847)

**Main entry point** for integration with preprocessing pipeline.

**Additional processing**:
- Parses `quote_datetime` as datetime
- Parses `expiration` as datetime

---

#### Utility Methods (Lines 875-897)

##### `get_cache_stats()` (Line 879)

Returns cache statistics.

##### `clear_cache(interval)` (Line 883)

Clears cache, optionally for specific interval.

---

### Factory Functions (Lines 899-953)

#### `create_s3_source(bucket, region, cache_dir, **kwargs)` (Line 903)

Factory function with defaults matching data-engine.

#### `create_s3_source_from_env()` (Line 931)

Creates source from environment variables.

**Expected variables**:
| Variable | Required | Default |
|----------|----------|---------|
| `NRDE_S3_BUCKET` | Yes | - |
| `NRDE_S3_REGION` | No | us-east-1 |
| `NRDE_CACHE_DIR` | No | ~/.nrde_cache |
| `AWS_ACCESS_KEY_ID` | No | - |
| `AWS_SECRET_ACCESS_KEY` | No | - |

---

### CLI Demo (Lines 955-998)

Command-line interface for testing.

**Arguments**:
| Argument | Default | Description |
|----------|---------|-------------|
| `--bucket` | maitlandoak | S3 bucket name |
| `--region` | us-east-1 | AWS region |
| `--interval` | 5min | Data interval |
| `--days` | 5 | Days to load |
| `--list-intervals` | - | List available intervals |
| `--coverage` | - | Show coverage for year |

**Usage examples**:
```bash
# List available intervals
python s3_data_source.py --list-intervals

# Show coverage for 2024
python s3_data_source.py --coverage 2024

# Load latest 5 days
python s3_data_source.py --days 5
```

---

## Usage Example

```python
from s3_data_source import S3DataSource, create_s3_source

# Create source with defaults
source = S3DataSource()
# Or: source = create_s3_source()

# Check what's available
print(source.list_intervals())  # ['5min', '15min', ...]
print(source.list_years('5min'))  # [2024, 2025]

# Coverage analysis
report = source.get_coverage('5min', 2024)
print(f"Coverage: {report.coverage_pct:.1f}%")
print(f"Missing: {len(report.missing_dates)} days")

# Load data
df = source.load_latest('5min', n_days=5)
print(f"Loaded {len(df):,} rows")

# Load specific date
df = source.load_file('5min', '2024-06-15')

# Load date range
df = source.load_date_range('5min', '2024-06-01', '2024-06-30')

# For preprocessing pipeline
df = source.load_for_preprocessing('5min', n_days=5)
```

---

## Function Reference Table

| Function/Class | Line | Purpose |
|----------------|------|---------|
| `S3Config` | 77 | Configuration container |
| `S3FileInfo` | 112 | File metadata |
| `CoverageReport` | 133 | Coverage analysis result |
| `S3Client` | 156 | Boto3 wrapper |
| `CacheManager` | 259 | Local cache management |
| `S3DataSource` | 359 | **Main data source class** |
| `S3DataSource.list_intervals` | 446 | List available intervals |
| `S3DataSource.list_years` | 471 | List years for interval |
| `S3DataSource.list_files` | 497 | List files |
| `S3DataSource.get_coverage` | 558 | Coverage analysis |
| `S3DataSource.find_gaps` | 617 | Find missing dates |
| `S3DataSource.load_file` | 694 | Load single date |
| `S3DataSource.load_date_range` | 737 | Load date range |
| `S3DataSource.load_latest` | 812 | Load recent days |
| `S3DataSource.load_for_preprocessing` | 847 | **Pipeline integration** |
| `create_s3_source` | 903 | Factory function |
| `create_s3_source_from_env` | 931 | Create from environment |
