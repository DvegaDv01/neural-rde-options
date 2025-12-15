"""
S3 Data Source for CBOE options data.

This module provides read-only access to CBOE options data stored in S3
by the data-engine Java application. It handles:

1. Listing available intervals and files
2. Downloading files with local caching
3. Coverage analysis (gaps, available dates)
4. Direct integration with the data loading pipeline

Configuration defaults match the data-engine Spring Boot application:
    - Bucket: maitlandoak
    - Region: us-east-1
    - Prefix: SPY
    - Path pattern: SPY/interval_{interval}/{year}/{filename}

The S3 bucket structure (created by data-engine) is:
    s3://maitlandoak/SPY/
    ├── interval_5min/2024/
    ├── interval_5min/2025/
    ├── interval_15min/
    └── ...

Example:
    >>> from s3_data_source import S3DataSource
    >>>
    >>> # Uses defaults from data-engine config (bucket=maitlandoak)
    >>> source = S3DataSource()
    >>>
    >>> # List available data
    >>> intervals = source.list_intervals()
    >>> coverage = source.get_coverage("5min", 2024)
    >>>
    >>> # Load past 5 trading days
    >>> df = source.load_latest("5min", n_days=5)
    >>>
    >>> # Or load single file
    >>> df = source.load_file("5min", "2024-06-15")
"""

import hashlib
import logging
import os
import re
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# =============================================================================
# Default Configuration (from data-engine Spring Boot app)
# =============================================================================

# Default bucket from k8s/base/data-engine/configmap.yaml
DEFAULT_BUCKET = "maitlandoak"
DEFAULT_REGION = "us-east-1"
DEFAULT_PREFIX = "SPY"

# Path pattern from application.properties: SPY/interval_{interval}/{year}/{filename}
# Interval mappings from application.properties:
#   300sec|5min -> interval_5min
#   30sec -> interval_30sec
#   eod|end_of_day -> interval_eod


@dataclass
class S3Config:
    """
    Configuration for S3 data source.

    Defaults are configured to match the data-engine Spring Boot application
    that uploads CBOE data to S3.

    Attributes:
        bucket: S3 bucket name (default: maitlandoak)
        region: AWS region (default: us-east-1)
        prefix: Base prefix in bucket (default: SPY)
        cache_dir: Local cache directory
        cache_enabled: Whether to cache downloaded files locally
        max_concurrent_downloads: Maximum parallel downloads
        connect_timeout: Connection timeout in seconds
        read_timeout: Read timeout in seconds
    """
    bucket: str = DEFAULT_BUCKET
    region: str = DEFAULT_REGION
    prefix: str = DEFAULT_PREFIX
    cache_dir: str = "~/.nrde_cache"
    cache_enabled: bool = True
    max_concurrent_downloads: int = 5
    connect_timeout: int = 10
    read_timeout: int = 60

    # AWS credentials (optional - uses default chain if not provided)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    def __post_init__(self):
        self.cache_dir = os.path.expanduser(self.cache_dir)


@dataclass
class S3FileInfo:
    """Information about a file in S3."""
    key: str
    size: int
    last_modified: datetime
    etag: str

    @property
    def filename(self) -> str:
        return os.path.basename(self.key)

    @property
    def date(self) -> Optional[date]:
        """Extract date from filename."""
        match = re.search(r'(\d{4}-\d{2}-\d{2})', self.filename)
        if match:
            return datetime.strptime(match.group(1), '%Y-%m-%d').date()
        return None


@dataclass
class CoverageReport:
    """Coverage report for an interval."""
    interval: str
    year: int
    total_files: int
    date_range: Tuple[date, date]
    available_dates: Set[date]
    missing_dates: Set[date]
    total_size_mb: float

    @property
    def coverage_pct(self) -> float:
        total_expected = len(self.available_dates) + len(self.missing_dates)
        if total_expected == 0:
            return 0.0
        return len(self.available_dates) / total_expected * 100


# =============================================================================
# S3 Client Wrapper
# =============================================================================

class S3Client:
    """
    Lightweight S3 client wrapper using boto3.

    Handles connection management and provides simplified interface
    for the operations needed by the data source.
    """

    def __init__(self, config: S3Config):
        self.config = config
        self._client = None

    @property
    def client(self):
        """Lazy initialization of boto3 client."""
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config as BotoConfig
            except ImportError:
                raise ImportError(
                    "boto3 is required for S3 data source. "
                    "Install with: pip install boto3"
                )

            boto_config = BotoConfig(
                connect_timeout=self.config.connect_timeout,
                read_timeout=self.config.read_timeout,
                retries={'max_attempts': 3, 'mode': 'adaptive'}
            )

            session_kwargs = {'region_name': self.config.region}
            if self.config.aws_access_key_id and self.config.aws_secret_access_key:
                session_kwargs['aws_access_key_id'] = self.config.aws_access_key_id
                session_kwargs['aws_secret_access_key'] = self.config.aws_secret_access_key

            session = boto3.Session(**session_kwargs)
            self._client = session.client('s3', config=boto_config)

            logger.info(f"Connected to S3 bucket: {self.config.bucket}")

        return self._client

    def list_objects(self, prefix: str) -> Iterator[S3FileInfo]:
        """
        List objects under a prefix with automatic pagination.

        Yields:
            S3FileInfo for each object
        """
        paginator = self.client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                yield S3FileInfo(
                    key=obj['Key'],
                    size=obj['Size'],
                    last_modified=obj['LastModified'],
                    etag=obj['ETag'].strip('"')
                )

    def download_file(self, key: str, local_path: str) -> bool:
        """
        Download a file from S3.

        Args:
            key: S3 object key
            local_path: Local destination path

        Returns:
            True if successful
        """
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.client.download_file(self.config.bucket, key, local_path)
            logger.debug(f"Downloaded: {key} -> {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {key}: {e}")
            return False

    def get_object_metadata(self, key: str) -> Optional[Dict]:
        """Get object metadata without downloading."""
        try:
            response = self.client.head_object(Bucket=self.config.bucket, Key=key)
            return {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'etag': response['ETag'].strip('"'),
                'metadata': response.get('Metadata', {})
            }
        except Exception:
            return None

    def object_exists(self, key: str) -> bool:
        """Check if object exists."""
        return self.get_object_metadata(key) is not None


# =============================================================================
# Local Cache Manager
# =============================================================================

class CacheManager:
    """
    Manages local file cache for downloaded S3 objects.

    Cache structure:
        cache_dir/
        ├── SPY/
        │   ├── interval_5min/
        │   │   ├── 2024/
        │   │   │   ├── file1.csv
        │   │   │   └── file2.csv
        │   │   └── 2025/
        │   └── interval_15min/
        └── .cache_index.json
    """

    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, s3_key: str) -> Path:
        """Get local cache path for an S3 key."""
        return self.cache_dir / s3_key

    def is_cached(self, s3_key: str, etag: Optional[str] = None) -> bool:
        """
        Check if file is cached and optionally validate checksum.

        Args:
            s3_key: S3 object key
            etag: Optional ETag to verify cache validity
        """
        if not self.enabled:
            return False

        cache_path = self.get_cache_path(s3_key)
        if not cache_path.exists():
            return False

        if etag:
            # Verify checksum matches
            cached_etag = self._get_cached_etag(cache_path)
            return cached_etag == etag

        return True

    def _get_cached_etag(self, cache_path: Path) -> Optional[str]:
        """Get stored ETag for cached file."""
        etag_path = cache_path.with_suffix(cache_path.suffix + '.etag')
        if etag_path.exists():
            return etag_path.read_text().strip()
        return None

    def store_etag(self, s3_key: str, etag: str):
        """Store ETag for a cached file."""
        cache_path = self.get_cache_path(s3_key)
        etag_path = cache_path.with_suffix(cache_path.suffix + '.etag')
        etag_path.parent.mkdir(parents=True, exist_ok=True)
        etag_path.write_text(etag)

    def clear_cache(self, prefix: Optional[str] = None):
        """Clear cached files, optionally under a specific prefix."""
        import shutil

        if prefix:
            target = self.cache_dir / prefix
            if target.exists():
                shutil.rmtree(target)
                logger.info(f"Cleared cache: {target}")
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared entire cache")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        total_size = 0
        file_count = 0

        for path in self.cache_dir.rglob('*'):
            if path.is_file() and not path.suffix == '.etag':
                total_size += path.stat().st_size
                file_count += 1

        return {
            'cache_dir': str(self.cache_dir),
            'file_count': file_count,
            'total_size_mb': total_size / (1024 * 1024),
            'enabled': self.enabled
        }


# =============================================================================
# S3 Data Source
# =============================================================================

class S3DataSource:
    """
    S3 data source for CBOE options data.

    Provides high-level interface for accessing options data stored in S3
    by the data-engine application. Handles caching, coverage analysis,
    and integration with the data loading pipeline.

    Example:
        >>> source = S3DataSource(bucket="my-bucket")
        >>>
        >>> # Check what's available
        >>> print(source.list_intervals())
        >>> print(source.get_coverage("5min", 2024))
        >>>
        >>> # Load past 5 trading days
        >>> df = source.load_latest("5min", n_days=5)
    """

    # Interval name mappings
    INTERVAL_ALIASES = {
        '5min': 'interval_5min',
        '15min': 'interval_15min',
        '30min': 'interval_30min',
        '60min': 'interval_60min',
        '1hour': 'interval_60min',
    }

    # Interval to seconds mapping (for filename parsing)
    INTERVAL_SECONDS = {
        '5min': 300,
        '15min': 900,
        '30min': 1800,
        '60min': 3600,
    }

    def __init__(
        self,
        bucket: str = DEFAULT_BUCKET,
        region: str = DEFAULT_REGION,
        prefix: str = DEFAULT_PREFIX,
        cache_dir: str = "~/.nrde_cache",
        cache_enabled: bool = True,
        **kwargs
    ):
        """
        Initialize S3 data source.

        Defaults match the data-engine Spring Boot application configuration.

        Args:
            bucket: S3 bucket name (default: maitlandoak)
            region: AWS region (default: us-east-1)
            prefix: Base prefix (default: SPY)
            cache_dir: Local cache directory
            cache_enabled: Whether to cache downloads locally
            **kwargs: Additional S3Config parameters
        """
        self.config = S3Config(
            bucket=bucket,
            region=region,
            prefix=prefix,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
            **kwargs
        )

        self.s3 = S3Client(self.config)
        self.cache = CacheManager(self.config.cache_dir, self.config.cache_enabled)

    def _normalize_interval(self, interval: str) -> str:
        """Normalize interval name to S3 folder format."""
        interval = interval.lower().replace('_', '').replace('-', '')
        return self.INTERVAL_ALIASES.get(interval, f"interval_{interval}")

    def _build_prefix(self, interval: str, year: Optional[int] = None) -> str:
        """Build S3 prefix for listing."""
        interval_folder = self._normalize_interval(interval)
        prefix = f"{self.config.prefix}/{interval_folder}"
        if year:
            prefix = f"{prefix}/{year}"
        return prefix

    # =========================================================================
    # Discovery Methods
    # =========================================================================

    def list_intervals(self) -> List[str]:
        """
        List available data intervals.

        Returns:
            List of interval names (e.g., ['5min', '15min', '30min', '60min'])
        """
        prefix = f"{self.config.prefix}/"
        intervals = set()

        # Use delimiter to list "folders"
        response = self.s3.client.list_objects_v2(
            Bucket=self.config.bucket,
            Prefix=prefix,
            Delimiter='/'
        )

        for prefix_info in response.get('CommonPrefixes', []):
            folder = prefix_info['Prefix'].rstrip('/').split('/')[-1]
            if folder.startswith('interval_'):
                interval = folder.replace('interval_', '')
                intervals.add(interval)

        return sorted(intervals)

    def list_years(self, interval: str) -> List[int]:
        """
        List available years for an interval.

        Args:
            interval: Interval name (e.g., '5min')

        Returns:
            List of years with data
        """
        prefix = self._build_prefix(interval) + "/"
        years = set()

        response = self.s3.client.list_objects_v2(
            Bucket=self.config.bucket,
            Prefix=prefix,
            Delimiter='/'
        )

        for prefix_info in response.get('CommonPrefixes', []):
            folder = prefix_info['Prefix'].rstrip('/').split('/')[-1]
            if folder.isdigit():
                years.add(int(folder))

        return sorted(years)

    def list_files(
        self,
        interval: str,
        year: Optional[int] = None,
        month: Optional[int] = None
    ) -> List[S3FileInfo]:
        """
        List available files for an interval.

        Args:
            interval: Interval name
            year: Optional year filter
            month: Optional month filter (requires year)

        Returns:
            List of S3FileInfo objects
        """
        prefix = self._build_prefix(interval, year)
        if year and month:
            # Filter by month in filename
            month_str = f"{year}-{month:02d}"
            files = [
                f for f in self.s3.list_objects(prefix)
                if month_str in f.filename and f.filename.endswith(('.csv', '.zip'))
            ]
        else:
            files = [
                f for f in self.s3.list_objects(prefix)
                if f.filename.endswith(('.csv', '.zip'))
            ]

        return sorted(files, key=lambda f: f.filename)

    def get_available_dates(
        self,
        interval: str,
        year: Optional[int] = None
    ) -> Set[date]:
        """
        Get set of dates with available data.

        Args:
            interval: Interval name
            year: Optional year filter

        Returns:
            Set of dates with data files
        """
        files = self.list_files(interval, year)
        dates = set()

        for f in files:
            if f.date:
                dates.add(f.date)

        return dates

    # =========================================================================
    # Coverage Analysis
    # =========================================================================

    def get_coverage(
        self,
        interval: str,
        year: int,
        exclude_weekends: bool = True
    ) -> CoverageReport:
        """
        Get coverage report for an interval and year.

        Args:
            interval: Interval name
            year: Year to analyze
            exclude_weekends: Whether to exclude weekends from expected dates

        Returns:
            CoverageReport with available and missing dates
        """
        files = self.list_files(interval, year)
        available_dates = {f.date for f in files if f.date}

        if not available_dates:
            return CoverageReport(
                interval=interval,
                year=year,
                total_files=0,
                date_range=(date(year, 1, 1), date(year, 12, 31)),
                available_dates=set(),
                missing_dates=set(),
                total_size_mb=0.0
            )

        # Determine date range
        min_date = min(available_dates)
        max_date = min(max(available_dates), date.today() - timedelta(days=1))

        # Generate expected dates
        expected_dates = set()
        current = min_date
        while current <= max_date:
            if exclude_weekends:
                if current.weekday() < 5:  # Monday = 0, Friday = 4
                    expected_dates.add(current)
            else:
                expected_dates.add(current)
            current += timedelta(days=1)

        missing_dates = expected_dates - available_dates
        total_size = sum(f.size for f in files) / (1024 * 1024)

        return CoverageReport(
            interval=interval,
            year=year,
            total_files=len(files),
            date_range=(min_date, max_date),
            available_dates=available_dates,
            missing_dates=missing_dates,
            total_size_mb=total_size
        )

    def find_gaps(
        self,
        interval: str,
        start_date: Union[str, date],
        end_date: Union[str, date],
        exclude_weekends: bool = True
    ) -> List[date]:
        """
        Find missing dates in a date range.

        Args:
            interval: Interval name
            start_date: Start of range
            end_date: End of range
            exclude_weekends: Whether to exclude weekends

        Returns:
            List of missing dates
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        available = self.get_available_dates(interval)

        gaps = []
        current = start_date
        while current <= end_date:
            if exclude_weekends and current.weekday() >= 5:
                current += timedelta(days=1)
                continue
            if current not in available:
                gaps.append(current)
            current += timedelta(days=1)

        return gaps

    # =========================================================================
    # Data Loading
    # =========================================================================

    def _download_file(self, file_info: S3FileInfo) -> Optional[Path]:
        """Download file if not cached, return local path."""
        local_path = self.cache.get_cache_path(file_info.key)

        # Check cache
        if self.cache.is_cached(file_info.key, file_info.etag):
            logger.debug(f"Cache hit: {file_info.filename}")
            return local_path

        # Download
        if self.s3.download_file(file_info.key, str(local_path)):
            self.cache.store_etag(file_info.key, file_info.etag)
            return local_path

        return None

    def _extract_zip(self, zip_path: Path) -> Path:
        """Extract ZIP file and return path to CSV."""
        extract_dir = zip_path.parent

        with zipfile.ZipFile(zip_path, 'r') as zf:
            csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
            if not csv_files:
                raise ValueError(f"No CSV files in {zip_path}")

            csv_name = csv_files[0]
            csv_path = extract_dir / csv_name

            # Only extract if not already extracted
            if not csv_path.exists():
                zf.extract(csv_name, extract_dir)
                logger.debug(f"Extracted: {csv_name}")

        return csv_path

    def load_file(
        self,
        interval: str,
        target_date: Union[str, date]
    ) -> pd.DataFrame:
        """
        Load data for a specific date.

        Args:
            interval: Interval name
            target_date: Date to load

        Returns:
            DataFrame with CBOE options data

        Raises:
            FileNotFoundError: If no data available for date
        """
        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, '%Y-%m-%d').date()

        # Find matching file
        files = self.list_files(interval, target_date.year)
        matching = [f for f in files if f.date == target_date]

        if not matching:
            raise FileNotFoundError(
                f"No data available for {interval} on {target_date}"
            )

        file_info = matching[0]
        local_path = self._download_file(file_info)

        if local_path is None:
            raise IOError(f"Failed to download {file_info.key}")

        # Handle ZIP files
        if local_path.suffix == '.zip':
            local_path = self._extract_zip(local_path)

        # Load CSV
        return pd.read_csv(local_path)

    def load_date_range(
        self,
        interval: str,
        start_date: Union[str, date],
        end_date: Union[str, date],
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Load data for a date range.

        Args:
            interval: Interval name
            start_date: Start of range (inclusive)
            end_date: End of range (inclusive)
            show_progress: Whether to show progress bar

        Returns:
            Combined DataFrame with all data
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        # Get files in range
        all_files = []
        for year in range(start_date.year, end_date.year + 1):
            files = self.list_files(interval, year)
            for f in files:
                if f.date and start_date <= f.date <= end_date:
                    all_files.append(f)

        if not all_files:
            raise FileNotFoundError(
                f"No data available for {interval} between {start_date} and {end_date}"
            )

        logger.info(f"Loading {len(all_files)} files for {interval} "
                   f"from {start_date} to {end_date}")

        # Load all files
        dfs = []
        iterator = all_files

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(all_files, desc=f"Loading {interval}")
            except ImportError:
                pass

        for file_info in iterator:
            try:
                local_path = self._download_file(file_info)
                if local_path is None:
                    continue

                if local_path.suffix == '.zip':
                    local_path = self._extract_zip(local_path)

                df = pd.read_csv(local_path)
                dfs.append(df)

            except Exception as e:
                logger.warning(f"Failed to load {file_info.filename}: {e}")
                continue

        if not dfs:
            raise IOError("Failed to load any files")

        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined):,} rows from {len(dfs)} files")

        return combined

    def load_latest(self, interval: str, n_days: int = 5) -> pd.DataFrame:
        """
        Load the most recent N trading days of data.

        Args:
            interval: Interval name
            n_days: Number of most recent trading days to load (default: 5)

        Returns:
            DataFrame with recent data
        """
        available = sorted(self.get_available_dates(interval), reverse=True)

        if not available:
            raise FileNotFoundError(f"No data available for {interval}")

        dates_to_load = available[:n_days]

        logger.info(f"Loading {len(dates_to_load)} most recent days for {interval}: "
                   f"{dates_to_load[-1]} to {dates_to_load[0]}")

        if len(dates_to_load) == 1:
            return self.load_file(interval, dates_to_load[0])

        return self.load_date_range(
            interval,
            min(dates_to_load),
            max(dates_to_load),
            show_progress=True
        )

    # =========================================================================
    # Integration with cboe_preprocessing
    # =========================================================================

    def load_for_preprocessing(
        self,
        interval: str,
        n_days: int = 5
    ) -> pd.DataFrame:
        """
        Load data formatted for cboe_preprocessing.py.

        This is the main entry point for integrating with the Neural RDE
        preprocessing pipeline.

        Args:
            interval: Interval name (e.g., '5min')
            n_days: Number of recent trading days to load

        Returns:
            DataFrame ready for CBOEDataLoader
        """
        df = self.load_latest(interval, n_days)

        # Ensure datetime columns are properly parsed
        if 'quote_datetime' in df.columns:
            df['quote_datetime'] = pd.to_datetime(df['quote_datetime'])
        if 'expiration' in df.columns:
            df['expiration'] = pd.to_datetime(df['expiration'])

        return df

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self.cache.get_cache_stats()

    def clear_cache(self, interval: Optional[str] = None):
        """Clear local cache."""
        if interval:
            prefix = self._build_prefix(interval)
            self.cache.clear_cache(prefix)
        else:
            self.cache.clear_cache()

    def __repr__(self) -> str:
        return (
            f"S3DataSource(bucket='{self.config.bucket}', "
            f"prefix='{self.config.prefix}', "
            f"region='{self.config.region}')"
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_s3_source(
    bucket: str = DEFAULT_BUCKET,
    region: str = DEFAULT_REGION,
    cache_dir: str = "~/.nrde_cache",
    **kwargs
) -> S3DataSource:
    """
    Factory function to create an S3DataSource.

    Defaults match the data-engine Spring Boot application.

    Args:
        bucket: S3 bucket name (default: maitlandoak)
        region: AWS region (default: us-east-1)
        cache_dir: Local cache directory
        **kwargs: Additional configuration

    Returns:
        Configured S3DataSource
    """
    return S3DataSource(
        bucket=bucket,
        region=region,
        cache_dir=cache_dir,
        **kwargs
    )


def create_s3_source_from_env() -> S3DataSource:
    """
    Create S3DataSource from environment variables.

    Expected environment variables:
        - NRDE_S3_BUCKET: S3 bucket name (required)
        - NRDE_S3_REGION: AWS region (default: us-east-1)
        - NRDE_CACHE_DIR: Cache directory (default: ~/.nrde_cache)
        - AWS_ACCESS_KEY_ID: AWS credentials (optional)
        - AWS_SECRET_ACCESS_KEY: AWS credentials (optional)
    """
    bucket = os.environ.get('NRDE_S3_BUCKET')
    if not bucket:
        raise ValueError("NRDE_S3_BUCKET environment variable not set")

    return S3DataSource(
        bucket=bucket,
        region=os.environ.get('NRDE_S3_REGION', 'us-east-1'),
        cache_dir=os.environ.get('NRDE_CACHE_DIR', '~/.nrde_cache'),
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    )


# =============================================================================
# CLI Demo
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="S3 Data Source for CBOE options")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help=f"S3 bucket name (default: {DEFAULT_BUCKET})")
    parser.add_argument("--region", default=DEFAULT_REGION, help=f"AWS region (default: {DEFAULT_REGION})")
    parser.add_argument("--interval", default="5min", help="Data interval (default: 5min)")
    parser.add_argument("--days", type=int, default=5, help="Number of days to load (default: 5)")
    parser.add_argument("--list-intervals", action="store_true", help="List available intervals")
    parser.add_argument("--coverage", type=int, help="Show coverage for year")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    source = S3DataSource(bucket=args.bucket, region=args.region)

    if args.list_intervals:
        print("\nAvailable intervals:")
        for interval in source.list_intervals():
            years = source.list_years(interval)
            print(f"  {interval}: {years}")

    elif args.coverage:
        report = source.get_coverage(args.interval, args.coverage)
        print(f"\nCoverage Report: {args.interval} / {args.coverage}")
        print(f"  Files: {report.total_files}")
        print(f"  Date Range: {report.date_range[0]} to {report.date_range[1]}")
        print(f"  Coverage: {report.coverage_pct:.1f}%")
        print(f"  Missing: {len(report.missing_dates)} days")
        print(f"  Size: {report.total_size_mb:.1f} MB")

    else:
        print(f"\nLoading {args.days} most recent days of {args.interval} data...")
        df = source.load_latest(args.interval, args.days)
        print(f"\nLoaded {len(df):,} rows")
        print(f"Columns: {list(df.columns)}")
        print(f"\nSample:")
        print(df.head())
