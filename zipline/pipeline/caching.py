import os

import bcolz
import numpy as np
import pandas as pd
from zipline.utils.sentinel import sentinel


class PipelineResult(object):
    """An object for caching and reading cached pipeline results.

    Parameters
    ----------
    ctable : bcolz.ctable
        The ctable that backs this pipeline result.

    See Also
    --------
    PipelineResult.from_dataframe
    PipelineResult.open
    """
    ALL_COLUMNS = sentinel(
        'ALL_COLUMNS',
        "Sentinel indicating that all columns should be read/written.",
    )

    # The result file format version. If the result of running a pipeline
    # change or we change the data format we need to increment this.
    version = 1

    _dates_column_name = 'dates'
    _sid_column_name = 'sid'
    _metadata_columns = frozenset({_dates_column_name, _sid_column_name})
    _metadata_attributes = frozenset({'start_date', 'end_date', 'version'})

    def __init__(self, ctable):
        if not self._metadata_columns <= set(ctable.names):
            raise ValueError(
                'missing expected metadata columns: %r' % (
                    self._metadata_columns - set(ctable.names)
                ),
            )
        if not self._metadata_attributes <= set(ctable.attrs.attrs):
            raise ValueError(
                'missing expected attributes: %r' % (
                    self._metadata_attributes - set(ctable.attrs.attrs)
                ),
            )
        if ctable.attrs['version'] != self.version:
            raise ValueError(
                'mismatched result version, found %r expected %r' % (
                    ctable.attrs['version'],
                    self.version,
                ),
            )

        self._ctable = ctable

    @classmethod
    def from_dataframe(cls, df):
        if (not isinstance(df.index, pd.MultiIndex) or
                len(df.index.levels) != 2):
            raise ValueError('expected a two level multi-indexed dataframe')

        df = df.reset_index(level=[0, 1])
        df.rename(
            columns={
                'level_0': cls._dates_column_name,
                'level_1': cls._sid_column_name,
            },
            inplace=True,
        )
        df[cls._sid_column_name] = df[cls._sid_column_name].astype(np.int64)
        ctable = bcolz.ctable.fromdataframe(df)
        dates = df[cls._dates_column_name]
        ctable.attrs['start_date'] = dates.iloc[0].value
        ctable.attrs['end_date'] = dates.iloc[-1].value
        ctable.attrs['version'] = cls.version
        return cls(ctable)

    def to_dataframe(self, columns):
        """
        Convert the PipelineResult into a DataFrame.

        Parameters
        ----------
        columns : list[str] or PipelineResult.ALL_COLUMNS
            A list of strings indicating which columns to read into memory.
            Passing ALL_COLUMNS indicates that all columns should be read.
        """
        ctable = self._ctable
        index_cols = [self._dates_column_name, self._sid_column_name]

        if columns is not self.ALL_COLUMNS:
            bad_columns = filter(index_cols.__contains__, columns)
            if bad_columns:
                raise ValueError(
                    "Invalid columns: {bad}. Use result.index.for metadata"
                    " columns.".format(bad=bad_columns)
                )

            ctable = ctable[columns + index_cols]

        return pd.DataFrame(ctable[:]).set_index(index_cols)

    def write(self, path, write_cols=ALL_COLUMNS):
        """Write the result to a given location.

        Parameters
        ----------
        path : str
            The file path to write this result to.
        write_cols : list
            The names of the columns in the written file. If None, all columns
            are used.

        """
        if write_cols is not self.ALL_COLUMNS:
            ctable = self._ctable[write_cols]
            for k, v in self._ctable.attrs:
                ctable.attrs[k] = v
        else:
            ctable = self._ctable

        if (ctable.rootdir is not None and
                os.path.abspath(ctable.rootdir) == os.path.abspath(path)):
            ctable.flush()
        else:
            copy = ctable.copy(rootdir=path, mode='w')
            for k, v in ctable.attrs:
                copy.attrs[k] = v
            copy.flush()

    @classmethod
    def open(cls, path):
        """Constructor from a filepath.

        Parameters
        ----------
        path : str
            The path to open.
        """
        return cls(bcolz.open(path))

    @property
    def start_date(self):
        return pd.Timestamp(self._ctable.attrs['start_date'], tz='utc')

    @property
    def end_date(self):
        return pd.Timestamp(self._ctable.attrs['end_date'], tz='utc')

    @property
    def term_names(self):
        return sorted(set(self._ctable.names) - self._metadata_columns)

    @property
    def path(self):
        return self._ctable.rootdir

    def dates_indexer(self, start_date, end_date):
        """Create an indexer into the results for the given date range.

        Parameters
        ----------
        start_date : pd.Timestamp
            The starting date of the slice.
        end_date : pd.Timestamp
            The ending date of the slice.

        Returns
        -------
        indexer : slice[int, int]
            The slice into the other columns to get the needed data.
        """
        if not (self.start_date <= start_date and
                self.end_date >= end_date):
            raise IndexError(
                'cannot build indexer for %s:%s' % (start_date, end_date),
            )

        dates = self.dates[:]
        start_idx = np.searchsorted(dates, start_date.to_datetime64())
        end_idx = np.searchsorted(dates, end_date.to_datetime64(), 'right')
        return np.s_[start_idx:end_idx]

    @property
    def dates(self):
        return self[self._dates_column_name]

    @property
    def sids(self):
        return self[self._sid_column_name]

    def __getitem__(self, key):
        return self._ctable[key]
