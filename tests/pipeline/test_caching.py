import os

import bcolz
import numpy as np
import pandas as pd

from zipline.pipeline.caching import PipelineResult
from zipline.testing import ZiplineTestCase
from zipline.testing.core import tmp_dir
from zipline.testing.predicates import assert_equal


class PipelineResultTestCase(ZiplineTestCase):

    @classmethod
    def init_class_fixtures(cls):
        super(PipelineResultTestCase, cls).init_class_fixtures()
        cls.dates = np.array(
            [
                '2017-01-03', '2017-01-03', '2017-01-04', '2017-01-04',
                '2017-01-05', '2017-01-05', '2017-01-06', '2017-01-06',
                '2017-01-09', '2017-01-09', '2017-01-10', '2017-01-10',
            ],
            dtype='datetime64'
        )
        cls.sid = np.tile([100, 8554], 6)
        cls.returns = np.random.normal(.0015, .015, 12)
        cls.close = np.random.normal(100, 10, 12)
        cls.open = np.random.normal(100, 10, 12)
        cls.version = 1
        cls.attrs = {
            'start_date': pd.Timestamp(cls.dates[0]).value,
            'end_date': pd.Timestamp(cls.dates[-1]).value,
            'version': cls.version
        }
        cls.columns = (cls.dates, cls.sid, cls.returns, cls.close, cls.open)
        cls.data_columns = ['returns', 'close', 'open']
        cls.names = ['dates', 'sid', 'returns', 'close', 'open']
        cls.data_root_dir = cls.enter_class_context(tmp_dir())
        cls.caching_dir = cls.data_root_dir.makedir('caching')

    def make_test_ctable(self,
                         columns=None,
                         names=None,
                         attrs=None):
        columns = self.columns if columns is None else columns
        names = self.names if names is None else names
        attrs = self.attrs if attrs is None else attrs

        ct = bcolz.ctable(columns=columns, names=names)
        for key, value in attrs.items():
            ct.attrs[key] = value
        return ct

    def test_pipeline_result_bad_init(self):
        """ The columns 'dates' and 'sid' are required to create a
        PipelineResult object.
        """

        table_missing_sid_column = self.make_test_ctable(
            columns=(self.dates, self.returns),
            names=['dates', 'returns']
        )
        with self.assertRaises(ValueError):
            PipelineResult(table_missing_sid_column)

        table_missing_dates_column = self.make_test_ctable(
            columns=(self.sid, self.returns),
            names=['sid', 'returns']
        )
        with self.assertRaises(ValueError):
            PipelineResult(table_missing_dates_column)

        table_missing_attrs_start_date = self.make_test_ctable(
            attrs={'end_date': self.dates[-1], 'version': self.version}
        )
        with self.assertRaises(ValueError):
            PipelineResult(table_missing_attrs_start_date)

        table_missing_attrs_end_date = self.make_test_ctable(
            attrs={'start_date': self.dates[0], 'version': self.version}
        )
        with self.assertRaises(ValueError):
            PipelineResult(table_missing_attrs_end_date)

        table_missing_attrs_version = self.make_test_ctable(
            attrs={'start_date': self.dates[0], 'end_date': self.dates[-1]}
        )
        with self.assertRaises(ValueError):
            PipelineResult(table_missing_attrs_version)

    def test_pipeline_result_properties(self):
        pr = PipelineResult(self.make_test_ctable())

        self.assertEqual(
            pr.start_date,
            pd.Timestamp(self.dates[0], tz='UTC')
        )
        self.assertEqual(
            pr.end_date,
            pd.Timestamp(self.dates[-1], tz='UTC')
        )
        assert_equal(set(pr.term_names), set(self.data_columns))
        self.assertIsNone(pr.path)
        assert_equal(pr.dates[:], self.dates)
        assert_equal(pr.sids[:], self.sid)

    def test_pipeline_result_from_dataframe(self):
        index = pd.MultiIndex.from_tuples(
            tuple(zip(self.dates, self.sid)),
            names=['dates', 'sid']
        )
        df = pd.DataFrame(
            {
                'returns': self.returns,
                'open': self.open,
                'close': self.close,
            },
            index=index
        )
        pr = PipelineResult.from_dataframe(df)

        self.assertEqual(
            pr.start_date,
            pd.Timestamp(self.dates[0], tz='UTC')
        )
        self.assertEqual(
            pr.end_date,
            pd.Timestamp(self.dates[-1], tz='UTC')
        )
        assert_equal(set(pr.term_names), set(self.data_columns))
        self.assertIsNone(pr.path)
        assert_equal(pr.dates[:], self.dates)
        assert_equal(pr.sids[:], self.sid)

    def test_pipeline_result_from_data_frame_value_error(self):
        df = pd.DataFrame({'returns': self.returns}, index=self.dates)
        with self.assertRaises(ValueError):
            PipelineResult.from_dataframe(df)

    def test_pipeline_result_to_dataframe(self):
        pr = PipelineResult(self.make_test_ctable())
        df = pr.to_dataframe(PipelineResult.ALL_COLUMNS)
        df_flat = df.reset_index(level=[0, 1])

        assert_equal(np.array(df_flat['dates']), self.dates)
        assert_equal(np.array(df_flat['sid']), self.sid)
        assert_equal(np.array(df_flat['returns']), self.returns)
        assert_equal(np.array(df_flat['close']), self.close)
        assert_equal(np.array(df_flat['open']), self.open)

    def test_pipeline_result_to_dateframe_specific_column(self):
        pr = PipelineResult(self.make_test_ctable())
        df = pr.to_dataframe(['returns'])
        assert_equal(df.columns.tolist(), ['returns'])

    def test_pipeline_result_to_dataframe_value_error(self):
        pr = PipelineResult(self.make_test_ctable())
        with self.assertRaises(ValueError):
            pr.to_dataframe(['sid'])
        with self.assertRaises(ValueError):
            pr.to_dataframe(['dates'])
        with self.assertRaises(ValueError):
            pr.to_dataframe(['some other col'])

    def test_pipeline_result_write(self):

        pr = PipelineResult(self.make_test_ctable())
        full_path = os.path.join(self.caching_dir, 'write_test_1')
        pr.write(full_path)

        expected = pd.DataFrame(
            {
                'dates': self.dates,
                'sid': self.sid,
                'returns': self.returns,
                'close': self.close,
                'open': self.open,
            }
        )
        result = bcolz.open(full_path).todataframe()

        self.assertEqual(
            set(expected.columns),
            set(result.columns)
        )
        self.assertTrue(result[self.names].equals(expected[self.names]))

    def test_pipeline_result_write_specific_column(self):
        column = 'returns'
        result_columns = ['dates', 'sid', 'returns']

        pr = PipelineResult(self.make_test_ctable())
        full_path = os.path.join(self.caching_dir, 'write_test_2')
        pr.write(full_path, [column])
        expected = pd.DataFrame(
            {
                'dates': self.dates,
                'sid': self.sid,
                'returns': self.returns,
            }
        )
        result = bcolz.open(full_path).todataframe()
        self.assertEqual(
            set(expected.columns),
            set(result.columns)
        )
        self.assertTrue(
            result[result_columns].equals(expected[result_columns])
        )

    def test_pipeline_result_write_specific_column_key_error(self):

        pr = PipelineResult(self.make_test_ctable())
        full_path = os.path.join(self.caching_dir, 'write_test_3')

        with self.assertRaises(ValueError):
            pr.write(full_path, ['some_other_column'])

    def test_pipeline_result_dates_indexer(self):
        pr = PipelineResult(self.make_test_ctable())
        indexer = pr.dates_indexer(
            pd.Timestamp('2017-01-04', tz='UTC'),
            pd.Timestamp('2017-01-09', tz='UTC')
        )
        result = pd.DataFrame(pr[indexer])[self.names]
        expected = pd.DataFrame(
            {
                'dates': self.dates[2:-2],
                'sid': self.sid[2:-2],
                'returns': self.returns[2:-2],
                'close': self.close[2:-2],
                'open': self.open[2:-2],
            }
        )[self.names]

        self.assertTrue(result.equals(expected))

    def test_pipeline_result_dates_indexer_bad_dates(self):
        pr = PipelineResult(self.make_test_ctable())
        # start date is before the PipelineResult start date
        with self.assertRaises(IndexError):
            pr.dates_indexer(
                pd.Timestamp('2016-12-31', tz='UTC'),
                pd.Timestamp('2017-01-09', tz='UTC')
            )
        # end date is after the PipelineResult end date
        with self.assertRaises(IndexError):
            pr.dates_indexer(
                pd.Timestamp('2017-01-04', tz='UTC'),
                pd.Timestamp('2017-05-23', tz='UTC')
            )

    def test_pipeline_result_open(self):
        start_pr = PipelineResult(self.make_test_ctable())
        full_path = os.path.join(self.caching_dir, 'open_test_1')
        start_pr.write(full_path)

        pr = PipelineResult.open(full_path)
        self.assertEqual(
            pr.start_date,
            pd.Timestamp(self.dates[0], tz='UTC')
        )
        self.assertEqual(
            pr.end_date,
            pd.Timestamp(self.dates[-1], tz='UTC')
        )
        assert_equal(set(pr.term_names), set(self.data_columns))
        self.assertEqual(full_path, pr.path)
        assert_equal(pr.dates[:], self.dates)
        assert_equal(pr.sids[:], self.sid)
