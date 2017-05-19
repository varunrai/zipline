from toolz import partition_all


def roll_dates_to_previous_session(calendar, *dates):
    """
    Roll ``dates`` to the next session of ``calendar``.

    Parameters
    ----------
    calendar : zipline.utils.calendars.trading_calendar.TradingCalendar
        The calendar to use as a reference.
    *dates : pd.Timestamp
        The dates for which the last trading date is needed.

    Returns
    -------
    rolled_dates: (np.datetime64, np.datetime64)
        The last trading date of the input dates, inclusive.

    """
    all_sessions = calendar.all_sessions

    locs = [all_sessions.get_loc(dt, method='ffill') for dt in dates]
    return all_sessions[locs].tolist()


def compute_date_range_chunks(calendar, start_date, end_date, chunksize):
    """Compute the start and end dates to run a pipeline for.

    Parameters
    ----------
    calendar : TradingCalendar
        The trading calendar to align the dates with.
    start_date : pd.Timestamp
        The first date in the pipeline.
    end_date : pd.Timestamp
        The last date in the pipeline.
    chunksize : int or None
        The size of the chunks to run.

    Returns
    -------
    ranges : iterable[(np.datetime64, np.datetime64)]
        A sequence of start and end dates to run the pipeline for.
    """
    if chunksize is None:
        dates = roll_dates_to_previous_session(calendar, start_date, end_date)

        return [(dates[0], dates[1])]

    all_sessions = calendar.all_sessions
    all_sessions.offset = None
    start_ix, end_ix = all_sessions.slice_locs(start_date, end_date)
    return (
        (r[0], r[-1]) for r in partition_all(
            chunksize, all_sessions[start_ix:end_ix]
        )
    )
