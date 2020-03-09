import numpy


def sync_time_series(time_series_a, time_kdtree_a, time_series_b,
                     time_kdtree_b):

    if len(time_series_a) < len(time_series_b):
        # Time series and KD tree
        refs_t_series = time_series_a
        sync_kdtree = time_kdtree_b

    else:
        # Time series and KD tree
        refs_t_series = time_series_b
        sync_kdtree = time_kdtree_a

    refs_t_series = numpy.array(refs_t_series, ndmin=2).T
    sync_index = sync_kdtree.query(refs_t_series, k=1)[1]
    assert sync_index.shape[0] == refs_t_series.shape[
        0], "Data are not of same length"

    return sync_index
