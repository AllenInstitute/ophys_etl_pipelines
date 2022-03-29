from ophys_etl.modules.event_detection.\
    resources.event_decay_time_lookup import event_decay_lookup_dict


def test_lookup():
    """
    Just make sure that event_decay_lookup is a well-formed dict mapping
    strings to floats
    """
    for k in event_decay_lookup_dict:
        assert isinstance(k, str)
        assert isinstance(event_decay_lookup_dict[k], float)
