def _winnow_process_list(process_list):
    to_pop = []
    for ii in range(len(process_list)-1, -1, -1):
        if process_list[ii].exitcode is not None:
            to_pop.append(ii)
    for ii in to_pop:
        process_list.pop(ii)
    return process_list
