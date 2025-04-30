from apis.offline_cached_massiveds_searched_results import (
    search_offline_cached_massiveds,
    format_offline_document_string,
)

def search_api(search_engine, question, client=None, model_name=None, use_query_rewriting=False, cache=None):
    if search_engine == 'offline_massiveds':
        return search_offline_cached_massiveds(), None
    else:
        raise NotImplementedError


def format_document_string(search_engine, results, top_k, max_doc_len=None):
    if search_engine == 'offline_massiveds':
        return format_offline_document_string(results, top_k, max_doc_len)
    else:
        raise NotImplementedError