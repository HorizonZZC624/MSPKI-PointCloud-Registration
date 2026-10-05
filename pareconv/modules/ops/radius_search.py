





from pareconv.modules.ops.gpu_preprocess import stacked_exact_knn


def radius_search(
    q_points,
    s_points,
    q_lengths,
    s_lengths,
    num_neighbors,
    query_chunk_size=1024,
    support_chunk_size=4096,
):
    return stacked_exact_knn(
        q_points,
        s_points,
        q_lengths,
        s_lengths,
        num_neighbors=num_neighbors,
        query_chunk_size=query_chunk_size,
        support_chunk_size=support_chunk_size,
    )
