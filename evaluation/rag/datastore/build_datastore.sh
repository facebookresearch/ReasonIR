conda activate scaling
cd retrieval-scaling


datastore_domain=mmlu_reasonir
checkpoint=reasonir/ReasonIR-8B

PYTHONPATH=.  python ric/main_ric.py \
    --config-name default \
    tasks.datastore.embedding=True \
    tasks.datastore.index=True \
    model.datastore_encoder=$checkpoint \
    model.query_encoder=$checkpoint \
    datastore.domain=$datastore_domain \
    datastore.embedding.passage_maxlength=2048 \
    datastore.embedding.per_gpu_batch_size=8 \
    datastore.index.projection_size=4096 \
    datastore.embedding.num_shards=1 \
    datastore.embedding.shard_ids=[0] \
    datastore.index.index_shard_ids=[0]