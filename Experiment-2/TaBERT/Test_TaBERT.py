import torch.nn.functional as F
from table_bert import TableBertModel
from table_bert import Table, Column

model = TableBertModel.from_pretrained(
    'pretrained/tabert_large_k3/model.bin',
)

table = Table(
    id='List of countries by GDP (PPP)',
    header=[
        Column('Nation', 'text', sample_value='United States'),
        Column('Gross Domestic Product', 'real', sample_value='21,439,453')
    ],
    data=[
        ['United States', '21,439,453'],
        ['China', '27,308,857'],
        ['European Union', '22,774,165'],
    ]
).tokenize(model.tokenizer)

context = 'show me countries ranked by GDP'

# model takes batched, tokenized inputs
context_encoding, column_encoding, info_dict = model.encode(
    contexts=[model.tokenizer.tokenize(context)],
    tables=[table]
)

# Pool: context (mean over tokens), columns (per-column already)
ctx_vec = context_encoding.mean(dim=1)  # (1, hidden_size)
col_vecs = column_encoding  # (1, num_cols, hidden_size)

# Cosine similarity between context and each column
sims = F.cosine_similarity(
    ctx_vec.unsqueeze(1), col_vecs, dim=-1
).squeeze(0)

print("Context:", repr(context))
print("Table:", table.id)
print("\nSimilarity (context vs each column):")
for col, sim in zip(table.header, sims.tolist()):
    print(f"  {col.name}: {sim:.4f}")
print(f"\nOverall (context vs mean of columns): {F.cosine_similarity(ctx_vec, col_vecs.mean(dim=1), dim=-1).item():.4f}")

