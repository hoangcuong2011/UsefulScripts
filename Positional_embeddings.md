There are two possible ways of implemnting positional embeddings for Transformer from some sources on the Internet.

1. Implemented something like this (I could not remember the public source)

            def positional_embedding(max_length, model_size):
                PE = np.zeros((max_length, model_size))
                for i in range(model_size//2):
                    for pos in range(max_length):
                        PE[pos][2*i] = np.sin(pos / (10000.0 ** (2.0*i / model_size)))
                        PE[pos][2*i+1] = np.cos(pos / (10000.0 ** (2.0*i / model_size)))
                print(type(PE))
                return PE

2. Implemented something like this (This is from fairseq)

            def positional_embedding(max_length, model_size, padding_idx=1):
                half_dim = model_size // 2
                emb = math.log(10000) / (half_dim - 1)
                emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
                emb = torch.arange(max_length, dtype=torch.float).unsqueeze(
                    1
                ) * emb.unsqueeze(0)
                emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
                    max_length, -1
                )
                if model_size % 2 == 1:
                    # zero pad
                    emb = torch.cat([emb, torch.zeros(max_length, 1)], dim=1)
                if padding_idx is not None:
                    emb[padding_idx, :] = 0
                return emb

    def make_positions(padding_idx=1):
        import torch
        tensor = torch.tensor([[6513, 1276,    2]])
        mask = tensor.ne(padding_idx).int()
        print("mask", mask)
        return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx



They are different. I did not have a comparison between them though.
