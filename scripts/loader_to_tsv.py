from torch.utils.data import DataLoader


def dump_to_tsv(loader: DataLoader, out_file: str, ids_token_map: dict, first_n=1000):
    out = None
    with open(out_file, "w") as f:
        i = 0
        for entry in loader:
            for entry_unbatched in entry[0]:
                for entry_singletext in entry_unbatched:
                    for token in entry_singletext:
                        new_out = ids_token_map[token.item()]+' '
                        if new_out != out:
                            f.write(new_out)
                            out = new_out
            # i += 1
            print(file=f)
            if i >= first_n:
                return True
