from torch.utils.data import DataLoader
import torch

def dump_to_tsv(loader: DataLoader, out_file: str, ids_token_map: dict, first_n=1000):
    out = None
    with open(out_file, "w") as f:
        i = 0
        for entry in loader:
            for entry_singletext in entry[0]:
                for entry_batches in torch.stack(entry_singletext).transpose(0, 1):
                    for token in entry_batches:
                        new_out = ids_token_map[token.item()]+' '
                        # if new_out != out:
                        f.write(new_out)
                            # out = new_out
            # i += 1
            print(file=f)
            if i >= first_n:
                return True
