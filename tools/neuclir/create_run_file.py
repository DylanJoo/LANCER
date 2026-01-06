from argparse import Namespace
import os
import json


def create_run_file(args: Namespace, queries: dict, rac_data: dict,
                    file_name="neuclir-run-1.json") -> None:
    """ Save 'runs' .json file on disk containing a ranked list of
        document IDs
    """

    runs = {}
    for q in queries:
        runs[q] = rac_data[q]["docids"]

    with open(os.path.join(args.crux_dir, file_name), "w") as file:
        json.dump(runs, file)

    # save TREC formatted run file
    os.makedirs(f"{args.crux_dir}/runs", exist_ok=True)
    output_path = f"{args.crux_dir}/runs/neuclir-plaidx-1.qrel"
    with open(output_path, "w") as f:
        for qid in runs:
            for docid in runs[qid]:
                line = f"{qid} 0 {docid} 3\n"
                f.write(line)
