from glob import glob
import ir_measures
from ir_measures import nDCG, P


for file in glob("runs/*autorerank*"):
    qrels = ir_measures.read_trec_qrels("/exp/scale25/neuclir/eval/qrel/neuclir24-test-request.qrel")
    run = ir_measures.read_trec_run(file)
    result = ir_measures.calc_aggregate([nDCG@10, P@10], qrels, run)
    print(file, result[nDCG@10], result[P@10])
