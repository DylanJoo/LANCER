export CRUX_ROOT=/home/dju/datasets/crux
export CRUX_ROOT=/home/hltcoe/jhueiju/datasets/crux

## 1. Neuclir'24 ReportGen
# for run_file in results/neuclir-runs/*.run;do
#     python -m crux.evaluation.rac_eval \
#         --run $run_file \
#         --qrel $CRUX_ROOT/crux-neuclir/qrels/neuclir24-test-request.qrel \
#         --judge $CRUX_ROOT/crux-neuclir/judge/ratings.human.jsonl 
# done

# 2. CRUX-MDS-DUC'04
for run_file in results/crux-mds-duc04-runs/*oracle*.run;do
    python -m crux.evaluation.rac_eval \
        --run $run_file \
        --qrel $CRUX_ROOT/crux-mds-duc04/qrels/div_qrels-tau3.txt \
        --judge $CRUX_ROOT/crux-mds-duc04/judge \
        --filter_by_oracle
done
