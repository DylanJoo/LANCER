# NeuCLIR
for retriever in bm25-neuclir lsr-neuclir qwen3-neuclir;do
    # LANCER vs others
    run1=runs/${retriever}+crux_reranking.run 
    python3 run_ttest_from_runs-neuclir.py $run1 runs/${retriever}.run
    for run2 in runs/${retriever}*autorerank*.run;do
        python3 run_ttest_from_runs-neuclir.py $run1 $run2
    done

    echo "-----------------------------------"
    runo=runs/${retriever}+crux_reranking:oracle.run 
    python3 run_ttest_from_runs-neuclir.py $runo runs/${retriever}.run
    for run2 in runs/${retriever}*autorerank*.run;do
        python3 run_ttest_from_runs-neuclir.py $runo $run2
    done
    python3 run_ttest_from_runs-neuclir.py $runo $run1

    echo "-----------------------------------"
done

for retriever in bm25-mds-duc04 lsr-mds-duc04 qwen3-mds-duc04;do
    # LANCER vs others
    run1=runs/${retriever}+crux_reranking.run 
    python3 run_ttest_from_runs-mds.py $run1 runs/${retriever}.run
    for run2 in runs/${retriever}*autorerank*.run;do
        python3 run_ttest_from_runs-mds.py $run1 $run2
    done
    echo "-----------------------------------"

    # LANCER vs others
    runo=runs/${retriever}+crux_reranking:oracle.run 
    python3 run_ttest_from_runs-mds.py $runo runs/${retriever}.run
    for run2 in runs/${retriever}*autorerank*.run;do
        python3 run_ttest_from_runs-mds.py $runo $run2
    done
    python3 run_ttest_from_runs-mds.py $runo $run1
    echo "-----------------------------------"
done

