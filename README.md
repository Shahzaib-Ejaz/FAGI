# Round-3 cache for branch rerun-real-dp-cicids

Trained CICIDS2017 models (weights), attack results, adversarial examples, canary and federated per-repeat caches, and run logs.

Restore at the repo root (on branch rerun-real-dp-cicids):

    git fetch origin round3-cache
    git --work-tree=/tmp/rc checkout origin/round3-cache -- .   # or download the parts
    cat /tmp/rc/fagi_round3_cache.tar.gz.part* > /tmp/fagi_round3_cache.tar.gz
    sha256sum /tmp/fagi_round3_cache.tar.gz   # compare with SHA256SUMS
    tar -xzf /tmp/fagi_round3_cache.tar.gz    # restores experiments/results/cache and logs

Safe to delete this branch once the paper is final.
