# Colab GPU bridge

This bridge lets a manually started Colab runtime execute DeepSurv jobs while
the local research loop retains control.

The shared data directory must contain only:

- `clinicalTrain.csv`
- `clinicalValidation.csv`

Never place `affyfRMATest.csv` in the Colab data directory or queue.

Submit, watch, and collect a job locally:

```bash
python3 colab_bridge/local_colab_queue.py submit \
  --queue-root "/path/to/autosota_colab_queue" \
  --job-id deepsurv_iter_001 \
  --require-pushed \
  --env DEEPSURV_ARENA_N_TRIALS=20 \
  --env DEEPSURV_ARENA_BOOTSTRAPS=2 \
  --env DEEPSURV_ARENA_EPOCHS=200
python3 colab_bridge/local_colab_queue.py watch \
  --queue-root "/path/to/autosota_colab_queue" \
  --job-id deepsurv_iter_001
python3 colab_bridge/local_colab_queue.py collect \
  --queue-root "/path/to/autosota_colab_queue" \
  --job-id deepsurv_iter_001
```

In Colab, mount Drive and start the worker:

```python
from google.colab import drive
drive.mount("/content/drive")

!python /content/drive/MyDrive/path/to/colab_gpu_worker.py \
  --queue-root /content/drive/MyDrive/autosota_colab_queue \
  --data-dir /content/drive/MyDrive/nsclc_clinical_train_valid \
  --repo-url https://github.com/osun24/autosota-nsclc-adj-chemo.git
```

Jobs move through `queued`, `running`, and `done` or `failed`. Heartbeats and
leases allow a later worker to requeue a job after a Colab disconnect.
