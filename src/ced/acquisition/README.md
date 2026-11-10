# Acquisition boundary

All implementations run on wenceslaus. `fetch` is the only command permitted to download. Builds call `locate_source`, which uses local paths, verified aria2 stores, or Hugging Face `local_files_only=True` and emits an exact fetch command when an artifact is absent.
