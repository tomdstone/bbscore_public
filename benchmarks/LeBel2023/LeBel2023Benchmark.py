from benchmarks.LeBel2023.LeBel2023TRBenchmark import LeBel2023TRBenchmark
from benchmarks.LeBel2023.LeBel2023AudioTRBenchmark import LeBel2023AudioTRBenchmark
from benchmarks.BBS import BenchmarkScore
from data.LeBel2023 import (
    LeBel2023StimulusSet, LeBel2023Assembly,
    LeBel2023AudioStimulusSet,
)
from benchmarks import BENCHMARK_REGISTRY

# Base class for single-subject LeBel benchmarks


class LeBel2023Base(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, subject_id, debug=False, batch_size=4):
        super().__init__(
            stimulus_train_class=LeBel2023StimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=LeBel2023Assembly,
            assembly_train_kwargs={'subjects': [subject_id]},
            assembly_test_kwargs={'subjects': [subject_id]},
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


# Register benchmarks for all 8 subjects (UTS01 to UTS08)
subjects = [f"UTS{i:02d}" for i in range(1, 9)]

for subj in subjects:
    class_name = f"LeBel2023{subj}"

    # Dynamic class creation to ensuring distinct types in registry
    class DynamicBenchmark(LeBel2023Base):
        def __init__(self, model_identifier, layer_name, debug=False, batch_size=4):
            super().__init__(model_identifier, layer_name, subj, debug, batch_size)

    DynamicBenchmark.__name__ = class_name
    BENCHMARK_REGISTRY[class_name] = DynamicBenchmark

# Generic fallback (defaults to UTS01)


class LeBel2023Benchmark(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=LeBel2023StimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=LeBel2023Assembly,
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


BENCHMARK_REGISTRY["LeBel2023"] = LeBel2023Benchmark

# --- TR-level benchmarks ---

for _subj in subjects:
    _tr_class_name = f"LeBel2023TR{_subj}"

    # Capture _subj in default arg to avoid late-binding closure issue
    class _DynamicTRBenchmark(LeBel2023TRBenchmark):
        def __init__(self, model_identifier, layer_name,
                     debug=False, batch_size=None,
                     _sid=_subj):
            if batch_size is None:
                batch_size = [4]
            super().__init__(
                model_identifier, layer_name,
                subject_id=_sid,
                batch_size=batch_size,
                debug=debug
            )

    _DynamicTRBenchmark.__name__ = _tr_class_name
    BENCHMARK_REGISTRY[_tr_class_name] = _DynamicTRBenchmark

# --- Audio average benchmarks ---


class LeBel2023AudioBase(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, subject_id,
                 debug=False, batch_size=4):
        super().__init__(
            stimulus_train_class=LeBel2023AudioStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=LeBel2023Assembly,
            assembly_train_kwargs={'subjects': [subject_id]},
            assembly_test_kwargs={'subjects': [subject_id]},
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


for _subj in subjects:
    _audio_class_name = f"LeBel2023Audio{_subj}"

    class _DynamicAudioBenchmark(LeBel2023AudioBase):
        def __init__(self, model_identifier, layer_name,
                     debug=False, batch_size=4, _sid=_subj):
            super().__init__(
                model_identifier, layer_name, _sid, debug, batch_size)

    _DynamicAudioBenchmark.__name__ = _audio_class_name
    BENCHMARK_REGISTRY[_audio_class_name] = _DynamicAudioBenchmark

# --- Audio TR-level benchmarks ---

for _subj in subjects:
    _audio_tr_class_name = f"LeBel2023AudioTR{_subj}"

    class _DynamicAudioTRBenchmark(LeBel2023AudioTRBenchmark):
        def __init__(self, model_identifier, layer_name,
                     debug=False, batch_size=None,
                     _sid=_subj):
            if batch_size is None:
                batch_size = [4]
            super().__init__(
                model_identifier, layer_name,
                subject_id=_sid,
                batch_size=batch_size,
                debug=debug
            )

    _DynamicAudioTRBenchmark.__name__ = _audio_tr_class_name
    BENCHMARK_REGISTRY[_audio_tr_class_name] = _DynamicAudioTRBenchmark
