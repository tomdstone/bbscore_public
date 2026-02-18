from benchmarks.BBS import BenchmarkScore
from data.BMD import (
    BMDStimulusTrainSet,
    BMDStimulusTestSet,
    BMDAssemblyV1,
    BMDAssemblyV2,
    BMDAssemblyV3,
    BMDAssemblyV4,
    BMDAssemblyIPS0,
    BMDAssemblyIPS123,
    BMDAssemblyLOC,
    BMDAssemblyPFop,
    BMDAssembly7AL,
    BMDAssemblyPFt,
    BMDAssemblyOFA,
    BMDAssemblyBA2,
    BMDAssemblyEBA,
    BMDAssemblyFFA,
    BMDAssemblyMT,
    BMDAssemblyPPA,
    BMDAssemblyRSC,
    BMDAssemblySTS,
    BMDAssemblyTOS,
    BMDAssemblyV3ab,
    BMDAssemblyV1d,
    BMDAssemblyV1v,
    BMDAssemblyV2d,
    BMDAssemblyV2v,
    BMDAssemblyV3d,
    BMDAssemblyV3v,
    BMDAssemblyBMD
)
from benchmarks import BENCHMARK_REGISTRY


class BMD_V1(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyV1,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_V1"] = BMD_V1


class BMD_V2(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyV2,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_V2"] = BMD_V2


class BMD_V3(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyV3,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_V3"] = BMD_V3


class BMD_V4(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyV4,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_V4"] = BMD_V4


class BMD_IPS123(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyIPS123,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_IPS123"] = BMD_IPS123


class BMD_IPS0(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyIPS0,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_IPS0"] = BMD_IPS0


class BMD_LOC(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyLOC,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_LOC"] = BMD_LOC


class BMD_PFop(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyPFop,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_PFop"] = BMD_PFop


class BMD_7AL(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssembly7AL,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_7AL"] = BMD_7AL


class BMD_PFt(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyPFt,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_PFt"] = BMD_PFt


class BMD_OFA(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyOFA,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_OFA"] = BMD_OFA


class BMD_BA2(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyBA2,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_BA2"] = BMD_BA2


class BMD_EBA(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyEBA,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_EBA"] = BMD_EBA


class BMD_FFA(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyFFA,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_FFA"] = BMD_FFA


class BMD_MT(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyMT,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_MT"] = BMD_MT


class BMD_RSC(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyRSC,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_RSC"] = BMD_RSC


class BMD_STS(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblySTS,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_STS"] = BMD_STS


class BMD_TOS(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyTOS,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_TOS"] = BMD_TOS


class BMD_V3ab(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyV3ab,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_V3ab"] = BMD_V3ab


class BMD_V1d(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyV1d,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_V1d"] = BMD_V1d


class BMD_V1v(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyV1v,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_V1v"] = BMD_V1v


class BMD_V2d(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyV2d,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_V2d"] = BMD_V2d


class BMD_V2v(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyV2v,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_V2v"] = BMD_V2v


class BMD_V3d(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyV3d,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_V3d"] = BMD_V3d


class BMD_V3v(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyV3v,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_V3v"] = BMD_V3v


class BMD_BMD(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=BMDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=BMDStimulusTestSet,
            assembly_class=BMDAssemblyBMD,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["BMD_BMD"] = BMD_BMD
