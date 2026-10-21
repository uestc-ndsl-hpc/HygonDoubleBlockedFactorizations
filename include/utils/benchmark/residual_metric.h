#pragma once

enum class ResidualMetric
{
    None = 0,
    SampledRelativeResidual,
    NormwiseBackwardError,
    CpuFp64RelativeResidual
};

inline const char* residual_metric_name(ResidualMetric metric)
{
    switch (metric) {
    case ResidualMetric::SampledRelativeResidual:
        return "sampled_relative_residual";
    case ResidualMetric::NormwiseBackwardError:
        return "normwise_backward_error";
    case ResidualMetric::CpuFp64RelativeResidual:
        return "cpu_fp64_full_relative_residual";
    default:
        return "none";
    }
}

inline const char* residual_metric_heading(ResidualMetric metric)
{
    switch (metric) {
    case ResidualMetric::SampledRelativeResidual:
        return "Sampled Rel. Residual";
    case ResidualMetric::NormwiseBackwardError:
        return "Normwise Backward Error";
    case ResidualMetric::CpuFp64RelativeResidual:
        return "CPU FP64 Rel. Residual";
    default:
        return "Residual";
    }
}

inline const char* residual_metric_limit_environment(ResidualMetric metric)
{
    switch (metric) {
    case ResidualMetric::SampledRelativeResidual:
        return "BENCHMARK_MAX_SAMPLED_RELATIVE_RESIDUAL";
    case ResidualMetric::NormwiseBackwardError:
        return "BENCHMARK_MAX_NORMWISE_BACKWARD_ERROR";
    case ResidualMetric::CpuFp64RelativeResidual:
        return "BENCHMARK_MAX_CPU_FP64_RELATIVE_RESIDUAL";
    default:
        return nullptr;
    }
}
