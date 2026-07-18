#pragma once

enum class ResidualMetric
{
    None = 0,
    SampledRelativeResidual,
    NormwiseBackwardError
};

inline const char* residual_metric_name(ResidualMetric metric)
{
    switch (metric) {
    case ResidualMetric::SampledRelativeResidual:
        return "sampled_relative_residual";
    case ResidualMetric::NormwiseBackwardError:
        return "normwise_backward_error";
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
    default:
        return nullptr;
    }
}
