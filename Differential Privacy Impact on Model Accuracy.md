# Differential Privacy Impact on Model Accuracy

## Model Performance Comparison

| Model Type | R² Score | MSE | Performance Change |
|------------|----------|-----|-------------------|
| Baseline (v2) | 0.5573 | 34,435 | - |
| DP Model | -0.0855 to -0.4276 | 84,432 to 111,042 | ↓ 64-98% |

## Privacy Budget Analysis

- **Epsilon (ε)**: 0.321758
- **Privacy Level**: Strong Protection
- **Noise Multiplier**: 1.8
- **Training Epochs**: 15

## Key Findings

- **Privacy Cost**: R² decreased by 0.64-0.98 points
- **MSE Impact**: Increased by 50,000-77,000 points
- **Trade-off**: Strong privacy protection with significant accuracy loss
- **Consistent Results**: Both lakeFS and DVC experiments show similar patterns

## Conclusion

Differential privacy provides strong privacy protection (ε < 1) but comes with substantial model performance degradation in this use case.