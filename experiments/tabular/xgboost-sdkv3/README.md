# XGBoost Training & Deployment with Amazon SageMaker Python SDK v3

End-to-end example of training an XGBoost model and deploying it to a real-time endpoint using the new **SageMaker Python SDK v3** unified APIs.

## Use Case: Gas Lift Optimization

Predicting oil production from well sensor data using the open-source [Petrobras 3W dataset](https://github.com/petrobras/3W).

## Quick Start

```bash
# Clone the repo
git clone https://github.com/dgallitelli/sagemaker-sdk-v3-xgboost-example.git
cd sagemaker-sdk-v3-xgboost-example

# Open the notebook
jupyter notebook notebook/gas_lift_optimization.ipynb
```

## Repository Structure

```
├── README.md
├── notebook/
│   └── gas_lift_optimization.ipynb   # Main notebook with complete workflow
├── training/
│   ├── train.py                      # Training script for SageMaker
│   └── requirements.txt              # Training dependencies (xgboost, scipy)
└── inference/
    ├── inference.py                  # Inference handlers for deployment
    └── requirements.txt              # Inference dependencies (xgboost)
```

## SDK v3 Classes Used

| Class | Purpose |
|-------|---------|
| `ModelTrainer` | Unified training API |
| `SourceCode` | Script + dependencies packaging |
| `Compute` | Instance configuration |
| `InputData` | Training data channels |
| `Model` | Model resource for deployment |
| `EndpointConfig` | Endpoint configuration |
| `Endpoint` | Real-time inference endpoint |

## Workflow

1. **Setup** - Configure SageMaker session and IAM role
2. **Prepare Data** - Load Petrobras 3W dataset, create training features
3. **Train** - Run training job on ml.m5.large (~5 min)
4. **Verify** - Download model and validate locally
5. **Deploy** - Create real-time endpoint (~5 min)
6. **Test** - Invoke endpoint with sample predictions
7. **Optimize** - Demonstrate gas allocation optimization
8. **Cleanup** - Delete endpoint to avoid charges

## Key Insight: Model Packaging for SKLearn Inference

The SKLearn container requires specific model tarball structure for deployment:

```
model.tar.gz/
├── model.joblib           # Model at root
├── feature_names.json     # Metadata at root
└── code/
    ├── inference.py       # Inference handlers
    └── requirements.txt   # Dependencies (xgboost)
```

Plus environment variables:
- `SAGEMAKER_PROGRAM`: `inference.py`
- `SAGEMAKER_SUBMIT_DIRECTORY`: `/opt/ml/model/code`

## Results

- **Training**: R² = 0.9537, RMSE = 5.48
- **Optimization**: +8.5% production improvement through optimal gas allocation

## References

- [SageMaker Python SDK v3 Documentation](https://sagemaker.readthedocs.io/)
- [Petrobras 3W Dataset](https://github.com/petrobras/3W)
- [Gas Lift Optimization](https://petrowiki.spe.org/Gas_lift)

## License

MIT
