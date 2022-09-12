# Silica concentration in mining data prediction web service

All steps of test, deploy and run are automated in Makefile.
Unit test used default model in mocks folder.
Logs saves in `app.log` file using logging.

### Project structure
- **_app_**: Flask application
  - **_utils_**: Unit tests
  - **_utils_**: Logger and S3Client classes
- **_integration_**: Integration tests
- `app.py`: Flask app
- `config.py`: Configuration
- `predict.py`: Prediction service
- `test`: Script for local manual testing

### Prerequisites
1. Activate virtual environment and install requirements
    ```bash
    pipenv shell
    pipenv install
    ```
2. Set up environment variables

## Usage

### Check code quality
```bash
make quality_checks
```

### Run unit tests
```bash
make unit_test
```

### Run integration tests
```bash
make integration_test
```

### Run locally
```bash
make run
```

### Publish local build docker container to Container Registry in Cloud
```bash
make build_publish
```

### Deploy app
```bash
make deploy
```

### Prediction format
```bash
{
    "measurements_count",
    'prediction',
    "model": {
        "run_id"
    }
}
```