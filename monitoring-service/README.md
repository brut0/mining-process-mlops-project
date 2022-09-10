# Model Monitoring

## Prerequisites

You need following tools installed:
- `docker`
- `docker-compose` (included to Docker Desktop for Mac and Docker Desktop for Windows )

## Preparation

Note: all actions expected to be executed in repo folder.

- Create virtual environment and activate it `virtualenv env && source ./env/bin/activate`)
- Install required packages `pip install -r requirements.txt`

## Monitoring Example

### Starting services

To start all required services, execute:
```bash
docker-compose up
```

It will start following services:
- `prometheus` - TSDB for metrics
- `grafana` - Visual tool for metrics
- `mongo` - MongoDB, for storing raw data, predictions, targets and profile reports
- `evidently_service` - Evindently RT-monitoring service (draft example)
- `prediction_service` - main service, which makes predictions
