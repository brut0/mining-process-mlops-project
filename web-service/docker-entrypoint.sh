ls -l
export FLASK_APP=api.py
export FLASK_RUN_HOST=0.0.0.0
export FLASK_RUN_PORT=80

exec gunicorn -b 0.0.0.0:80 api:app
