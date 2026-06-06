#!/bin/sh

set -e
exec gunicorn main:app -c gunicorn/gunicorn_config.py --preload
