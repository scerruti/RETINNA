"""API Lambda entry point (behind a Lambda function URL). Lifespan is off, so the API never
loads the model or imports torch: it only reads job records, S3 results and perimeters."""

from mangum import Mangum

from .api import app

handler = Mangum(app, lifespan="off")
