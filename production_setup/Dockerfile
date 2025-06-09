FROM public.ecr.aws/lambda/python:3.12

# Copy requirements file
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the dependencies
RUN pip install -r requirements.txt
RUN mkdir -p /tmp/logs
# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}
COPY main.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "main.handler" ]