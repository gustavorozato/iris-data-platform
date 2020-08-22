# data-platform
In order to make predictions you need first to build your docker image which will also train the model:

```docker build -t iris_classification:latest .```

To deploy the image into a container, execute:

```docker run -p 5000:5000 iris_classification:latest```

Flask SwaggerUI server will run in: 0.0.0.0:5000

It is possible to test the application making a GET request with the 4 params to:

curl -X GET "http://0.0.0.0:5000/iris_classification/?sepal_length=3&sepal_width=3&petal_length=3&petal_width=3" -H "accept: application/json"

Architecture can be found in AWS folder.
